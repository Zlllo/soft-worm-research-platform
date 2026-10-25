"""Common demonstration initialization followed by measured closed-loop PPO.

Actors retain four-second causal windows. A common independent critic sees
exactly actor features, never geometry. Total data budget includes teacher
interaction, reported separately from new learner interaction.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from thermotaxis.navigation import NavigationBatch,NavigationConfig
from thermotaxis.navigation_window_policy import NavigationFeatures,WindowActorCritic
from navigation_teacher import data
from navigation_preflight import summary
from tracking_learning_surface import atomic_json


CONFIG=NavigationConfig()


def feature_data(encoding):
    path=Path(f'results/navigation-feature-teacher-{encoding}-2026-10-25.npz')
    if not path.exists():
        original=data();features=NavigationFeatures(original['x'].shape[2],encoding);xs=[]
        for packet in original['x']:xs.append(features.observe(packet))
        np.savez_compressed(path,x=np.stack(xs).reshape(-1,41,7),y=original['y'].reshape(-1),seeds=original['seeds'])
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def fit_bc(model,train,seed,updates=2048):
    rng=np.random.default_rng(seed);optimizer=torch.optim.Adam(model.parameters(),lr=.003);curves=[]
    model.train()
    for step in range(updates):
        idx=rng.integers(0,len(train['y']),64);x=torch.from_numpy(train['x'][idx]);target=torch.from_numpy(train['y'][idx])
        loss=(torch.tanh(model.actor_mean(x))-target).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite controlled-window BC')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%128==0 or step+1==updates:curves.append(dict(update=step+1,minibatch_mse=float(loss.detach())))
    return curves


def evaluate(model,encoding,config,seeds,phase_mode='correct'):
    env=NavigationBatch(config,seeds,phase_mode=phase_mode);features=NavigationFeatures(len(seeds),encoding);packet=env.initial_packet()
    model.eval()
    with torch.no_grad():
        while True:
            obs=features.observe(packet);mean=model.actor_mean(torch.from_numpy(obs))
            action=np.ones((len(seeds),2));action[:,0]=torch.tanh(mean).numpy()
            packet,_,done=env.step(action)
            if done:break
    return env.metrics()


def fit_rl(model,encoding,seed,budgets,out,stem,config=CONFIG,batch=32):
    optimizer=torch.optim.Adam(model.parameters(),lr=.0003);rng=np.random.default_rng(seed+70000)
    torch.manual_seed(seed+90000);gamma=np.exp(-config.control_dt_s/30.);lam=.95
    epoch=0;total=0;updates=[];endpoints={};started=time.monotonic()
    def reset(epoch):
        env=NavigationBatch(config,[700000+seed*100000+epoch*batch+i for i in range(batch)])
        features=NavigationFeatures(batch,encoding)
        return env,features,features.observe(env.initial_packet())
    env,features,observation=reset(epoch)
    for budget in budgets:
        while total<budget:
            observations=[];latents=[];old_logps=[];old_values=[];rewards=[];done=False
            for _ in range(min(64,(budget-total)//batch)):
                x=torch.from_numpy(observation)
                with torch.no_grad():
                    mean,value=model(x);dist=torch.distributions.Normal(mean,model.log_std.clamp(-2.,.5).exp())
                    latent=dist.sample();action=torch.tanh(latent)
                    logp=dist.log_prob(latent)-torch.log(1-action.square()+1e-6)
                a=np.ones((batch,2));a[:,0]=action.numpy()
                packet,r,done=env.step(a)
                observations.append(x);latents.append(latent);old_logps.append(logp);old_values.append(value);rewards.append(torch.from_numpy(r))
                observation=features.observe(packet);total+=batch
                if done:break
            with torch.no_grad():bootstrap=torch.zeros(batch) if done else model(torch.from_numpy(observation))[1]
            advantage=torch.zeros(batch);advantages=[];next_value=bootstrap
            for i in reversed(range(len(rewards))):
                delta=rewards[i]+gamma*next_value-old_values[i]
                advantage=delta+gamma*lam*advantage;advantages.append(advantage);next_value=old_values[i]
            adv=torch.stack(list(reversed(advantages))).reshape(-1)
            values=torch.stack(old_values).reshape(-1);targets=adv+values
            adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-6)
            xs=torch.stack(observations).reshape(-1,41,7);us=torch.stack(latents).reshape(-1);old=torch.stack(old_logps).reshape(-1)
            for _ in range(2):
                indices=rng.permutation(len(xs))
                for start in range(0,len(xs),512):
                    ix=indices[start:start+512];mean,value=model(xs[ix]);dist=torch.distributions.Normal(mean,model.log_std.clamp(-2.,.5).exp())
                    action=torch.tanh(us[ix]);logp=dist.log_prob(us[ix])-torch.log(1-action.square()+1e-6)
                    ratio=(logp-old[ix]).exp()
                    actor=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean()
                    clipped_value=values[ix]+(value-values[ix]).clamp(-.2,.2)
                    critic=torch.maximum((value-targets[ix]).square(),(clipped_value-targets[ix]).square()).mean()
                    loss=actor+.5*critic-.001*dist.entropy().mean()
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite PPO loss')
                    optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(norm):raise RuntimeError('nonfinite PPO gradient')
                    optimizer.step()
            updates.append(dict(learner_interactions=total,actor_loss=float(actor.detach()),critic_loss=float(critic.detach()),
                reward_mean=float(torch.stack(rewards).mean()),completed_episodes=epoch*batch+(batch if done else 0)))
            if done:epoch+=1;env,features,observation=reset(epoch)
        filename=f'{stem}-RL{budget}.pt';torch.save(model.state_dict(),out/filename)
        endpoints[str(budget)]=dict(checkpoint=filename,learner_interactions=total,learner_physical_s=total*config.control_dt_s)
        print('PPO',stem,budget,'sec',round(time.monotonic()-started,1),flush=True)
    return dict(endpoints=endpoints,updates=updates,wall_time_s=time.monotonic()-started)


def preflight():
    root=Path('results/navigation-window-ppo-preflight-2026-10-25')
    if root.exists():raise RuntimeError('preflight output exists')
    root.mkdir(parents=True);torch.set_num_threads(1);torch.manual_seed(7)
    plan=dict(status='development_controlled_window_ppo',name='gru16',encoding='raw',seed=7,bc_updates=2048,
        rl_budgets=[65536,262144],validation_seeds=list(range(42001,42017)),config=asdict(CONFIG))
    atomic_json(root/'design.json',plan)
    model=WindowActorCritic('gru16','full');curves=fit_bc(model,feature_data('raw'),7)
    torch.save(model.state_dict(),root/'BC.pt')
    result=dict(design=plan,bc_curves=curves,runs=[])
    rows=evaluate(model,'raw',CONFIG,plan['validation_seeds']);result['runs'].append(dict(learner_interactions=0,summary=summary(rows),rows=rows));print('BC',summary(rows),flush=True)
    fitted=fit_rl(model,'raw',7,plan['rl_budgets'],root,'gru16')
    for budget,endpoint in fitted['endpoints'].items():
        model.load_state_dict(torch.load(root/endpoint['checkpoint'],map_location='cpu',weights_only=True))
        rows=evaluate(model,'raw',CONFIG,plan['validation_seeds']);result['runs'].append(dict(learner_interactions=int(budget),summary=summary(rows),rows=rows));print('DEV PPO',budget,summary(rows),flush=True)
    result['rl']=fitted;atomic_json(root/'results.json',result)


if __name__=='__main__':preflight()
