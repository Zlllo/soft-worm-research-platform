"""Causal recurrent n-step A2C for fixed-budget closed-loop navigation.

Environment geometry is used only for scalar rewards and offline metrics.
No demonstration labels, oracle gradients or future states enter the learner.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from thermotaxis.navigation import NavigationBatch,NavigationConfig
from thermotaxis.navigation_policy import NavigationPolicy
from tracking_learning_surface import atomic_json
from navigation_preflight import summary


SOURCES=['analysis/train_navigation.py','src/thermotaxis/navigation.py','src/thermotaxis/navigation_policy.py',
    'src/thermotaxis/connectome.py','src/thermotaxis/sensory_adaptation.py']


def evaluate(model,config,seeds,phase_mode='correct'):
    env=NavigationBatch(config,seeds,phase_mode=phase_mode)
    state=model.initial(len(seeds));packet=env.initial_packet()
    with torch.no_grad():
        while True:
            mean,_,state=model(torch.from_numpy(packet),state)
            a=np.ones((len(seeds),2));a[:,:model.action_dims]=torch.tanh(mean).numpy()
            packet,_,done=env.step(a)
            if done:break
    return env.metrics()


def train_one(name,interface,encoding,seed,budgets,output,config=None,batch=32,lr=.003):
    config=config or NavigationConfig();torch.set_num_threads(1)
    torch.manual_seed(seed);model=NavigationPolicy(name,interface,encoding)
    metadata=model.metadata();torch.manual_seed(seed+90000)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    record=dict(name=name,interface=interface,encoding=encoding,seed=seed,metadata=metadata,
        endpoints={},updates=[],status='running')
    gamma=math.exp(-config.control_dt_s/30.);lam=.95
    episode=0
    def make_env(episode):
        return NavigationBatch(config,[300000+seed*100000+episode*batch+i for i in range(batch)])
    env=make_env(episode);packet=env.initial_packet();state=model.initial(batch)
    total=0;started=time.monotonic()
    stem=f'{name}-{interface}-{encoding}-{seed}'
    for budget in budgets:
        if budget%batch:raise ValueError('budgets must be multiples of batch')
        while total<budget:
            state=tuple(x.detach() for x in state)
            logps=[];values=[];rewards=[];entropies=[];done=False
            steps=min(32,(budget-total)//batch)
            for _ in range(steps):
                mean,value,state=model(torch.from_numpy(packet),state)
                dist=torch.distributions.Normal(mean,model.log_std.clamp(-2.,.5).exp())
                latent=dist.sample();action=torch.tanh(latent)
                logp=(dist.log_prob(latent)-torch.log(1-action.square()+1e-6)).sum(-1)
                a=np.ones((batch,2));a[:,:model.action_dims]=action.detach().numpy()
                packet,r,done=env.step(a)
                logps.append(logp);values.append(value);rewards.append(torch.from_numpy(r));entropies.append(dist.entropy().sum(-1))
                total+=batch
                if done:break
            with torch.no_grad():
                bootstrap=torch.zeros(batch) if done else model(torch.from_numpy(packet),state)[1]
            advantage=torch.zeros(batch);advantages=[]
            next_value=bootstrap
            for i in reversed(range(len(rewards))):
                delta=rewards[i]+gamma*next_value-values[i].detach()
                advantage=delta+gamma*lam*advantage
                advantages.append(advantage);next_value=values[i].detach()
            advantages=torch.stack(list(reversed(advantages)))
            value_tensor=torch.stack(values);targets=advantages+value_tensor.detach()
            normalized=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-6)
            policy_loss=-(torch.stack(logps)*normalized).mean()
            value_loss=(value_tensor-targets).square().mean()
            loss=policy_loss+.5*value_loss-.005*torch.stack(entropies).mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite A2C loss')
            optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError('nonfinite A2C gradient')
            optimizer.step()
            record['updates'].append(dict(interactions=total,loss=float(loss.detach()),policy_loss=float(policy_loss.detach()),
                value_loss=float(value_loss.detach()),mean_reward=float(torch.stack(rewards).mean()),completed_episodes=episode*batch+(batch if done else 0)))
            if done:
                episode+=1;env=make_env(episode);packet=env.initial_packet();state=model.initial(batch)
        checkpoint=stem+f'-N{budget}.pt';torch.save(model.state_dict(),output/checkpoint)
        record['endpoints'][str(budget)]=dict(checkpoint=checkpoint,environment_interactions=total,
            physical_interaction_s=total*config.control_dt_s,completed_episodes=episode*batch)
        atomic_json(output/(stem+'-training.json'),record)
        print('A2C',stem,total,'seconds',round(time.monotonic()-started,1),flush=True)
    record['status']='complete';record['wall_time_s']=time.monotonic()-started
    atomic_json(output/(stem+'-training.json'),record)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--name',default='gru8');parser.add_argument('--interface',default='full')
    parser.add_argument('--encoding',default='raw');parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--budgets',nargs='+',type=int,default=[16384,65536]);args=parser.parse_args()
    if args.output.exists():raise RuntimeError('unused output required')
    args.output.mkdir(parents=True)
    config=NavigationConfig()
    plan=dict(status='development_trainability_preflight',name=args.name,interface=args.interface,encoding=args.encoding,
        seed=args.seed,budgets=args.budgets,config=asdict(config),algorithm='recurrent n-step A2C, horizon32, batch32',
        sources={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in SOURCES},
        validation_seeds=list(range(42001,42017)),test_access='development cases retired after inspection')
    atomic_json(args.output/'design.json',plan)
    for n in SOURCES:
        p=args.output/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,p)
    row=train_one(args.name,args.interface,args.encoding,args.seed,args.budgets,args.output,config)
    results=[]
    for budget in args.budgets:
        model=NavigationPolicy(args.name,args.interface,args.encoding)
        model.load_state_dict(torch.load(args.output/row['endpoints'][str(budget)]['checkpoint'],map_location='cpu',weights_only=True));model.eval()
        cases=evaluate(model,config,plan['validation_seeds'])
        results.append(dict(interactions=budget,cases=cases,summary=summary(cases)))
        print('DEV',budget,results[-1]['summary'],flush=True)
    atomic_json(args.output/'results.json',dict(design=plan,training=row,runs=results))


if __name__=='__main__':main()
