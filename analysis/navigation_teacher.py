"""Causal local-reference data and burn-in imitation trainability checks.

Demonstration interactions are counted, never confused with learner rollouts.
Short recurrent chunks use burn-in; threshold state is recomputed only from
preceding observed temperatures, never from oracle/reference internal states.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from thermotaxis.navigation import NavigationBatch,NavigationConfig
from thermotaxis.isothermal_scene import SweepReading,PhaseRegressionController,RegressionControlConfig
from thermotaxis.navigation_policy import NavigationPolicy
from thermotaxis.sensory_adaptation import ThresholdAdaptation
from train_navigation import evaluate
from navigation_preflight import summary


CACHE=Path('results/navigation-teacher-2026-10-25.npz')


def collect(seeds):
    c=NavigationConfig();env=NavigationBatch(c,seeds)
    controls=[PhaseRegressionController(RegressionControlConfig(preferred_K=c.preferred_K,comfort_scale_K=c.comfort_K,
        speed_m_s=c.speed_m_s,head_forward_offset_m=c.length_m/2,head_lateral_amplitude_m=c.sweep_m,
        window_s=2/c.frequency_Hz,max_turn_rate_rad_s=c.turn_limit_rad_s)) for _ in seeds]
    packet=env.initial_packet();xs=[];ys=[];thresholds=[];flags=[]
    threshold=np.zeros(len(seeds));initialized=np.zeros(len(seeds),dtype=bool);sense=ThresholdAdaptation()
    while True:
        thresholds.append(threshold.copy());flags.append(initialized.copy());xs.append(packet.copy())
        for k,sample in enumerate(packet):
            valid=sample[:,4]>0
            level=sample[:,0]-c.preferred_K
            before=np.where(initialized,threshold,level)
            threshold=np.where(valid,sense.update(level,before,c.sensor_dt_s),threshold);initialized|=valid
            for i,controller in enumerate(controls):
                if valid[i]:controller.observe(SweepReading(env.time-(len(packet)-1-k)*c.sensor_dt_s,
                    float(sample[i,0]),float(np.arctan2(sample[i,1],sample[i,2]))))
        a=np.ones((len(seeds),2));a[:,0]=[v.command(env.time)/c.turn_limit_rad_s for v in controls]
        ys.append(a[:,0].copy());packet,_,done=env.step(a)
        if done:break
    return dict(x=np.stack(xs),y=np.stack(ys).astype(np.float32),threshold=np.stack(thresholds).astype(np.float32),
        initialized=np.stack(flags),seeds=np.asarray(seeds),config=np.array(json.dumps(asdict(c))))


def data():
    if not CACHE.exists():
        values=collect(list(range(51001,51033)));np.savez_compressed(CACHE,**values)
    with np.load(CACHE,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def fit_bc(model,train,seed,updates=2048,batch=16,burn=32,length=32):
    optimizer=torch.optim.Adam(model.parameters(),lr=.003);rng=np.random.default_rng(seed)
    curves=[];model.train();started=time.monotonic()
    for update in range(updates):
        indices=rng.integers(0,train['x'].shape[2],batch)
        start=int(rng.integers(0,len(train['x'])-burn-length+1))
        initial=model.initial(batch)
        state=(initial[0],torch.from_numpy(train['threshold'][start,indices]),torch.from_numpy(train['initialized'][start,indices]))
        with torch.no_grad():
            for t in range(start,start+burn):
                packet=torch.from_numpy(train['x'][t,:,indices].transpose(1,0,2).copy())
                _,_,state=model(packet,state)
        state=tuple(v.detach() for v in state);loss=0.
        for t in range(start+burn,start+burn+length):
            packet=torch.from_numpy(train['x'][t,:,indices].transpose(1,0,2).copy())
            mean,_,state=model(packet,state)
            target=torch.from_numpy(train['y'][t,indices])
            loss=loss+(torch.tanh(mean[:,0])-target).square().mean()/length
        if not torch.isfinite(loss):raise RuntimeError('nonfinite navigation imitation loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if update%128==0 or update+1==updates:curves.append(dict(update=update+1,minibatch_mse=float(loss.detach())))
    model.eval();return dict(updates=updates,curves=curves,wall_time_s=time.monotonic()-started,
        demonstration_interactions=int(np.prod(train['y'].shape)),teacher_physical_s=float(np.prod(train['y'].shape)*.2))


if __name__=='__main__':
    root=Path('results/navigation-bc-preflight-2026-10-25');root.mkdir(exist_ok=True)
    torch.set_num_threads(1);train=data();torch.manual_seed(7)
    model=NavigationPolicy('gru8','full','raw')
    record=fit_bc(model,train,7,updates=2048)
    torch.save(model.state_dict(),root/'gru8-full-raw-7.pt')
    rows=evaluate(model,NavigationConfig(),list(range(42001,42017)))
    result=dict(status='development_imitation_trainability',training=record,rows=rows,summary=summary(rows))
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(result['summary'],flush=True)
