"""Bridge the immutable old four-second-window models to the new task."""
from pathlib import Path
import json
import time

import numpy as np
import torch

from compare_tracking_topology import make_model
from navigation_teacher import data
from navigation_preflight import summary
from thermotaxis.navigation import NavigationBatch,NavigationConfig


class WindowBatch:
    def __init__(self,batch):self.values=np.zeros((batch,41,4),dtype=np.float32)
    def observe(self,packet):
        for sample in packet:
            valid=sample[:,4]>0
            if not valid.any():continue
            shifted=np.roll(self.values,-1,axis=1)
            shifted[:,-1]=np.stack([(sample[:,0]-293.15)/.05,sample[:,1],sample[:,2],sample[:,4]],axis=1)
            self.values=np.where(valid[:,None,None],shifted,self.values)
        return self.values


def window_data():
    cache=Path('results/navigation-window-teacher-2026-10-25.npz')
    if not cache.exists():
        original=data();windows=WindowBatch(original['x'].shape[2]);xs=[]
        for packet in original['x']:xs.append(windows.observe(packet).copy())
        values=np.stack(xs)
        np.savez_compressed(cache,x=values.reshape(-1,41,4),y=original['y'].reshape(-1),
            original_seeds=original['seeds'])
    with np.load(cache,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def evaluate_window(model,config,seeds,phase_mode='correct'):
    env=NavigationBatch(config,seeds,phase_mode=phase_mode);windows=WindowBatch(len(seeds));packet=env.initial_packet()
    model.eval()
    with torch.no_grad():
        while True:
            x=windows.observe(packet)
            action=model(torch.from_numpy(x)).numpy()/.3
            actions=np.ones((len(seeds),2));actions[:,0]=action
            packet,_,done=env.step(actions)
            if done:break
    return env.metrics()


def fit_window(name,interface,seed,updates,output):
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=make_model(name,'demeaned',interface);train=window_data()
    optimizer=torch.optim.Adam(model.parameters(),lr=.003);rng=np.random.default_rng(seed);curves=[]
    start=time.monotonic();model.train()
    for update in range(updates):
        index=rng.integers(0,len(train['y']),64)
        prediction=model(torch.from_numpy(train['x'][index]))/.3
        target=torch.from_numpy(train['y'][index]);loss=(prediction-target).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite window imitation loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if update%128==0 or update+1==updates:curves.append(dict(update=update+1,minibatch_mse=float(loss.detach())))
    file=f'{name}-{interface}-{seed}-U{updates}.pt';torch.save(model.state_dict(),output/file)
    return dict(name=name,interface=interface,seed=seed,updates=updates,checkpoint=file,curves=curves,
        demonstration_interactions=len(train['y']),seconds=time.monotonic()-start),model


if __name__=='__main__':
    root=Path('results/navigation-window-preflight-2026-10-25');root.mkdir(exist_ok=True)
    row,model=fit_window('gru16','full',7,2048,root)
    rows=evaluate_window(model,NavigationConfig(),list(range(42001,42017)))
    result=dict(status='development_old_model_new_task_trainability',training=row,summary=summary(rows),rows=rows)
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(result['summary'],flush=True)
