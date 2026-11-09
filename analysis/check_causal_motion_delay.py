"""Supplemental continuous-delay check after the circular-history assay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_navigation_failures import write
from run_active_sensing_stage3 import BODY,causal_motion,check,make_features,metric
from thermotaxis.navigation_window_policy import WindowActorCritic


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/active-sensing-stage3-2026-11-03'))
    a=p.parse_args();root=a.root;torch.set_num_threads(1)
    d=json.loads((root/'design.json').read_text());check(d)
    meta=json.loads((root/'dataset.json').read_text());clock=json.loads((BODY/'baseline-clock.json').read_text())
    delay_steps=round(clock['period_s']/4/.1)
    declaration=dict(status='supplemental_after_circular_mismatch_results',
        purpose='check whether reference misalignment also matters for a physically continuous causal sensor delay',
        delay_rule='quarter nominal gait period rounded to the 0.1s observation clock',delay_s=delay_steps*.1,
        train=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(root/'causal-delay-design.json',declaration)
    with np.load(root/'dataset.npz') as z:data={k:z[k] for k in z.files}
    ids=np.flatnonzero(data['split']==2);x=make_features(data,ids,'paired');delayed=np.empty_like(x[:,:,4:6])
    scale=np.array(meta['motion_scale'])
    for run in np.unique(data['run'][ids]):
        row=meta['runs'][int(run)]
        with np.load(BODY/row['path']/'trajectory.npz') as z:m=(causal_motion(z['pose'][::5],.1)/scale).astype('float32')
        lookup={value.tobytes():j for j,value in enumerate(m)}
        positions=np.flatnonzero(data['run'][ids]==run)
        for i in positions:
            j=lookup[data['motion'][ids[i],-1].tobytes()]
            if not np.array_equal(m[j-40:j+1],data['motion'][ids[i]]):raise RuntimeError('could not align exact source window')
            if j-40-delay_steps<0:raise RuntimeError('delay would require missing prehistory')
            delayed[i]=m[j-40-delay_steps:j+1-delay_steps]
    actual=x[:,:,4:6].copy();x[:,:,4:6]=delayed
    rows=[]
    for seed in d['generic']['seeds']:
        model=WindowActorCritic('gru16','full')
        model.load_state_dict(torch.load(root/f'gru-paired-{seed}.pt',map_location='cpu',weights_only=True));model.eval()
        scores=[]
        with torch.no_grad():
            for i in range(0,len(x),1024):scores.append(model.actor_mean(torch.from_numpy(x[i:i+1024])).numpy())
        scores=np.concatenate(scores)
        rows.append(dict(seed=seed,test=metric(data['label'][ids],scores,data['run'][ids])))
    result=dict(design=declaration,rows=rows,mean_accuracy=float(np.mean([r['test']['balanced_accuracy'] for r in rows])),
        delayed_to_actual_rms_ratio=(np.sqrt(np.mean(delayed.astype('float64')**2,axis=(0,1)))/np.sqrt(np.mean(actual.astype('float64')**2,axis=(0,1)))).tolist(),
        source_windows_matched_exactly=True,complete=True)
    write(root/'causal-delay.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
