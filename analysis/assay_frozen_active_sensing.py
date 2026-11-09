"""Planned acute reference-mismatch assay, separately from refitted decoders."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from run_active_sensing_stage3 import CONDITIONS,check,make_features,metric
from diagnose_navigation_failures import write
from thermotaxis.navigation_window_policy import WindowActorCritic


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/active-sensing-stage3-2026-11-03'))
    a=p.parse_args();root=a.root;torch.set_num_threads(1)
    d=json.loads((root/'design.json').read_text());check(d)
    with np.load(root/'dataset.npz') as z:data={k:z[k] for k in z.files}
    ids=np.flatnonzero(data['split']==2);rows=[]
    hashes={}
    for seed in d['generic']['seeds']:
        path=root/f'gru-paired-{seed}.pt';hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        model=WindowActorCritic('gru16','full');model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
        for condition in CONDITIONS:
            score=[]
            with torch.no_grad():
                for start in range(0,len(ids),1024):
                    score.append(model.actor_mean(torch.from_numpy(make_features(data,ids[start:start+1024],condition))).numpy())
            score=np.concatenate(score);noise=data['noise'][ids];y=data['label'][ids];r=data['run'][ids]
            rows.append(dict(seed=seed,condition=condition,test=metric(y,score,r),
                test_noisy=metric(y[noise>0],score[noise>0],r[noise>0])))
            assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[path.name]
    write(root/'frozen-paired-transfer.json',dict(status='planned_acute_assay_distinct_from_refitting',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_sha256=hashes,rows=rows,complete=True))
    print([(r['seed'],r['condition'],r['test']['balanced_accuracy']) for r in rows])


if __name__=='__main__':main()
