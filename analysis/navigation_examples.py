"""Fixed example cases and models declared before main heldout evaluation."""
import json
from pathlib import Path

import numpy as np
import torch

from thermotaxis.navigation import NavigationBatch,NavigationConfig
from thermotaxis.navigation_window_policy import NavigationFeatures,WindowActorCritic


ROOT=Path('results/navigation-factorial-2026-10-25')
OUT=Path('results/navigation-examples-2026-10-25')
SEEDS=[61001,61002]
MODELS=['real8','rewired8-11','gru16']


def main():
    torch.set_num_threads(1);plan=json.loads((ROOT/'design.json').read_text())
    for s in plan['specs']:assert (ROOT/f'fit-{s["id"]:03d}'/'record.json').exists()
    OUT.mkdir(exist_ok=True);index=[]
    for name in MODELS:
        spec=next(s for s in plan['specs'] if s['name']==name and s['seed']==7 and s['interface']=='full' and s['encoding']=='raw')
        record=json.loads((ROOT/f'fit-{spec["id"]:03d}'/'record.json').read_text())
        if record['status']!='complete':index.append(dict(name=name,status='failed_fit'));continue
        model=WindowActorCritic(name,'full');model.load_state_dict(torch.load(ROOT/f'fit-{spec["id"]:03d}'/record['endpoints']['262144']['checkpoint'],map_location='cpu',weights_only=True));model.eval()
        env=NavigationBatch(NavigationConfig(),SEEDS);features=NavigationFeatures(2,'raw');packet=env.initial_packet()
        with torch.no_grad():
            while True:
                x=features.observe(packet);a=np.ones((2,2));a[:,0]=torch.tanh(model.actor_mean(torch.from_numpy(x))).numpy()
                packet,_,done=env.step(a)
                if done:break
        path=OUT/(name+'.npz');np.savez_compressed(path,history=np.stack(env.history),angle=env.angle,sign=env.sign,event=env.event,fields=env.fields)
        index.append(dict(name=name,status='complete',spec=spec,cases=env.metrics(),path=path.name))
    (OUT/'index.json').write_text(json.dumps(dict(status='illustrative_replays_not_new_independent_cases',predeclared_seeds=SEEDS,models=MODELS,runs=index),indent=2)+'\n')


if __name__=='__main__':main()
