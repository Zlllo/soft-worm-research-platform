"""Bounded scheduler for exactly the eight preregistered full runs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'results/b-motor-feedback-causality-2026-11-09'
TASKS=[('s13_live_pert',13,True,None,'live'), ('s29_baseline',29,False,None,'live'),
       ('s13_replay_base_pert',13,True,'s13_baseline','live'),
       ('s13_off_pert',13,True,None,'off'), ('s13_replay_pert',13,True,'s13_live_pert','live'),
       ('s29_live_pert',29,True,None,'live'), ('s29_replay_base_pert',29,True,'s29_baseline','live'),
       ('s29_replay_pert',29,True,'s29_live_pert','live')]

def main():
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
             PYTHONPATH=str(ROOT/'runtime/src'))
    children={}; started=time.monotonic()
    while True:
        completed={name for name,*_ in TASKS if (ROOT/name/'record.json').exists()}
        for name in completed:
            assert json.loads((ROOT/name/'record.json').read_text())['complete']
        for name,p in children.items():
            if p.poll() is not None and p.returncode:
                raise RuntimeError(f'{name} exited {p.returncode}; inspect log; never auto-rerun')
        if len(completed)==8: break
        active={name for name,*_ in TASKS if (ROOT/name).exists() and name not in completed}
        for name,seed,perturb,replay,feedback in TASKS:
            if len(active)>=3: break
            if name in completed or name in active: continue
            if replay and not (ROOT/replay/'record.json').exists(): continue
            args=[sys.executable,str(Path(__file__).with_name('feedback_run.py')),'--name',name,'--seed',str(seed),'--feedback',feedback]
            if perturb:args+=['--perturb']
            if replay:args+=['--replay',replay]
            with (ROOT/(name+'.log')).open('w') as log:
                children[name]=subprocess.Popen(args,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            active.add(name)
            print('started',name,flush=True)
        if time.monotonic()-started>7200: raise TimeoutError('scheduler exceeded two hours')
        time.sleep(2)
    for p in children.values():p.wait()
    record=dict(new_full_runs=8,completed=sorted(completed),max_concurrent=3,auto_reruns=0)
    (ROOT/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
