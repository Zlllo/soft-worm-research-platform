"""Bounded two-worker orchestration; no retry or parameter-search path."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]/'results/full302-joint-numerics-2026-11-12'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')


def main(stage):
    assert read(ROOT/'preflight.json')['passed'] and read(ROOT/'generated-code-validation.json')['passed']
    assert sha(ROOT/'protocol.json')==read(ROOT/'execution-lock.json')['protocol_sha256']
    if stage=='initial':jobs=[('off',.025),('live',.025),('off',.0125),('live',.0125)]
    else:
        decision=read(ROOT/'refinement-decision.json');assert decision['fine_required']
        assert decision['protocol_sha256']==sha(ROOT/'protocol.json')
        assert len(list(ROOT.glob('*_dt*/spec.json')))==4
        jobs=[('off',.00625),('live',.00625)]
    assert not (ROOT/f'batch-{stage}.json').exists(), 'No stage retries'
    assert all(not (ROOT/f'{mode}_dt{dt:g}').exists() for mode,dt in jobs)
    status={}
    def launch(job):
        mode,dt=job;name=f'{mode}_dt{dt:g}'
        with (ROOT/f'{name}.log').open('w') as log:
            code=subprocess.run([sys.executable,str(Path(__file__).with_name('joint_run.py')),'--feedback',mode,'--dt',str(dt)],
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'}).returncode
        return name,dict(exit_code=code,complete=code==0,failed_condition_preserved=code!=0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(launch,j) for j in jobs]):
            name,result=future.result();status[name]=result
            write(ROOT/f'batch-{stage}.json',dict(stage=stage,planned_conditions=len(jobs),status=status,retries=0))
            print(name,result,flush=True)
    assert len(list(ROOT.glob('*_dt*/spec.json')))<=6
    # A numerical failure is evidence, not an invitation to retry this stage.
    print(json.dumps(dict(stage=stage,completed=len(status),failed=sum(not q['complete'] for q in status.values()),retries=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initial','fine']);main(p.parse_args().stage)
