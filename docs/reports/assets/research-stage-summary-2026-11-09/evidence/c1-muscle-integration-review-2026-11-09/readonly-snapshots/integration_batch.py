"""Run exactly the preregistered isolated cases, at most two workers at once."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]/'results/c1-muscle-integration-isolation-2026-11-09'


def main(stage):
    p=json.loads((ROOT/'protocol.json').read_text())
    jobs=[(case,variant,dt) for case in (['gates','pool'] if stage=='controls' else ['main'])
        for variant in (p['controls']['variants'] if stage=='controls' else p['variants']) for dt in p['dt_ms']]
    assert len(jobs)==12
    status={};target=ROOT/f'batch-{stage}.json'
    def launch(job):
        case,variant,dt=job;name=f'{case}_{variant}_dt{dt:g}'
        if (ROOT/'runs'/name/'record.json').exists():return name,dict(status='already_completed')
        command=[sys.executable,str(Path(__file__).with_name('integration_run.py')),'--case',case,'--variant',variant,'--dt',str(dt)]
        with (ROOT/f'{name}.log').open('w') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'}).returncode
        return name,dict(status='complete' if code==0 else 'failed',exit_code=code)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(launch,j) for j in jobs]
        for future in as_completed(futures):
            name,result=future.result();status[name]=result
            target.write_text(json.dumps(dict(stage=stage,declared_jobs=12,status=status),indent=2)+'\n')
            print(name,result['status'],flush=True)
    assert all(x['status'] in ['complete','already_completed'] for x in status.values())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['controls','main']);main(p.parse_args().stage)
