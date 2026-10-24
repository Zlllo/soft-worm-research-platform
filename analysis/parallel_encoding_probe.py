"""Resume the frozen encoding experiment with independent subprocesses.

Each child executes the unchanged original experiment script for exactly one
predeclared case. The parent alone merges results. Budgets, inputs, optimizer,
random streams and original source snapshots are unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def key(spec):
    return tuple(spec[k] for k in ('architecture','encoding','topology_seed','seed'))


def stem(spec):
    return f'{spec["architecture"]}-{spec["encoding"]}-topo{spec["topology_seed"]}-seed{spec["seed"]}'


def atomic_json(path, value):
    temp = path.with_suffix('.writing')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def verify_sources(design):
    for name, sha in design['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f'frozen experiment source changed: {name}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT/'results/encoding-topology-2026-10-24')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.workers<1 or args.workers>4:
        parser.error('use 1 to 4 worker processes')
    directory=args.root.resolve()
    design=json.loads((directory/'design.json').read_text())
    report=json.loads((directory/'results.json').read_text())
    if report['design']!=design:
        raise RuntimeError('results/design mismatch')
    verify_sources(design)
    planned={key(x):x for x in design['runs']}
    if len(planned)!=len(design['runs']):
        raise RuntimeError('duplicate planned cases')
    completed={key(x):x for x in report['runs']}
    if len(completed)!=len(report['runs']) or not set(completed)<=set(planned):
        raise RuntimeError('duplicate or unexpected completed cases')
    if not (directory/'sequential_results.json').exists():
        atomic_json(directory/'sequential_results.json',report)
    scheduler=directory/'source/analysis/parallel_encoding_probe.py'
    scheduler.parent.mkdir(parents=True,exist_ok=True)
    scheduler.write_bytes(Path(__file__).read_bytes())
    execution=dict(strategy='unchanged original CLI in isolated per-case subprocesses',workers=args.workers,
        initial_completed_cases=[list(k) for k in completed],planned_cases=len(planned),
        scheduler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_sources_verified=True,completed_cases=len(completed))
    atomic_json(directory/'execution.json',execution)
    shards=directory/'shards'
    shards.mkdir(exist_ok=True)

    def run_case(spec):
        verify_sources(design)
        child=shards/stem(spec)
        result=child/'results.json'
        if result.exists():
            previous=json.loads(result.read_text())
            if len(previous['runs'])==1 and key(previous['runs'][0])==key(spec):
                run=previous['runs'][0]
                if run['curves'][-1]['update']==design['updates']:
                    return dict(run,artifact_directory=str(child.relative_to(directory)))
        if (child/'design.json').exists():
            raise RuntimeError(f'incomplete shard must be inspected before restart: {child}')
        log=shards/(stem(spec)+'.log')
        command=[sys.executable,str(ROOT/'analysis/encoding_topology_probe.py'),
            '--updates',str(design['updates']),'--seeds',str(spec['seed']),
            '--topology-seeds',str(spec['topology_seed']),'--architectures',spec['architecture'],
            '--encodings',spec['encoding'],'--output',str(child)]
        environment=os.environ.copy()
        environment['PYTHONPATH']=str(ROOT/'src')+os.pathsep+environment.get('PYTHONPATH','')
        with log.open('a') as output:
            process=subprocess.run(command,cwd=ROOT,env=environment,stdout=output,stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'child failed: {spec}, see {log}')
        run=json.loads(result.read_text())['runs'][0]
        if key(run)!=key(spec) or run['curves'][-1]['update']!=design['updates']:
            raise RuntimeError('child did not complete prescribed case/budget')
        return dict(run,artifact_directory=str(child.relative_to(directory)))

    remaining=[spec for k,spec in planned.items() if k not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(run_case,spec):spec for spec in remaining}
        for future in as_completed(jobs):
            run=future.result()
            completed[key(run)]=run
            report['runs']=[completed[k] for k in planned if k in completed]
            report['execution']=execution
            execution['completed_cases']=len(completed)
            atomic_json(directory/'results.json',report)
            atomic_json(directory/'execution.json',execution)
            print(f'{len(completed)}/{len(planned)} {stem(run)} residence={run["behavior_validation"]["residence_fraction"]:.4f}',flush=True)
    verify_sources(design)
    if len(completed)!=len(planned):
        raise RuntimeError('not all planned cases completed')
    print('COMPLETE',directory,flush=True)


if __name__=='__main__':
    main()
