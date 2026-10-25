"""Preregistered bounded factorial after retired trainability pilots.

Fit and heldout evaluation are separate global stages. Parallel workers own
unique rows; no model-dependent early stopping, retry tuning or checkpoint
selection. This is BC-initialized PPO, not from-scratch RL sample complexity.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from navigation_window_ppo import feature_data,fit_bc,fit_rl,evaluate,CONFIG
from thermotaxis.navigation_window_policy import WindowActorCritic
from tracking_learning_surface import atomic_json
from navigation_preflight import reference,summary


ROOT=Path('results/navigation-factorial-2026-10-25')
SOURCES=['analysis/navigation_factorial.py','analysis/navigation_window_ppo.py','analysis/navigation_teacher.py',
    'analysis/navigation_preflight.py','analysis/tracking_learning_surface.py',
    'src/thermotaxis/navigation.py','src/thermotaxis/navigation_window_policy.py',
    'src/thermotaxis/sensory_adaptation.py','src/thermotaxis/connectome.py','src/thermotaxis/isothermal_scene.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specifications():
    models=['real8']+['rewired8-'+str(s) for s in [11,22,33,44,55,66,77,88,99,110]]+['isolated8']
    specs=[]
    for encoding in ['raw','adaptive']:
        included=models if encoding=='raw' else ['real8','rewired8-11','rewired8-44','isolated8']
        for name in included:
            for interface in ['full','restricted']:
                for seed in [7,17,27]:specs.append(dict(name=name,interface=interface,encoding=encoding,seed=seed))
        for name in (['gru2','gru16'] if encoding=='raw' else ['gru16']):
            for seed in [7,17,27]:specs.append(dict(name=name,interface='full',encoding=encoding,seed=seed))
    for i,s in enumerate(specs):s['id']=i
    return specs


def prepare():
    if ROOT.exists():raise RuntimeError('preserve existing design')
    # Initialize shared caches once, before parallel readers.
    for encoding in ['raw','adaptive']:feature_data(encoding)
    ROOT.mkdir(parents=True)
    specs=specifications();assert len(specs)==105
    plan=dict(status='exploratory_preregistered_bc_initialized_ppo_factorial',specs=specs,
        config=asdict(CONFIG),bc=dict(teacher_episodes=32,teacher_interactions=38400,updates=2048,lr=.003,batch=64),
        rl=dict(budgets=[65536,262144],batch=32,rollout=64,epochs=2,minibatch=512,lr=.0003,
            gamma=float(np.exp(-.2/30)),gae_lambda=.95,policy_clip=.2,value_clip=.2,entropy_coef=.001,grad_clip=1.),
        endpoint_rule='BC0, learner65536, learner262144; fixed endpoints; no best selection',
        heldout_seeds=list(range(61001,61033)),transfer_seeds=list(range(61001,61017)),
        transfer_conditions=['phase_none','phase_shifted','frequency_075','event_early90'],
        learning_threshold=dict(complete_success=.75,post_tracking=.5),
        phases=['all105fits','then all heldout evaluations','final endpoint transfer only'],
        source_sha256={n:sha(n) for n in SOURCES},
        input_sha256={n:sha(n) for n in ['results/navigation-teacher-2026-10-25.npz',
            'results/navigation-feature-teacher-raw-2026-10-25.npz','results/navigation-feature-teacher-adaptive-2026-10-25.npz']},
        versions=dict(torch=torch.__version__,numpy=np.__version__),
        limitations=['Conditional on common demonstrations, not from-scratch RL efficiency.',
            'Exploratory ten graph layouts x three training seeds; no uniform random graph sampling claim.',
            'Adaptive subset graphs11/44 fixed in advance, not selected from task performance.',
            'I/O projection and scalar rate nodes remain engineering approximations.',
            'Auxiliary adaptive channels add causal long history; response amplitudes not calcium calibrated.',
            'Actor effective capacities differ; critic architecture and initial values common.',
            'Raw auxiliary inputs are zero-masked; nominal/active capacity differ.',
            'Old online A2C and recurrent-chunk pilots retained; development cases retired.'])
    atomic_json(ROOT/'design.json',plan)
    for n in SOURCES:
        p=ROOT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,p)
    print('FIXED DESIGN',len(specs),'fits',len(specs)*3,'primary endpoints',flush=True)


def check(plan):
    for n,h in plan['source_sha256'].items():assert sha(n)==h,n
    for n,h in plan['input_sha256'].items():assert sha(n)==h,n


def fit_stage(plan,shard,shards):
    for s in plan['specs'][shard::shards]:
        out=ROOT/f'fit-{s["id"]:03d}';out.mkdir(exist_ok=True)
        result=out/'record.json'
        if result.exists() and json.loads(result.read_text()).get('status') in ('complete','failed'):continue
        started=time.monotonic()
        try:
            torch.manual_seed(s['seed']);model=WindowActorCritic(s['name'],s['interface'])
            bc=fit_bc(model,feature_data(s['encoding']),s['seed'],plan['bc']['updates'])
            torch.save(model.state_dict(),out/'BC.pt')
            atomic_json(out/'bc-curves.json',bc)
            rl=fit_rl(model,s['encoding'],s['seed'],plan['rl']['budgets'],out,'model')
            record=dict(spec=s,status='complete',bc=bc,rl=rl,metadata=model.metadata(),
                endpoints={'0':{'checkpoint':'BC.pt','learner_interactions':0},**rl['endpoints']},
                elapsed_s=time.monotonic()-started)
        except Exception as exc:
            record=dict(spec=s,status='failed',error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-started)
        atomic_json(result,record)
        print('FIT COMPLETE',s['id'],s['name'],s['interface'],s['encoding'],s['seed'],record['status'],flush=True)
    check(plan)


def evaluation_stage(plan,shard,shards):
    records=[]
    for s in plan['specs']:
        p=ROOT/f'fit-{s["id"]:03d}'/'record.json'
        if not p.exists():raise RuntimeError('all fits must finish before any heldout access')
        r=json.loads(p.read_text())
        if r.get('status') not in ('complete','failed'):raise RuntimeError('incomplete fit')
        records.append(r)
    conditions=dict(phase_none=(CONFIG,'none'),phase_shifted=(CONFIG,'shifted'),
        frequency_075=(replace(CONFIG,frequency_Hz=.75),'correct'),event_early90=(replace(CONFIG,event_s=90.),'correct'))
    for s in plan['specs'][shard::shards]:
        dest=ROOT/f'evaluation-{s["id"]:03d}.json'
        if dest.exists() and json.loads(dest.read_text()).get('complete'):continue
        record=records[s['id']];results=dict(spec=s,status=record['status'],primary=[],transfer={})
        if record['status']=='complete':
            model=WindowActorCritic(s['name'],s['interface'])
            for budget,endpoint in record['endpoints'].items():
                model.load_state_dict(torch.load(ROOT/f'fit-{s["id"]:03d}'/endpoint['checkpoint'],map_location='cpu',weights_only=True));model.eval()
                rows=evaluate(model,s['encoding'],CONFIG,plan['heldout_seeds'])
                results['primary'].append(dict(learner_interactions=int(budget),total_interactions=38400+int(budget),cases=rows,summary=summary(rows)))
            # Final is fixed by the design, not selected by scores.
            assert record['endpoints'].get('262144')
            for name,(config,phase_mode) in conditions.items():
                rows=evaluate(model,s['encoding'],config,plan['transfer_seeds'],phase_mode)
                results['transfer'][name]=dict(cases=rows,summary=summary(rows))
        results['complete']=True;atomic_json(dest,results)
        print('EVALUATED',s['id'],s['name'],s['interface'],s['encoding'],s['seed'],flush=True)
    check(plan)


def collect(plan):
    allrows=[];fits=[]
    for s in plan['specs']:
        p=ROOT/f'evaluation-{s["id"]:03d}.json'
        if not p.exists():raise RuntimeError('all evaluations required')
        row=json.loads(p.read_text());assert row['complete'];allrows.append(row)
        fits.append(json.loads((ROOT/f'fit-{s["id"]:03d}'/'record.json').read_text()))
    controls={}
    for phase in ['correct','none','shifted']:
        rows=reference(CONFIG,plan['heldout_seeds'],phase_mode=phase)
        controls['reference_'+phase]=dict(cases=rows,summary=summary(rows))
    result=dict(design=plan,fits=fits,runs=allrows,controls=controls,complete=True,
        source_hashes_unchanged={n:sha(n)==h for n,h in plan['source_sha256'].items()},
        input_hashes_unchanged={n:sha(n)==h for n,h in plan['input_sha256'].items()})
    check(plan);atomic_json(ROOT/'results.json',result)
    print('COLLECTED',len(allrows),'successful fits',sum(r['status']=='complete' for r in fits),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','fit','evaluate','collect'])
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1);args=parser.parse_args()
    torch.set_num_threads(1)
    if args.stage=='prepare':prepare();return
    plan=json.loads((ROOT/'design.json').read_text());check(plan)
    if args.stage=='fit':fit_stage(plan,args.shard,args.shards)
    elif args.stage=='evaluate':evaluation_stage(plan,args.shard,args.shards)
    else:collect(plan)


if __name__=='__main__':main()
