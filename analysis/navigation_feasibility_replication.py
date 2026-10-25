"""Physics-motivated replication after detecting finite-horizon task coupling.

Original full-I/O BC checkpoints are reused. Only training field rotation
changes from60 to15deg; learner budget/algorithm unchanged. New case seeds are
unseen until all42 RL fits finish. Matched cross-evaluation diagnoses geometry.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from navigation_window_ppo import fit_rl,CONFIG
from navigation_preflight import summary
from thermotaxis.navigation import NavigationBatch
from thermotaxis.navigation_window_policy import NavigationFeatures,WindowActorCritic
from tracking_learning_surface import atomic_json


ROOT=Path('results/navigation-feasible-2026-10-25');PARENT=Path('results/navigation-factorial-2026-10-25')
FEASIBLE=replace(CONFIG,field_turn_rad=math.pi/12)


def evaluate(model,config,seeds):
    env=NavigationBatch(config,seeds);features=NavigationFeatures(len(seeds),'raw');packet=env.initial_packet()
    with torch.no_grad():
        while True:
            x=features.observe(packet);a=np.ones((len(seeds),2));a[:,0]=torch.tanh(model.actor_mean(torch.from_numpy(x))).numpy()
            packet,_,done=env.step(a)
            if done:break
    rows=env.metrics();hist=np.stack(env.history);idx=np.round(env.event/.2).astype(int);cols=np.arange(len(seeds))
    pose_x=hist[idx,cols,3];pose_y=hist[idx,cols,4];pre_error=env.field(pose_x,pose_y,env.event-1e-7)
    post_error=hist[idx,cols,1]
    for i,row in enumerate(rows):
        minimum=max(0,abs(post_error[i])-config.comfort_K)/config.gradient_K_m/config.speed_m_s
        remaining=config.duration_s-env.event[i]
        row.update(pre_event_in_band=bool(abs(pre_error[i])<=config.comfort_K),event_error_K=float(post_error[i]),
            optimistic_entry_lower_bound_s=float(minimum),remaining_s=float(remaining),
            speed_bound_rules_out_stable_recovery=bool(minimum+config.stable_s>remaining+1e-9))
    return rows


def prepare():
    if ROOT.exists():raise RuntimeError('preserve feasibility replication')
    ROOT.mkdir();parent=json.loads((PARENT/'design.json').read_text())
    specs=[s for s in parent['specs'] if s['encoding']=='raw' and s['interface']=='full'];assert len(specs)==42
    sources=['analysis/navigation_feasibility_replication.py','analysis/navigation_window_ppo.py',
        'src/thermotaxis/navigation.py','src/thermotaxis/navigation_window_policy.py']
    plan=dict(status='posthoc_physics_motivated_replication_with_new_cases',specs=specs,config=asdict(FEASIBLE),
        original_angle_deg=60,replication_angle_deg=15,rl_budgets=[65536,262144],
        baseline='exact same32-demo2048-update BC checkpoints reused; not42 new BC fits',
        heldout_seeds=list(range(71001,71033)),cross_seeds=list(range(71001,71017)),
        motivation='Detected a speed-limit recovery lower-bound violation in predeclared original example, not chosen by graph ranking.',
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources},
        parent_design_sha256=hashlib.sha256((PARENT/'design.json').read_bytes()).hexdigest(),
        geometric_bound=dict(comfort_half_width_m=.05/40,max_pre_tracking_tangential_distance_m=140*.0002,
            field_shift_m=.08/40,minimum_post_time_s=100),
        limits=['Necessary straight-line speed bound is not a sufficient controllability proof.',
            'Only full/raw families replicated; original interface/adaptation matrix remains a stress study.',
            'New test cases never used for tuning; sourceBC data still from original scenario.',
            'Cross-evaluation is paired on new16 cases; primary new evaluation has32.'])
    atomic_json(ROOT/'design.json',plan)


def check(plan):
    for n,h in plan['source_sha256'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n


def fit_stage(plan,shard,shards):
    for s in plan['specs'][shard::shards]:
        output=ROOT/f'fit-{s["id"]:03d}';output.mkdir(exist_ok=True);record=output/'record.json'
        if record.exists():continue
        parent=json.loads((PARENT/f'fit-{s["id"]:03d}'/'record.json').read_text());assert parent['status']=='complete'
        model=WindowActorCritic(s['name'],'full');model.load_state_dict(torch.load(PARENT/f'fit-{s["id"]:03d}'/'BC.pt',map_location='cpu',weights_only=True))
        rl=fit_rl(model,'raw',s['seed'],plan['rl_budgets'],output,'feasible',config=FEASIBLE)
        atomic_json(record,dict(spec=s,status='complete',rl=rl,parent_BC_sha256=hashlib.sha256((PARENT/f'fit-{s["id"]:03d}'/'BC.pt').read_bytes()).hexdigest()))
        print('FEASIBLE FIT',s['id'],s['name'],s['seed'],flush=True)
    check(plan)


def evaluation_stage(plan,shard,shards):
    for s in plan['specs']:assert (ROOT/f'fit-{s["id"]:03d}'/'record.json').exists()
    for s in plan['specs'][shard::shards]:
        dest=ROOT/f'evaluation-{s["id"]:03d}.json'
        if dest.exists():continue
        parent=json.loads((PARENT/f'fit-{s["id"]:03d}'/'record.json').read_text())
        fitted=json.loads((ROOT/f'fit-{s["id"]:03d}'/'record.json').read_text());model=WindowActorCritic(s['name'],'full');rows=[];cross={}
        checkpoints={'0':PARENT/f'fit-{s["id"]:03d}'/'BC.pt',**{b:ROOT/f'fit-{s["id"]:03d}'/v['checkpoint'] for b,v in fitted['rl']['endpoints'].items()}}
        for budget,path in checkpoints.items():
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
            cases=evaluate(model,FEASIBLE,plan['heldout_seeds']);rows.append(dict(learner_interactions=int(budget),cases=cases,summary=summary(cases)))
        cross['trained15_test60']=evaluate(model,CONFIG,plan['cross_seeds'])
        model.load_state_dict(torch.load(PARENT/f'fit-{s["id"]:03d}'/parent['endpoints']['262144']['checkpoint'],map_location='cpu',weights_only=True));model.eval()
        cross['trained60_test15']=evaluate(model,FEASIBLE,plan['cross_seeds'])
        cross['trained60_test60']=evaluate(model,CONFIG,plan['cross_seeds'])
        atomic_json(dest,dict(spec=s,primary=rows,cross=cross,complete=True));print('FEASIBLE EVAL',s['id'],flush=True)
    check(plan)


def collect(plan):
    rows=[]
    for s in plan['specs']:
        row=json.loads((ROOT/f'evaluation-{s["id"]:03d}.json').read_text());assert row['complete'];rows.append(row)
    groups={}
    for name in dict.fromkeys(s['name'] for s in plan['specs']):
        selected=[r for r in rows if r['spec']['name']==name];item={}
        for budget in [0,65536,262144]:
            vals=[next(p for p in r['primary'] if p['learner_interactions']==budget) for r in selected]
            item[str(budget)]=dict(complete_success=float(np.mean([p['summary']['complete_success'] for p in vals])),
                tracking=float(np.mean([p['summary']['full_tracking'] for p in vals])),
                by_seed={str(r['spec']['seed']):p['summary'] for r,p in zip(selected,vals)},
                ruled_out_good_event_cases=sum(c['pre_event_in_band'] and c['speed_bound_rules_out_stable_recovery'] for p in vals for c in p['cases']))
        item['cross']={label:dict(complete_success=float(np.mean([summary(r['cross'][label])['complete_success'] for r in selected])),
            ruled_out_good_event_cases=sum(c['pre_event_in_band'] and c['speed_bound_rules_out_stable_recovery'] for r in selected for c in r['cross'][label]),
            good_event_cases=sum(c['pre_event_in_band'] for r in selected for c in r['cross'][label]))
            for label in ['trained15_test60','trained60_test15','trained60_test60']}
        groups[name]=item
    check(plan);atomic_json(ROOT/'results.json',dict(design=plan,runs=rows,groups=groups,complete=True,
        source_hashes_unchanged={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in plan['source_sha256'].items()}))
    print('FEASIBLE COLLECT', {n:(x['262144']['complete_success'],x['262144']['tracking']) for n,x in groups.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit','evaluate','collect']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='prepare':prepare()
    else:
        plan=json.loads((ROOT/'design.json').read_text());check(plan)
        if a.stage=='fit':fit_stage(plan,a.shard,a.shards)
        elif a.stage=='evaluate':evaluation_stage(plan,a.shard,a.shards)
        else:collect(plan)
