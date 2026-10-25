"""Supplementary zero-lateral-sweep control, before analyst score inspection.

Declared after evaluation launch, so labeled supplementary rather than part of
primary preregistration. Separates lateral head motion from center-vs-head
forward-offset changes. All actors/checkpoints included without score selection.
"""
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path

import torch

from navigation_window_ppo import evaluate,CONFIG
from navigation_preflight import summary
from thermotaxis.navigation_window_policy import WindowActorCritic
from tracking_learning_surface import atomic_json


ROOT=Path('results/navigation-zero-sweep-2026-10-25');PARENT=Path('results/navigation-factorial-2026-10-25')


def prepare():
    ROOT.mkdir(exist_ok=False)
    plan=dict(status='supplementary_design_after_evaluation_launch_before_analyst_score_inspection',
        reason='Separate lateral sweep removal from head-to-center forward-offset change.',
        all105_specifications=True,checkpoint=262144,seeds=list(range(61001,61017)),
        config=asdict(replace(CONFIG,sweep_m=0.)),
        parent_design_sha256=hashlib.sha256((PARENT/'design.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(ROOT/'design.json',plan)


def run(shard,shards):
    plan=json.loads((ROOT/'design.json').read_text());parent=json.loads((PARENT/'design.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan['script_sha256']
    for s in parent['specs']:
        assert (PARENT/f'fit-{s["id"]:03d}'/'record.json').exists()
    for s in parent['specs'][shard::shards]:
        output=ROOT/f'evaluation-{s["id"]:03d}.json'
        if output.exists():continue
        record=json.loads((PARENT/f'fit-{s["id"]:03d}'/'record.json').read_text());result=dict(spec=s,status=record['status'])
        if record['status']=='complete':
            model=WindowActorCritic(s['name'],s['interface'])
            model.load_state_dict(torch.load(PARENT/f'fit-{s["id"]:03d}'/record['endpoints']['262144']['checkpoint'],map_location='cpu',weights_only=True));model.eval()
            rows=evaluate(model,s['encoding'],replace(CONFIG,sweep_m=0.),plan['seeds'])
            result.update(cases=rows,summary=summary(rows))
        atomic_json(output,result);print('ZERO SWEEP',s['id'],flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='prepare':prepare()
    else:run(a.shard,a.shards)
