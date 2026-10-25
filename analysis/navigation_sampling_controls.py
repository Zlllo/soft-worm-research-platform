"""Predeclared frozen head-versus-center sampling control after all main fits.

Changes physical sensor location only, preserving paired noise streams, motor
phase, actions, task and metrics. It tests trained-policy dependence, not the
best achievable navigation with a center sensor.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from thermotaxis.navigation import NavigationBatch,NavigationConfig
from thermotaxis.navigation_window_policy import NavigationFeatures,WindowActorCritic
from tracking_learning_surface import atomic_json
from navigation_preflight import summary


ROOT=Path('results/navigation-sampling-2026-10-25')
PARENT=Path('results/navigation-factorial-2026-10-25')


class CenterSensorEnv(NavigationBatch):
    def _sample(self):
        c=self.config;phase=self.phase0+2*math.pi*c.frequency_Hz*self.time
        noise=np.array([r.normal(0,c.noise_K) for r in self.noise_rng])
        temp=c.preferred_K+self.field(self.x,self.y,self.time)+noise
        observed=phase if self.phase_mode=='correct' else phase+math.pi/2
        sine=np.sin(observed) if self.phase_mode!='none' else np.zeros(self.n)
        cosine=np.cos(observed) if self.phase_mode!='none' else np.zeros(self.n)
        return np.stack([temp,sine,cosine,self.velocity/c.speed_m_s,np.ones(self.n)],axis=1).astype(np.float32)


def prepare():
    if ROOT.exists():raise RuntimeError('preserve sampling design')
    ROOT.mkdir(parents=True)
    sources=['analysis/navigation_sampling_controls.py','src/thermotaxis/navigation.py','src/thermotaxis/navigation_window_policy.py']
    plan=dict(status='exploratory_frozen_sensor_location',declared_before_main_heldout_access=True,
        parent_design_sha256=hashlib.sha256((PARENT/'design.json').read_bytes()).hexdigest(),
        seeds=list(range(61001,61017)),config=asdict(NavigationConfig()),sensor_location='center',
        stage='only after all105 main fits; final262144 checkpoint only',
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources},
        limitations=['Frozen dependence, not impossibility of center-only navigation.',
            'Biological head motion remains prescribed, not an elastic body.'])
    atomic_json(ROOT/'design.json',plan)
    for n in sources:
        p=ROOT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,p)


def run(shard,shards):
    plan=json.loads((ROOT/'design.json').read_text());parent=json.loads((PARENT/'design.json').read_text())
    assert hashlib.sha256((PARENT/'design.json').read_bytes()).hexdigest()==plan['parent_design_sha256']
    for spec in parent['specs']:
        assert (PARENT/f'fit-{spec["id"]:03d}'/'record.json').exists(),'all fits first'
    for spec in parent['specs'][shard::shards]:
        output=ROOT/f'evaluation-{spec["id"]:03d}.json'
        if output.exists():continue
        record=json.loads((PARENT/f'fit-{spec["id"]:03d}'/'record.json').read_text())
        result=dict(spec=spec,status=record['status'])
        if record['status']=='complete':
            model=WindowActorCritic(spec['name'],spec['interface'])
            model.load_state_dict(torch.load(PARENT/f'fit-{spec["id"]:03d}'/record['endpoints']['262144']['checkpoint'],map_location='cpu',weights_only=True));model.eval()
            env=CenterSensorEnv(NavigationConfig(),plan['seeds']);features=NavigationFeatures(len(plan['seeds']),spec['encoding']);packet=env.initial_packet()
            with torch.no_grad():
                while True:
                    x=features.observe(packet);a=np.ones((len(plan['seeds']),2));a[:,0]=torch.tanh(model.actor_mean(torch.from_numpy(x))).numpy()
                    packet,_,done=env.step(a)
                    if done:break
            rows=env.metrics();result.update(cases=rows,summary=summary(rows))
        atomic_json(output,result);print('CENTER',spec['id'],flush=True)
    for n,h in plan['source_sha256'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(1)
    if a.stage=='prepare':prepare()
    else:run(a.shard,a.shards)
