"""Diagnostic reproductions for PR #1 at 9f9288e, inspected 2026-09-22.
Run against an extracted snapshot, never against the baseline accidentally.
Requires numpy and torch. Does not modify that snapshot or train saved models.
Results are diagnostic counterexamples in uncalibrated code units.
Actor bound check deliberately sets an allowed parameter state; it is not a
claim that ordinary training inevitably reaches that state.
"""
import contextlib, io, json, sys, random
from pathlib import Path
import numpy as np
import torch
if len(sys.argv) != 2:
    raise SystemExit('Usage: python3 pr1_probe.py /path/to/pr1-snapshot')
pr_root = Path(sys.argv[1]).resolve()
if not (pr_root / '程序/core/actor_critic.py').is_file():
    raise SystemExit('Expected PR #1 source snapshot at the supplied path')
sys.path.insert(0, str(pr_root / '程序'))
with contextlib.redirect_stdout(io.StringIO()):
    from core.worm_body import ActiveDeformationBody
    from core.environment import Environment2D

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)

def body(noise=None, **kw):
    p = dict(sample_count=17, body_length=12., wave_length=12., sub_steps=10,
             wave_amplitude=1., wave_frequency=.25, ac_batch_size=2, ac_hidden_size=16)
    p.update(kw)
    return ActiveDeformationBody((50,50),100,100,p,noise or {})

out = {}
a,b = body(wave_frequency=.25,wave_speed=2),body(wave_frequency=.5,wave_speed=1)
ra,rb = a.step_physics(dt=.01),b.step_physics(dt=.01)
out['equivalent_phase'] = dict(phase_a=a.wave_phase,phase_b=b.wave_phase,
    movement_ratio_b_a=rb['movement']/ra['movement'], dissipation_ratio_b_a=rb['dissipation']/ra['dissipation'])

b=body(wave_frequency=0)
before=b.centerline.copy()
r=b.step_physics(action={'wave_amplitude':2.,'steer_bias':.06},dt=.1)
out['shape_change_at_zero_frequency']=dict(max_point_displacement=float(np.linalg.norm(b.centerline-before,axis=1).max()),dissipation=r['dissipation'])

b=body(wave_amplitude=2.,wave_frequency=.8)
e0=b.energy
total=sum(b.step_physics()['dissipation'] for _ in range(60))
out['dissipation_saturation']=dict(accumulated=total,budget_difference=e0-b.energy)
b.energy=0
pos=b.com.copy()
r=b.step_physics()
out['zero_energy']=dict(displacement=float(np.linalg.norm(b.com-pos)),dissipation=r['dissipation'])

env=Environment2D(np.tile(np.linspace(10,30,100),(100,1)))
a,b=body(noise={'thermal_noise':0}),body(noise={'thermal_noise':100})
out['thermal_noise_observation']=dict(max_abs_difference=float(np.max(np.abs(a.get_state_v2(env)-b.get_state_v2(env)))))

b=body(curriculum_freeze_steps=100)
with contextlib.redirect_stdout(io.StringIO()): b.setup_actor_critic()
agent=b.actor_critic_agent
agent.act=lambda state,add_noise=True:(1.,.25,.06)
agent.train_interval=1
b.decide_move_actor_critic(env)
out['frozen_action']=dict(executed_bias=b.steer_bias,replay_bias=float(agent.replay_buffer.buffer[-1][1][2]))
agent.eval()
before=[p.detach().clone() for p in agent.actor.parameters()]
size=len(agent.replay_buffer)
b.decide_move_actor_critic(env)
out['evaluation']=dict(replay_growth=len(agent.replay_buffer)-size,actor_changed=any(not torch.equal(x,y) for x,y in zip(before,agent.actor.parameters())))

with torch.no_grad():
    for p in agent.actor.net.parameters(): p.zero_()
    agent.actor.action_scales.fill_(3.)
    raw=agent.actor(torch.zeros(1,14))[0].tolist()
out['actor_bounds']=dict(raw_output=raw,upper_bounds=agent.actor.hi.tolist())
print(json.dumps(out,indent=2))
