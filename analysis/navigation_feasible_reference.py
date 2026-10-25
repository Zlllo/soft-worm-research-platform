"""Matched local-reference controls for the physics-motivated replication."""
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from thermotaxis.navigation import NavigationBatch
from thermotaxis.isothermal_scene import PhaseRegressionController,RegressionControlConfig,SweepReading
from navigation_feasibility_replication import ROOT,FEASIBLE,CONFIG
from navigation_preflight import summary


def run(config,seeds):
    env=NavigationBatch(config,seeds);controllers=[PhaseRegressionController(RegressionControlConfig(
        preferred_K=config.preferred_K,comfort_scale_K=config.comfort_K,speed_m_s=config.speed_m_s,
        head_forward_offset_m=config.length_m/2,head_lateral_amplitude_m=config.sweep_m,
        window_s=2/config.frequency_Hz,max_turn_rate_rad_s=config.turn_limit_rad_s)) for _ in seeds]
    packet=env.initial_packet()
    while True:
        for k,p in enumerate(packet):
            for i,c in enumerate(controllers):
                if p[i,4]:c.observe(SweepReading(env.time-(len(packet)-1-k)*.1,float(p[i,0]),float(np.arctan2(p[i,1],p[i,2]))))
        a=np.ones((len(seeds),2));a[:,0]=[c.command(env.time)/config.turn_limit_rad_s for c in controllers]
        packet,_,done=env.step(a)
        if done:break
    rows=env.metrics();h=np.stack(env.history);ix=np.round(env.event/.2).astype(int);cols=np.arange(len(seeds))
    before=env.field(h[ix,cols,3],h[ix,cols,4],env.event-1e-7);after=h[ix,cols,1]
    for i,row in enumerate(rows):
        minimum=max(0,abs(after[i])-config.comfort_K)/config.gradient_K_m/config.speed_m_s+config.stable_s
        row.update(pre_event_in_band=bool(abs(before[i])<=config.comfort_K),optimistic_recovery_plus_dwell_s=float(minimum),
            remaining_s=float(config.duration_s-env.event[i]),ruled_out=bool(minimum>config.duration_s-env.event[i]+1e-9))
    return dict(cases=rows,summary=summary(rows),ruled_out_good_cases=sum(x['pre_event_in_band'] and x['ruled_out'] for x in rows))


if __name__=='__main__':
    plan=json.loads((ROOT/'design.json').read_text())
    for s in plan['specs']:assert (ROOT/f'fit-{s["id"]:03d}'/'record.json').exists()
    result=dict(test_seeds=plan['heldout_seeds'],reference15=run(FEASIBLE,plan['heldout_seeds']),reference60=run(CONFIG,plan['heldout_seeds']))
    (ROOT/'reference-controls.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v['summary']['complete_success'] for k,v in result.items() if k.startswith('reference')})
