"""Separate local-reference feasibility audit before navigation learning."""
from dataclasses import asdict,replace
import json
from pathlib import Path

import numpy as np

from thermotaxis.navigation import NavigationBatch,NavigationConfig
from thermotaxis.isothermal_scene import SweepReading,PhaseRegressionController,RegressionControlConfig


def reference(config,seeds,phase_mode='correct'):
    env=NavigationBatch(config,seeds,phase_mode=phase_mode)
    controllers=[PhaseRegressionController(RegressionControlConfig(preferred_K=config.preferred_K,
        comfort_scale_K=config.comfort_K,speed_m_s=config.speed_m_s,head_forward_offset_m=config.length_m/2,
        head_lateral_amplitude_m=config.sweep_m,window_s=2/config.frequency_Hz,max_turn_rate_rad_s=config.turn_limit_rad_s)) for _ in seeds]
    packet=env.initial_packet()
    while True:
        for k,sample in enumerate(packet):
            for i,controller in enumerate(controllers):
                if sample[i,4]:
                    time=env.time-(len(packet)-1-k)*config.sensor_dt_s
                    controller.observe(SweepReading(time,float(sample[i,0]),float(np.arctan2(sample[i,1],sample[i,2]))))
        actions=np.zeros((len(seeds),2));actions[:,0]=[c.command(env.time)/config.turn_limit_rad_s for c in controllers]
        packet,_,done=env.step(actions)
        if done:break
    return env.metrics()


def summary(rows):
    keys=['complete_success','full_tracking','pre_tracking','post_tracking','residence']
    result={k:float(np.mean([r[k] for r in rows])) for k in keys}
    for k in ['first_entry_s','pre_stable_s','reentry_s','post_stable_s']:
        vals=[r[k] for r in rows if r[k] is not None]
        result[k]=dict(successes=len(vals),conditional_mean_s=float(np.mean(vals)) if vals else None)
    return result


if __name__=='__main__':
    out=Path('results/navigation-preflight-2026-10-25');out.mkdir(exist_ok=True)
    config=NavigationConfig();seeds=list(range(21001,21017))
    rows=reference(config,seeds)
    data=dict(status='development_preflight_retired_cases',config=asdict(config),rows=rows,summary=summary(rows))
    (out/'results.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data['summary'],indent=2))
