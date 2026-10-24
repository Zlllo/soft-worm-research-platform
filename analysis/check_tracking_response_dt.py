"""Time-step check of frozen response examples and grid-dependent roughness."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from compare_tracking_topology import make_model
from thermotaxis.isothermal_scene import TrackingConfig,run_tracking
from thermotaxis.tracking_learning import NeuralTrackingController


def main():
    torch.set_num_threads(1)
    root=Path('results/tracking-response-probe-2026-10-24')
    out=root/'dt-check'
    if out.exists():raise RuntimeError('time-step check exists')
    out.mkdir()
    sources=['analysis/check_tracking_response_dt.py','src/thermotaxis/isothermal_scene.py']
    plan=dict(status='posthoc_no_training_time_step_check',models=['real8','gru16'],training_seed=7,
        field='radial',case_seed=9501,noise_std_K=.01,response_tau_s=[0.,.3,1.2],physics_dt_s=[.01,.005],
        purpose='Tracking convergence; instantaneous-actuator acceleration RMS is grid dependent.',
        fixed_observation_interval_s=.2,
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources})
    (out/'design.json').write_text(json.dumps(plan,indent=2)+'\n')
    for n in sources:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,p)
    original=json.loads((root/'results.json').read_text());result=dict(design=plan,runs=[])
    for name in plan['models']:
        previous=next(r for r in original['runs'] if r['model']==name and r['training_seed']==7)
        model=make_model(name,'demeaned','full')
        model.load_state_dict(torch.load(previous['checkpoint'],map_location='cpu',weights_only=True));model.eval()
        for tau in plan['response_tau_s']:
            for dt in plan['physics_dt_s']:
                config=replace(TrackingConfig(),field='radial',noise_std_K=.01,turn_response_tau_s=tau,physics_dt_s=dt)
                row=run_tracking(config,NeuralTrackingController(model),9501,record_trajectory=True)
                rates=row['trajectory'][:,4];stride=round(.2/dt)
                result['runs'].append(dict(model=name,tau_s=tau,dt_s=dt,metrics=row['metrics'],
                    physics_grid_acceleration_rms=float(np.sqrt(np.mean((np.diff(rates)/dt)**2))),
                    control_grid_rate_change_rms=float(np.sqrt(np.mean((np.diff(rates[::stride])/.2)**2)))))
    result['source_hashes_unchanged']={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in plan['source_sha256'].items()}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('12 frozen time-step cases complete',flush=True)


if __name__=='__main__':main()
