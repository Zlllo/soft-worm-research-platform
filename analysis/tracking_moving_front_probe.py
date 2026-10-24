"""Frozen transfer to literature-motivated moving fronts; no new fitting."""
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from compare_tracking_topology import make_model
from train_tracking_network import expert_for
from thermotaxis.isothermal_scene import TrackingConfig,BlindTrackingController,run_tracking
from thermotaxis.tracking_learning import NeuralTrackingController
from thermotaxis.moving_isotherm import moving_tracking_metrics,isotherm_normal_speed


def main():
    torch.set_num_threads(1)
    out=Path('results/tracking-moving-front-2026-10-24')
    if out.exists():raise RuntimeError('moving-front probe already exists')
    out.mkdir(parents=True)
    sources=['analysis/tracking_moving_front_probe.py','src/thermotaxis/moving_isotherm.py',
             'src/thermotaxis/isothermal_scene.py','src/thermotaxis/tracking_circuit.py',
             'src/thermotaxis/tracking_learning.py','analysis/compare_tracking_topology.py','analysis/train_tracking_network.py']
    conditions={'static':(0.,60.),'wave_005_60':(.05,60.),'wave_010_120':(.1,120.)}
    plan=dict(status='exploratory_frozen_zero_shot_not_training',scene=asdict(TrackingConfig()),
        models=['real8','rewired8-11','gru3','gru16'],training_seeds=[7,17,27],
        training_episodes_per_field=16,training_updates=2048,case_seeds=[9701,9702],
        conditions=conditions,source_url='https://doi.org/10.1242/jeb.02590',
        metric='comfort and direction of velocity relative to the moving isotherm; old static metric retained separately',
        limitations=['No model is trained on moving fronts.','Engineering teacher assumes temperature drift is motion-related; not an optimal moving-front oracle.',
                     'Kinematic feasibility does not establish identifiability from local observations.',
                     'Biological flower/loop trajectories are not evidence that worms optimize this exact tracking metric.'],
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources})
    (out/'design.json').write_text(json.dumps(plan,indent=2)+'\n')
    for n in sources:
        target=out/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,target)
    models=[]
    for name in plan['models']:
        root=Path('results/tracking-learning-surface-2026-10-24')/('grus' if name.startswith('gru') else 'rates')
        study=json.loads((root/'results.json').read_text());assert study['fitting_complete']
        for seed in plan['training_seeds']:
            row=next(r for r in study['runs'] if r['model']==name and r['seed']==seed and r['episodes_per_field']==16)
            checkpoint=root/row['endpoints']['2048']['checkpoint']
            model=make_model(name,'demeaned','full')
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
            models.append((name,seed,model,str(checkpoint)))
    models.extend([(n,None,None,None) for n in ['expert','blind']])
    report=dict(design=plan,runs=[],reference_checks={})
    traces={}
    for label,(amplitude,period) in conditions.items():
        config=replace(TrackingConfig(),uniform_wave_amplitude_K=amplitude,uniform_wave_period_s=period)
        t=np.arange(round(config.duration_s/config.physics_dt_s)+1)*config.physics_dt_s
        vx=isotherm_normal_speed(config,t);vy=np.sqrt(config.speed_m_s**2-vx**2)
        x=-amplitude*np.sin(2*np.pi*t/period)/config.gradient_K_m
        y=np.r_[0,np.cumsum((vy[1:]+vy[:-1])*.5*config.physics_dt_s)]
        heading=np.arctan2(vy,vx);omega=np.gradient(heading,config.physics_dt_s)
        reference=dict(trajectory=np.column_stack((t,x,y,heading,omega,x,y,t*0)),initial_state={'field_angle_rad':0.})
        report['reference_checks'][label]=dict(metrics=moving_tracking_metrics(config,reference),
            max_required_heading_deviation_deg=float(np.rad2deg(np.arcsin(np.max(np.abs(vx))/config.speed_m_s))),
            max_required_turn_rate_rad_s=float(np.max(np.abs(omega))))
    for name,training_seed,model,checkpoint in models:
        for label,(amplitude,period) in conditions.items():
            for field in ['linear','radial']:
                config=replace(TrackingConfig(),field=field,uniform_wave_amplitude_K=amplitude,uniform_wave_period_s=period)
                for seed in plan['case_seeds']:
                    controller=expert_for(config) if name=='expert' else BlindTrackingController() if name=='blind' else NeuralTrackingController(model)
                    row=run_tracking(config,controller,seed,record_trajectory=True)
                    metrics=moving_tracking_metrics(config,row)
                    assert abs(metrics['static_tangent_tracking_fraction']-row['metrics']['tracking_fraction'])<1e-10
                    report['runs'].append(dict(model=name,training_seed=training_seed,condition=label,field=field,seed=seed,
                        initial_state=row['initial_state'],moving_metrics=metrics,original_metrics=row['metrics'],checkpoint=checkpoint))
                    if name in ['real8','gru16'] and training_seed==7 and field=='radial' and seed==9701:
                        traces[name+'-'+label]=row['trajectory'][::10]
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print('MOVING',name,training_seed,'complete',flush=True)
    report['source_hashes_unchanged']={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in plan['source_sha256'].items()}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/'example-traces.npz',**traces)


if __name__=='__main__':main()
