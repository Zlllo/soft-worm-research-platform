"""Frozen phase replication, with initial temperature/radius held comparable."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch

from compare_tracking_topology import make_model
from train_tracking_network import expert_for
from thermotaxis.isothermal_scene import TrackingConfig,BlindTrackingController,run_tracking
from thermotaxis.tracking_learning import NeuralTrackingController
from thermotaxis.phased_field import PhasedTemperatureField,phased_tracking_metrics


def main():
    torch.set_num_threads(1)
    out=Path('results/tracking-moving-phase-2026-10-24')
    if out.exists():raise RuntimeError('phase output exists')
    out.mkdir(parents=True)
    sources=['analysis/check_moving_front_phase.py','src/thermotaxis/phased_field.py',
             'src/thermotaxis/moving_isotherm.py','src/thermotaxis/isothermal_scene.py',
             'src/thermotaxis/tracking_learning.py','src/thermotaxis/tracking_circuit.py',
             'analysis/compare_tracking_topology.py','analysis/train_tracking_network.py']
    conditions={'wave_005_60':(.05,60.),'wave_010_120':(.1,120.)}
    plan=dict(status='posthoc_frozen_phase_replication_no_fitting',
        models=['real8','rewired8-11','rewired8-22','rewired8-33','gru16'],training_seeds=[7,17,27],
        phases_rad=[0.,math.pi/2,math.pi,3*math.pi/2],case_seeds=[10101,10102],conditions=conditions,
        temperature_wave='A*(sin(2*pi*t/P+phi)-sin(phi)) in the initial-isotherm coordinate frame; initial error comparable',
        radial_scope='Physical mean radius fixed5mm; initialize at R_mean-A*sin(phi)/gradient. Mean field and entire radius range stay fixed.',
        linear_scope='Translated initial-isotherm coordinates; physical start position shifts by -A*sin(phi)/gradient along field normal.',
        discarded_design='First anchored fixed-initial-radius design stopped: A=.1K phi=3pi/2 would collapse the target to origin; retained separately, excluded.',
        late_start_s=20.,late_scope='Late interval is a startup diagnostic, not a steady-state claim.',
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources})
    (out/'design.json').write_text(json.dumps(plan,indent=2)+'\n')
    for n in sources:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,p)
    specs=[]
    for name in plan['models']:
        root=Path('results/tracking-learning-surface-2026-10-24')/('grus' if name.startswith('gru') else 'rates')
        study=json.loads((root/'results.json').read_text())
        for seed in plan['training_seeds']:
            row=next(r for r in study['runs'] if r['model']==name and r['seed']==seed and r['episodes_per_field']==16)
            checkpoint=root/row['endpoints']['2048']['checkpoint'];model=make_model(name,'demeaned','full')
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
            specs.append((name,seed,model,str(checkpoint)))
    specs.extend([(n,None,None,None) for n in ['expert','blind']])
    report=dict(design=plan,runs=[])
    for name,training_seed,model,checkpoint in specs:
        for phase in plan['phases_rad']:
            for label,(amplitude,period) in conditions.items():
                for field in ['linear','radial']:
                    config=replace(TrackingConfig(),field=field,uniform_wave_amplitude_K=amplitude,uniform_wave_period_s=period)
                    if field=='radial':
                        mean_radius=config.target_radius_m
                        assert mean_radius-amplitude/config.gradient_K_m>0
                        config=replace(config,target_radius_m=mean_radius-amplitude*math.sin(phase)/config.gradient_K_m)
                    for seed in plan['case_seeds']:
                        controller=expert_for(config) if name=='expert' else BlindTrackingController() if name=='blind' else NeuralTrackingController(model)
                        wrapper=PhasedTemperatureField(controller,config,phase)
                        row=run_tracking(config,wrapper,seed,record_trajectory=True)
                        full=phased_tracking_metrics(config,row,phase)
                        late=dict(row,trajectory=row['trajectory'][round(plan['late_start_s']/config.physics_dt_s):])
                        report['runs'].append(dict(model=name,training_seed=training_seed,phase_rad=phase,
                            condition=label,field=field,seed=seed,initial_state=row['initial_state'],
                            mean_target_radius_m=(config.target_radius_m+amplitude*math.sin(phase)/config.gradient_K_m) if field=='radial' else None,
                            metrics=full,late_metrics=phased_tracking_metrics(config,late,phase),checkpoint=checkpoint))
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PHASE',name,training_seed,flush=True)
    report['source_hashes_unchanged']={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in plan['source_sha256'].items()}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
