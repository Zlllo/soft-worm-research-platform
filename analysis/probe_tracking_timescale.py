"""Paired frozen circuit timescale sensitivity across all ten null graphs.

This study intervenes after training; it never estimates retraining outcomes.
New cases and all conditions are saved before evaluation, without selection.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from compare_tracking_topology import make_model
from tracking_learning_surface import atomic_json
from train_tracking_network import expert_for
from thermotaxis.isothermal_scene import TrackingConfig, BlindTrackingController, run_tracking
from thermotaxis.tracking_learning import NeuralTrackingController
from thermotaxis.tracking_interventions import with_time_constant
from thermotaxis.phased_field import PhasedTemperatureField, phased_tracking_metrics


OUTPUT = Path('results/tracking-timescale-2026-10-25')
PARENT = Path('results/tracking-expanded-graphs-2026-10-24')
SOURCES = ['analysis/probe_tracking_timescale.py', 'analysis/compare_tracking_topology.py',
    'analysis/train_tracking_network.py', 'analysis/tracking_learning_surface.py',
    'src/thermotaxis/tracking_interventions.py', 'src/thermotaxis/tracking_circuit.py',
    'src/thermotaxis/tracking_learning.py', 'src/thermotaxis/isothermal_scene.py',
    'src/thermotaxis/phased_field.py', 'src/thermotaxis/moving_isotherm.py',
    'src/thermotaxis/connectome.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    parent = json.loads((PARENT / 'fitting.json').read_text())
    previous = json.loads((PARENT / 'results.json').read_text())
    assert parent['complete'] and len(parent['runs']) == 36 and len(previous['runs']) == 38
    assert all(previous['source_hashes_unchanged'].values())
    if not OUTPUT.exists():
        OUTPUT.mkdir(parents=True)
        plan = dict(status='exploratory_frozen_timescale_intervention',
            parent=str(PARENT), parent_results_sha256=digest(PARENT / 'results.json'),
            models=['real8'] + ['rewired8-' + str(s) for s in
                previous['design']['old_graph_seeds'] + previous['design']['new_graph_seeds']],
            training_seeds=[7, 17, 27], tau_s=[.1, .3, .9], trained_tau_s=.3,
            case_seeds=[10501, 10502], phases_rad=[0., math.pi / 2], fields=['linear', 'radial'],
            wave_amplitude_K=.05, wave_period_s=60., late_start_s=20.,
            baseline_scene=asdict(TrackingConfig()),
            primary='Within each trained model, paired moving-tracking change relative to tau=.3; graph and training seed separate.',
            secondary='Static tracking and late moving tracking; inspect startup separately.',
            controls=['GRU16 unchanged', 'engineered expert', 'blind'],
            expected_conditions=104, expected_trajectories=1248,
            limits=['Frozen weight sensitivity, not optimal tau or retraining benefit.',
                'Finite four-second history resets each query; tau is a neuronal relaxation coefficient, not total memory duration.',
                'Full-node input/output and demeaning remain engineering choices.',
                'Two new cases per field and two starting phases only; no population significance claim.',
                'Three training seeds crossed with ten graph layouts; graphs and seeds are not interchangeable.',
                'Tau changes all node relaxation equally; it affects chemical and gap terms through the same integration coefficient.'],
            source_sha256={n: digest(n) for n in SOURCES},
            checkpoints={r['endpoints']['2048']['checkpoint']:digest(PARENT / r['endpoints']['2048']['checkpoint'])
                for r in parent['runs']}, versions=dict(torch=torch.__version__, numpy=np.__version__))
        atomic_json(OUTPUT / 'design.json', plan)
        for n in SOURCES:
            dest=OUTPUT/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,dest)
        for n in plan['checkpoints']:
            dest=OUTPUT/'checkpoints'/n;dest.parent.mkdir(exist_ok=True);shutil.copyfile(PARENT/n,dest)
    elif not args.resume:
        raise RuntimeError('Use --resume with an existing experiment; no results are overwritten.')
    plan=json.loads((OUTPUT/'design.json').read_text())
    for n,h in plan['source_sha256'].items():
        assert digest(n)==h,n
    for n,h in plan['checkpoints'].items():
        assert digest(OUTPUT/'checkpoints'/n)==h,n
    path=OUTPUT/'results.json'
    result=json.loads(path.read_text()) if path.exists() else dict(design=plan,runs=[])
    completed={(r['model'],r['training_seed'],r['tau_s']) for r in result['runs']}
    specs=[]
    for name in plan['models']+['gru16']:
        for seed in plan['training_seeds']:
            row=next(r for r in parent['runs'] if r['model']==name and r['seed']==seed)
            checkpoint=row['endpoints']['2048']['checkpoint']
            for tau in plan['tau_s'] if name!='gru16' else [None]:
                specs.append((name,seed,tau,checkpoint))
    specs.extend((name,None,None,None) for name in ['expert','blind'])
    for name,seed,tau,checkpoint in specs:
        if (name,seed,tau) in completed:continue
        started=time.monotonic();model=None
        if checkpoint:
            model=make_model(name,'demeaned','full')
            model.load_state_dict(torch.load(OUTPUT/'checkpoints'/checkpoint,map_location='cpu',weights_only=True))
            model.eval()
            if tau is not None:
                original=model;model=with_time_constant(original,tau)
                assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.state_dict().items())
        conditions=[]
        # Identical initial condition seed and phase under each intervention.
        for scene in ['static','moving']:
            for phase in ([0.] if scene=='static' else plan['phases_rad']):
                for field in plan['fields']:
                    config=replace(TrackingConfig(**plan['baseline_scene']),field=field,
                        uniform_wave_amplitude_K=0. if scene=='static' else plan['wave_amplitude_K'],
                        uniform_wave_period_s=plan['wave_period_s'])
                    if field=='radial':
                        config=replace(config,target_radius_m=config.target_radius_m-
                            config.uniform_wave_amplitude_K*math.sin(phase)/config.gradient_K_m)
                    for case in plan['case_seeds']:
                        controller=expert_for(config) if name=='expert' else BlindTrackingController() if name=='blind' else NeuralTrackingController(model)
                        row=run_tracking(config,PhasedTemperatureField(controller,config,phase),case,record_trajectory=True)
                        late=dict(row,trajectory=row['trajectory'][round(plan['late_start_s']/config.physics_dt_s):])
                        metric=phased_tracking_metrics(config,row,phase)
                        late_metric=phased_tracking_metrics(config,late,phase)
                        conditions.append(dict(scene=scene,phase_rad=phase,field=field,case_seed=case,
                            initial_state=row['initial_state'],metrics=metric,late_metrics=late_metric,
                            static_metrics=row['metrics'] if scene=='static' else None))
        result['runs'].append(dict(model=name,training_seed=seed,tau_s=tau,
            alpha=model.alpha if tau is not None else None,checkpoint=checkpoint,
            conditions=conditions,seconds=time.monotonic()-started))
        atomic_json(path,result)
        print('DONE',len(result['runs']), '/',plan['expected_conditions'],name,seed,tau,
              'seconds',round(time.monotonic()-started,1),flush=True)
    assert len(result['runs'])==plan['expected_conditions']
    assert sum(len(r['conditions']) for r in result['runs'])==plan['expected_trajectories']
    result['complete']=True
    result['source_hashes_unchanged']={n:digest(n)==h for n,h in plan['source_sha256'].items()}
    result['checkpoint_hashes_unchanged']={n:digest(OUTPUT/'checkpoints'/n)==h for n,h in plan['checkpoints'].items()}
    assert all(result['source_hashes_unchanged'].values()) and all(result['checkpoint_hashes_unchanged'].values())
    atomic_json(path,result)


if __name__=='__main__':main()
