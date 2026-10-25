"""Exploratory real-graph refits separate frozen disruption from trainability.

Not a topology comparison: only the real graph is refitted. Six runs are fixed
before looking at their behavior, using the exact old teacher data and budget.
"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from compare_tracking_topology import make_model, fit, subset
from tracking_learning_surface import atomic_json, cache_data
from train_tracking_network import dataset_digest
from thermotaxis.tracking_interventions import with_time_constant
from thermotaxis.tracking_learning import NeuralTrackingController
from thermotaxis.isothermal_scene import TrackingConfig, run_tracking
from thermotaxis.phased_field import PhasedTemperatureField, phased_tracking_metrics
from probe_tracking_timescale import SOURCES


ROOT=Path('results/tracking-timescale-refit-real-2026-10-25')


def main():
    torch.set_num_threads(1)
    if ROOT.exists():raise RuntimeError('Refit output exists; preserve this batch.')
    ROOT.mkdir(parents=True)
    training,val,fingerprint=cache_data()
    parent=json.loads(Path('results/tracking-topology-fullio-budgets-2026-10-24/results.json').read_text())
    assert fingerprint==parent['data_fingerprint']
    assert dataset_digest(training)==parent['full_training_data']
    assert dataset_digest(val)==parent['validation_data']
    data=subset(training,16)
    source_names=SOURCES+['analysis/refit_tracking_timescale.py']
    plan=dict(status='exploratory_real_graph_refit_only',model='real8',training_seeds=[7,17,27],
        tau_s=[.1,.9],episodes_per_field=16,total_training_episodes=32,updates=2048,
        optimizer=dict(name='Adam',lr=.003,batch=64,clipping=1.),data_fingerprint=fingerprint,
        training_digest=dataset_digest(data),validation_digest=dataset_digest(val),
        case_seeds=[10501,10502],phases_rad=[0.,math.pi/2],fields=['linear','radial'],
        wave_amplitude_K=.05,wave_period_s=60.,late_start_s=20.,
        intent='Distinguish frozen real-graph damage from fixed-budget learnability; no topology advantage claim.',
        constraints=['All six fits finish before behavior evaluation.','Tau=.3 parent is reused, not independently fitted.',
            'Behavior cases match frozen probes for paired diagnostic; not an untouched final test.',
            'Same seed reproduces initial weight values and optimizer minibatch sequence.'],
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in source_names})
    atomic_json(ROOT/'design.json',plan)
    for n in source_names:
        q=ROOT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,q)
    report=dict(design=plan,fits=[],runs=[])
    for tau in plan['tau_s']:
        for seed in plan['training_seeds']:
            torch.manual_seed(seed)
            model=with_time_constant(make_model('real8','demeaned','full'),tau)
            curves=fit(model,data,val,plan['updates'],seed)
            checkpoint=f'real8-{seed}-tau{tau}-U2048.pt';torch.save(model.state_dict(),ROOT/checkpoint)
            report['fits'].append(dict(tau_s=tau,training_seed=seed,checkpoint=checkpoint,curves=curves))
            atomic_json(ROOT/'results.json',report)
            print('REFIT',tau,seed,'validation',curves[-1]['validation']['all']['normalized_mse'],flush=True)
    report['fitting_complete']=True;atomic_json(ROOT/'results.json',report)
    for record in report['fits']:
        model=with_time_constant(make_model('real8','demeaned','full'),record['tau_s'])
        model.load_state_dict(torch.load(ROOT/record['checkpoint'],map_location='cpu',weights_only=True));model.eval()
        cells=[]
        for scene in ['static','moving']:
            for phase in ([0.] if scene=='static' else plan['phases_rad']):
                for field in plan['fields']:
                    config=replace(TrackingConfig(),field=field,uniform_wave_amplitude_K=0. if scene=='static' else .05,
                        uniform_wave_period_s=60.)
                    if field=='radial':config=replace(config,target_radius_m=config.target_radius_m-
                        config.uniform_wave_amplitude_K*math.sin(phase)/config.gradient_K_m)
                    for case in plan['case_seeds']:
                        row=run_tracking(config,PhasedTemperatureField(NeuralTrackingController(model),config,phase),case,record_trajectory=True)
                        late=dict(row,trajectory=row['trajectory'][round(20/config.physics_dt_s):])
                        cells.append(dict(scene=scene,phase_rad=phase,field=field,case_seed=case,initial_state=row['initial_state'],
                            metrics=phased_tracking_metrics(config,row,phase),late_metrics=phased_tracking_metrics(config,late,phase)))
        report['runs'].append(dict(model='real8',training_seed=record['training_seed'],tau_s=record['tau_s'],
            checkpoint=record['checkpoint'],conditions=cells))
        atomic_json(ROOT/'results.json',report);print('REFIT CASES',record['tau_s'],record['training_seed'],flush=True)
    report['source_hashes_unchanged']={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in plan['source_sha256'].items()}
    assert all(report['source_hashes_unchanged'].values())
    report['complete']=True;atomic_json(ROOT/'results.json',report)


if __name__=='__main__':main()
