"""Paired execution-response/noise probe with frozen policies; no training.

This tests robustness and action filtering, not benefits to training or energy.
The selection of models/budget/conditions is fixed before viewing probe cases.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from compare_tracking_topology import make_model
from train_tracking_network import expert_for
from thermotaxis.tracking_learning import NeuralTrackingController
from thermotaxis.isothermal_scene import TrackingConfig, BlindTrackingController, run_tracking


def main():
    torch.set_num_threads(1)
    output=Path('results/tracking-response-probe-2026-10-24')
    if output.exists():raise RuntimeError('probe output exists; do not overwrite')
    output.mkdir(parents=True)
    sources=['analysis/tracking_response_probe.py','analysis/tracking_learning_surface.py',
             'analysis/compare_tracking_topology.py','analysis/train_tracking_network.py',
             'src/thermotaxis/tracking_circuit.py','src/thermotaxis/tracking_learning.py',
             'src/thermotaxis/isothermal_scene.py']
    plan=dict(status='exploratory_frozen_policy_physical_response_not_training',
        models=['real8','rewired8-11','gru3','gru16'],training_seeds=[7,17,27],
        training_episodes_per_field=16,training_updates=2048,training_response_tau_s=.3,
        tested_response_tau_s=[0.,.3,1.2],noise_std_K=[.002,.01],
        case_seeds=[9501,9502],fields=['linear','radial'],scene=asdict(TrackingConfig()),
        observation='same local noisy head temperature, actuator phase and causal window',
        limits=['no fitting in altered physical conditions','not a sample-efficiency experiment',
                'angular acceleration and command variation are not mechanical energy or metabolism',
                'one rewired graph is a mechanism reference, not the whole random-graph ensemble'],
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources})
    (output/'design.json').write_text(json.dumps(plan,indent=2)+'\n')
    for n in sources:
        target=output/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,target)
    model_specs=[]
    for name in plan['models']:
        shard='grus' if name.startswith('gru') else 'rates'
        root=Path('results/tracking-learning-surface-2026-10-24')/shard
        study=json.loads((root/'results.json').read_text())
        assert study['fitting_complete'] and all(study['source_hashes_unchanged'].values())
        for seed in plan['training_seeds']:
            row=next(r for r in study['runs'] if r['model']==name and r['seed']==seed and r['episodes_per_field']==16)
            filename=root/row['endpoints']['2048']['checkpoint']
            model=make_model(name,'demeaned','full')
            model.load_state_dict(torch.load(filename,map_location='cpu',weights_only=True));model.eval()
            model_specs.append((name,seed,model,str(filename),hashlib.sha256(filename.read_bytes()).hexdigest()))
    model_specs.extend([(n,None,None,None,None) for n in ['expert','blind']])
    report=dict(design=plan,runs=[])
    traces={}
    for name,training_seed,model,checkpoint,weight_hash in model_specs:
        for tau in plan['tested_response_tau_s']:
            for noise in plan['noise_std_K']:
                for field in plan['fields']:
                    config=replace(TrackingConfig(),field=field,turn_response_tau_s=tau,noise_std_K=noise)
                    for seed in plan['case_seeds']:
                        controller=expert_for(config) if name=='expert' else BlindTrackingController() if name=='blind' else NeuralTrackingController(model)
                        result=run_tracking(config,controller,seed,record_trajectory=True)
                        trace=result['trajectory'];commands=result['control_commands']
                        metrics=dict(result['metrics'],
                            commanded_turn_rms_rad_s=float(np.sqrt(np.mean(commands[:-1,1]**2))),
                            executed_turn_rms_rad_s=float(np.sqrt(np.mean(trace[:-1,4]**2))),
                            command_change_rms_rad_s2=float(np.sqrt(np.mean((np.diff(commands[:,1])/config.control_dt_s)**2))),
                            executed_acceleration_rms_rad_s2=float(np.sqrt(np.mean((np.diff(trace[:,4])/config.physics_dt_s)**2))))
                        report['runs'].append(dict(model=name,training_seed=training_seed,field=field,seed=seed,
                            tau_s=tau,noise_std_K=noise,metrics=metrics,initial_state=result['initial_state'],
                            checkpoint=checkpoint,checkpoint_sha256=weight_hash))
                        if name in ('real8','gru3') and training_seed==7 and field=='radial' and seed==9501 and noise==.01:
                            label=f'{name}-tau{tau}'
                            traces[label+'-trajectory']=trace[::10]
                            traces[label+'-commands']=commands
        (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PROBE',name,training_seed,'complete',flush=True)
    report['source_hashes_unchanged']={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in plan['source_sha256'].items()}
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output/'example-traces.npz',**traces)


if __name__=='__main__':
    main()
