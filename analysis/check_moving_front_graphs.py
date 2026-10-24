"""Extend frozen moving-front checks to all rewires and a fresh case cohort."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import torch

from compare_tracking_topology import make_model
from train_tracking_network import expert_for
from thermotaxis.isothermal_scene import TrackingConfig,BlindTrackingController,run_tracking
from thermotaxis.tracking_learning import NeuralTrackingController
from thermotaxis.moving_isotherm import moving_tracking_metrics


def main():
    torch.set_num_threads(1)
    out=Path('results/tracking-moving-graphs-2026-10-24')
    if out.exists():raise RuntimeError('moving graph check exists')
    out.mkdir(parents=True)
    original_path=Path('results/tracking-moving-front-2026-10-24/results.json')
    previous=json.loads(original_path.read_text());assert all(previous['source_hashes_unchanged'].values())
    sources=['analysis/check_moving_front_graphs.py','src/thermotaxis/moving_isotherm.py',
             'src/thermotaxis/isothermal_scene.py','src/thermotaxis/tracking_circuit.py',
             'src/thermotaxis/tracking_learning.py','analysis/compare_tracking_topology.py','analysis/train_tracking_network.py']
    plan=dict(status='posthoc_all_graph_and_fresh_case_frozen_replication_no_training',
        models=['real8','rewired8-11','rewired8-22','rewired8-33','gru3','gru16'],training_seeds=[7,17,27],
        cohorts={'previous':[9701,9702],'fresh':[9901,9902]},conditions=previous['design']['conditions'],
        data_episodes_per_field=16,updates=2048,interface='full',
        previous_results_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
        limitation='New cases and all three graphs check robustness; neither cohort is a formal final population estimate.',
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources})
    (out/'design.json').write_text(json.dumps(plan,indent=2)+'\n')
    for name in sources:
        p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,p)
    specs=[]
    for name in plan['models']:
        root=Path('results/tracking-learning-surface-2026-10-24')/('grus' if name.startswith('gru') else 'rates')
        study=json.loads((root/'results.json').read_text());assert study['fitting_complete']
        for seed in plan['training_seeds']:
            row=next(r for r in study['runs'] if r['model']==name and r['seed']==seed and r['episodes_per_field']==16)
            checkpoint=root/row['endpoints']['2048']['checkpoint'];model=make_model(name,'demeaned','full')
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
            specs.append((name,seed,model,str(checkpoint)))
    specs.extend([(n,None,None,None) for n in ['expert','blind']])
    old={(r['model'],r['training_seed'],r['condition'],r['field'],r['seed']):r for r in previous['runs']}
    result=dict(design=plan,runs=[])
    for name,training_seed,model,checkpoint in specs:
        for cohort,seeds in plan['cohorts'].items():
            for label,(amplitude,period) in plan['conditions'].items():
                for field in ['linear','radial']:
                    config=replace(TrackingConfig(),field=field,uniform_wave_amplitude_K=amplitude,uniform_wave_period_s=period)
                    for seed in seeds:
                        key=(name,training_seed,label,field,seed)
                        if key in old:
                            row=dict(old[key],cohort=cohort,reused=True)
                        else:
                            controller=expert_for(config) if name=='expert' else BlindTrackingController() if name=='blind' else NeuralTrackingController(model)
                            rollout=run_tracking(config,controller,seed,record_trajectory=True)
                            row=dict(model=name,training_seed=training_seed,condition=label,field=field,seed=seed,cohort=cohort,reused=False,
                                     initial_state=rollout['initial_state'],moving_metrics=moving_tracking_metrics(config,rollout),
                                     original_metrics=rollout['metrics'],checkpoint=checkpoint)
                        result['runs'].append(row)
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        print('MOVING GRAPH',name,training_seed,flush=True)
    result['source_hashes_unchanged']={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in plan['source_sha256'].items()}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
