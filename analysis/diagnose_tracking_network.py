"""Post-training paired diagnostic evaluation; no new model fitting/selection."""
import argparse
from dataclasses import replace,asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from train_tracking_network import VARIANTS,evaluate,expert_for
from thermotaxis.isothermal_scene import TrackingConfig,run_tracking
from thermotaxis.tracking_learning import TrackingNetworkConfig,TrackingNetwork,NeuralTrackingController


def mean_tracking(rows):
    return float(np.mean([r['metrics']['tracking_fraction'] for r in rows]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('results/neural-tracking-2026-10-24'))
    args=parser.parse_args()
    output=args.root/'diagnostics'
    if (output/'design.json').exists():
        parser.error('diagnostic design exists')
    output.mkdir(parents=True,exist_ok=True)
    original=json.loads((args.root/'results.json').read_text())
    sources=['analysis/diagnose_tracking_network.py','analysis/train_tracking_network.py',
             'src/thermotaxis/tracking_learning.py','src/thermotaxis/isothermal_scene.py']
    conditions={**original['design']['transfer_conditions'],
                'smaller_radius':{'target_radius_m':.003},'larger_radius':{'target_radius_m':.008},
                'slower_sweep':{'head_frequency_Hz':.4},'faster_sweep':{'head_frequency_Hz':.75}}
    plan=dict(status='posthoc_exploratory_evaluation_no_training',
        original_planned_transfer_baselines='expert/blind added at original transfer seeds6001/6002',
        paired_stage1='all stage1 checkpoints evaluated on same final basic cases as final models; no training follows',
        additional_conditions=conditions,additional_seeds=[6101,6102],
        phase_intervention='frozen final model, scrambled reported phase, same physical sweep and noise',
        expert_caveat='expert window is two known actuator cycles; network horizon fixed at 4s',
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    (output/'design.json').write_text(json.dumps(plan,indent=2)+'\n')
    for p in sources:
        target=output/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    base=TrackingConfig(**original['design']['scene'])
    result=dict(design=plan,baselines={},runs=[])
    for condition,changes in conditions.items():
        config=replace(base,**changes)
        seeds=original['design']['episode_splits']['transfer'] if condition in original['design']['transfer_conditions'] else plan['additional_seeds']
        result['baselines'][condition]={name:evaluate(name,config,seeds) for name in ('expert','blind')}
    for run in original['runs']:
        variant,seed=run['variant'],run['seed']
        architecture,encoding=VARIANTS[variant]
        model=TrackingNetwork(TrackingNetworkConfig(architecture=architecture,encoding=encoding))
        model.load_state_dict(torch.load(args.root/f'{variant}-{seed}-stage1.pt',map_location='cpu',weights_only=True))
        model.eval()
        stage1=evaluate(model,base,original['design']['episode_splits']['final_exploratory'])
        model.load_state_dict(torch.load(args.root/f'{variant}-{seed}-final.pt',map_location='cpu',weights_only=True))
        model.eval()
        extra={}
        for label in ('smaller_radius','larger_radius','slower_sweep','faster_sweep'):
            extra[label]=evaluate(model,replace(base,**conditions[label]),plan['additional_seeds'])
        intervention=[]
        for field in ('linear','radial'):
            for case_seed in plan['additional_seeds']:
                row=run_tracking(replace(base,field=field),NeuralTrackingController(model),case_seed,phase_mode='scrambled')
                intervention.append(dict(field=field,seed=case_seed,metrics=row['metrics']))
        result['runs'].append(dict(variant=variant,seed=seed,stage1_same_final_cases=stage1,
            final_basic=run['final_behavior'],additional_transfer=extra,scrambled_phase=intervention))
        (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        print(variant,seed,'paired stage1/final',round(mean_tracking(stage1),3),
              round(mean_tracking(run['final_behavior']),3),'phase scrambled',round(mean_tracking(intervention),3),flush=True)
    result['source_hashes_unchanged']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in plan['source_sha256'].items()}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
