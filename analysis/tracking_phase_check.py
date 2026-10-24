"""Complete a paired frozen-policy phase intervention; no training changes."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from train_tracking_network import VARIANTS,evaluate
from thermotaxis.isothermal_scene import TrackingConfig
from thermotaxis.tracking_learning import TrackingNetworkConfig,TrackingNetwork


def main():
    root=Path('results/neural-tracking-2026-10-24')
    previous=json.loads((root/'diagnostics/results.json').read_text())
    if (root/'diagnostics/phase_reference.json').exists():
        raise RuntimeError('phase reference already exists')
    plan=dict(status='posthoc_no_training_paired_phase_reference',seeds=[6101,6102],
        purpose='same frozen weights, same config and seeds as prior scrambled intervention',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runs=[])
    (root/'diagnostics/phase_reference_design.json').write_text(json.dumps({k:v for k,v in plan.items() if k!='runs'},indent=2)+'\n')
    for row in previous['runs']:
        variant,seed=row['variant'],row['seed']
        architecture,encoding=VARIANTS[variant]
        model=TrackingNetwork(TrackingNetworkConfig(architecture=architecture,encoding=encoding))
        model.load_state_dict(torch.load(root/f'{variant}-{seed}-final.pt',map_location='cpu',weights_only=True))
        model.eval()
        reference=evaluate(model,TrackingConfig(),plan['seeds'])
        plan['runs'].append(dict(variant=variant,seed=seed,correct_phase=reference,scrambled_phase=row['scrambled_phase']))
    (root/'diagnostics/phase_reference.json').write_text(json.dumps(plan,indent=2)+'\n')
    target=root/'diagnostics/source/analysis/tracking_phase_check.py'
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
