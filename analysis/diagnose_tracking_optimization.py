"""Paired frozen optimization-control checks on eight independent cases."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from compare_tracking_topology import make_model, subset
from train_tracking_network import evaluate, validation, tracking_mean
from thermotaxis.isothermal_scene import TrackingConfig


def main():
    torch.set_num_threads(1)
    output = Path('results/tracking-topology-optimization-diagnostics-2026-10-24')
    if output.exists():
        raise RuntimeError('choose a new diagnostic output; do not overwrite')
    output.mkdir(parents=True)
    sources = ['analysis/diagnose_tracking_optimization.py', 'analysis/compare_tracking_topology.py',
               'src/thermotaxis/tracking_circuit.py']
    plan = dict(status='posthoc_frozen_no_fitting_or_selection', independent_seeds=[9101, 9102, 9103, 9104],
                models=['real8', 'rewired8-11', 'gru16'], training_seeds=[7, 17, 27], episodes_per_field=16,
                update_pairs=[512, 2048], note='Same eight cases as main frozen check; all weights fitted before case results were viewed.',
                source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources})
    (output / 'design.json').write_text(json.dumps(plan, indent=2) + '\n')
    for n in sources:
        target=output/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,target)
    with np.load('results/tracking-topology-data-2026-10-24.npz', allow_pickle=False) as cache:
        data = {k:cache['train_'+k].copy() for k in ('x','y','times','fields')}
        data['episodes'] = json.loads(str(cache['train_episodes']))
    data = subset(data, 16)
    optimization = json.loads(Path('results/tracking-topology-optimization-2026-10-24/results.json').read_text())
    result = dict(design=plan, runs=[])
    for row in optimization['runs']:
        name,seed=row['model'],row['seed']
        old_root = Path('results/tracking-topology-'+('budgets' if name=='gru16' else 'fullio-budgets')+'-2026-10-24')
        model = make_model(name, 'demeaned', 'full')
        record = dict(model=name,seed=seed,stages={})
        for updates,root in [(512,old_root),(2048,Path('results/tracking-topology-optimization-2026-10-24'))]:
            model.load_state_dict(torch.load(root/row['checkpoint'],map_location='cpu',weights_only=True));model.eval()
            rows=evaluate(model,TrackingConfig(),plan['independent_seeds'])
            record['stages'][str(updates)]=dict(behavior=rows,mean_tracking=tracking_mean(rows),training=validation(model,data))
        result['runs'].append(record)
        (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        print(name,seed,'new cases',*[round(record['stages'][str(n)]['mean_tracking'],4) for n in (512,2048)],flush=True)
    result['source_hashes_unchanged']={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in plan['source_sha256'].items()}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
