"""Frozen-checkpoint training error and independent-case checks; no fitting."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from compare_tracking_topology import make_model, subset
from train_tracking_network import dataset_digest, evaluate, validation, tracking_mean
from thermotaxis.isothermal_scene import TrackingConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--data-cache', type=Path, default=Path('results/tracking-topology-data-2026-10-24.npz'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    design_path = args.output / 'design.json'
    if args.plan_only:
        if args.output.exists():
            parser.error('choose an unused diagnostics output')
        args.output.mkdir(parents=True)
        paths = [Path(__file__).resolve().relative_to(Path.cwd().resolve()), Path('analysis/compare_tracking_topology.py'),
                 Path('src/thermotaxis/tracking_circuit.py')]
        plan = dict(status='exploratory_frozen_checkpoint_checks_no_training', roots=[str(p) for p in args.roots],
                    independent_behavior_seeds=[9101, 9102, 9103, 9104],
                    independent_behavior_budget_per_field=64, scene=asdict(TrackingConfig()),
                    training_error='all windows in each fitted whole-episode subset; no resampling',
                    selection='all scheduled budget runs, no performance-dependent checkpoint selection',
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        design_path.write_text(json.dumps(plan, indent=2) + '\n')
        for p in paths:
            target = args.output / 'source' / p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        print('Independent-case check plan saved; no fitting', flush=True)
        return
    plan = json.loads(design_path.read_text())
    if [str(p) for p in args.roots] != plan['roots']:
        raise RuntimeError('root list differs from saved plan')
    if (args.output / 'results.json').exists():
        raise RuntimeError('diagnostic result exists')
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    with np.load(args.data_cache, allow_pickle=False) as cache:
        training = {k: cache['train_' + k].copy() for k in ('x', 'y', 'times', 'fields')}
        training['episodes'] = json.loads(str(cache['train_episodes']))
        fingerprint = str(cache['fingerprint'])
    base = TrackingConfig(**plan['scene'])
    result = dict(design=plan, runs=[],
                  independent_baselines={name: evaluate(name, base, plan['independent_behavior_seeds'])
                                         for name in ('expert', 'blind')})
    for root in args.roots:
        source = json.loads((root / 'results.json').read_text())
        assert fingerprint == source['data_fingerprint']
        assert all(source['source_hashes_unchanged'].values()), root
        expected = (len(source['design']['models']) * len(source['design']['training_seeds']) *
                    len(source['design']['episodes_per_field']))
        assert len(source['runs']) == expected
        for row in source['runs']:
            model = make_model(row['model'], source['design']['encoding'], source['design']['interface'])
            if hasattr(model, 'structural_metadata'):
                assert model.structural_metadata()['chemical_mask'] == row['metadata']['chemical_mask']
            model.load_state_dict(torch.load(root / row['checkpoint'], map_location='cpu', weights_only=True))
            model.eval()
            data = subset(training, row['episodes_per_field'])
            assert dataset_digest(data) == source['datasets'][str(row['episodes_per_field'])]['digest']
            info = dict(root=str(root), model=row['model'], seed=row['seed'],
                        episodes_per_field=row['episodes_per_field'], training=validation(model, data))
            if row['episodes_per_field'] == plan['independent_behavior_budget_per_field']:
                rows = evaluate(model, base, plan['independent_behavior_seeds'])
                info['independent_behavior'] = rows
                info['independent_mean_tracking'] = tracking_mean(rows)
            result['runs'].append(info)
            (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        print(root, 'checked', expected, 'frozen checkpoints', flush=True)
    result['source_hashes_unchanged'] = {n: hashlib.sha256(Path(n).read_bytes()).hexdigest() == h
                                         for n, h in plan['source_sha256'].items()}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
