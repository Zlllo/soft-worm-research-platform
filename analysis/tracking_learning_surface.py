"""Data x optimization x capacity experiment with validated endpoint reuse.

Fitting and fresh-case evaluation are separate stages. The previous experiments
remain immutable; reused endpoints and exact 512-update prefixes are recorded.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from compare_tracking_topology import make_model, subset, SOURCES
from train_tracking_network import dataset_digest, validation, evaluate, tracking_mean
from thermotaxis.isothermal_scene import TrackingConfig


MODELS = ['real8', 'rewired8-11', 'rewired8-22', 'rewired8-33', 'gru2', 'gru3', 'gru16']
PARENTS = [Path('results/tracking-topology-fullio-budgets-2026-10-24'),
           Path('results/tracking-topology-budgets-2026-10-24'),
           Path('results/tracking-topology-optimization-2026-10-24')]


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def cache_data():
    with np.load('results/tracking-topology-data-2026-10-24.npz', allow_pickle=False) as c:
        training = {k: c['train_' + k].copy() for k in ('x', 'y', 'times', 'fields')}
        val = {k: c['val_' + k].copy() for k in ('x', 'y', 'times', 'fields')}
        training['episodes'] = json.loads(str(c['train_episodes']))
        return training, val, str(c['fingerprint'])


def parent_endpoints():
    endpoints = {}
    for root in PARENTS:
        result = json.loads((root / 'results.json').read_text())
        assert all(result['source_hashes_unchanged'].values()), root
        for row in result['runs']:
            if row['model'] not in MODELS:
                continue
            if not row['model'].startswith('gru') and result['design']['interface'] != 'full':
                continue
            key = (row['model'], row['seed'], row['episodes_per_field'], result['design']['updates'])
            item = dict(root=str(root), checkpoint=str(root / row['checkpoint']),
                        validation=row['curves'][-1]['validation'], curves=row['curves'])
            if key in endpoints:
                assert endpoints[key]['validation'] == item['validation'], key
            endpoints[key] = item
    return endpoints


def fit_one(name, seed, budget, data, val, parents, output):
    torch.manual_seed(seed)
    model = make_model(name, 'demeaned', 'full')
    metadata = model.structural_metadata() if hasattr(model, 'structural_metadata') else dict(
        parameters=sum(p.numel() for p in model.parameters()), hidden_size=model.config.hidden_size)
    record = dict(model=name, seed=seed, episodes_per_field=budget, labels=len(data['y']), metadata=metadata, endpoints={})
    full_key = (name, seed, budget, 2048)
    if full_key in parents:
        for updates in (512, 2048):
            source = parents[(name, seed, budget, updates)]
            filename = f'{name}-{seed}-B{budget}-U{updates}.pt'
            shutil.copyfile(source['checkpoint'], output / filename)
            record['endpoints'][str(updates)] = dict(checkpoint=filename, validation=source['validation'],
                                                    reused_from=source['checkpoint'])
        record['curves'] = parents[full_key]['curves']
        record['new_fit'] = False
        return record
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    rng = np.random.default_rng(seed)
    curves = []
    model.train()
    for update in range(2049):
        if update % 128 == 0:
            metrics = validation(model, val)
            curves.append(dict(update=update, validation=metrics))
            if update in (512, 2048):
                filename = f'{name}-{seed}-B{budget}-U{update}.pt'
                torch.save(model.state_dict(), output / filename)
                endpoint = dict(checkpoint=filename, validation=metrics, reused_from=None)
                key = (name, seed, budget, update)
                if key in parents:
                    assert metrics == parents[key]['validation'], ('parent prefix mismatch', key)
                    endpoint['exact_parent_validation_match'] = parents[key]['checkpoint']
                record['endpoints'][str(update)] = endpoint
        if update == 2048:
            break
        index = rng.integers(0, len(data['y']), 64)
        y = torch.from_numpy(data['y'][index])
        prediction = model(torch.from_numpy(data['x'][index]))
        loss = ((prediction - y) / model.config.max_turn_rate_rad_s).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite surface training loss')
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    record['curves'] = curves
    record['new_fit'] = True
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--stage', choices=['fit', 'evaluate'], default='fit')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    design_path = args.output / 'design.json'
    if not design_path.exists():
        if args.stage != 'fit' or args.output.exists():
            parser.error('new fitting requires an unused output directory')
        args.output.mkdir(parents=True)
        sources = list(SOURCES) + ['analysis/tracking_learning_surface.py']
        design = dict(status='exploratory_data_optimization_capacity_surface', models=args.models,
                      training_seeds=[7, 17, 27], episodes_per_field=[4, 16, 64], update_endpoints=[512, 2048],
                      fresh_behavior_seeds=[9301, 9302, 9303, 9304], scene=asdict(TrackingConfig()),
                      interface='full', encoding='demeaned', optimizer=dict(lr=.003,batch=64,clipping=1.),
                      parent_directories=[str(p) for p in PARENTS],
                      source_sha256={n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources},
                      constraints=['No case evaluation before this shard finishes fitting.',
                         'Fixed nested trajectory subsets, not independent data-subset repetitions.',
                         'Only data used for fitting counts toward displayed training episode budgets; fixed validation excluded.',
                         'Reuse exact previously fitted endpoints; do not count duplicates as independent runs.',
                         'Full node input/output interface is engineering; anatomical inference limited to chemical layout.',
                         'Two optimizer budgets are not a proof of globally sufficient optimization.'])
        atomic_json(design_path, design)
        for name in sources:
            target = args.output / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, target)
    design = json.loads(design_path.read_text())
    if design['models'] != args.models:
        raise RuntimeError('model list differs from saved design')
    for name, digest in design['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    result_path = args.output / 'results.json'
    if args.stage == 'fit':
        if result_path.exists() and not args.resume:
            raise RuntimeError('use --resume for an interrupted fitting run')
        data, val, fingerprint = cache_data()
        parent = json.loads((PARENTS[0] / 'results.json').read_text())
        assert fingerprint == parent['data_fingerprint']
        assert dataset_digest(data) == parent['full_training_data']
        assert dataset_digest(val) == parent['validation_data']
        result = json.loads(result_path.read_text()) if result_path.exists() else dict(
            design=design, data_fingerprint=fingerprint, training_data=dataset_digest(data), validation_data=dataset_digest(val), runs=[])
        parents = parent_endpoints()
        completed = {(r['model'], r['seed'], r['episodes_per_field']) for r in result['runs']}
        for budget in design['episodes_per_field']:
            selected = subset(data, budget)
            for name in args.models:
                for seed in design['training_seeds']:
                    if (name,seed,budget) in completed:
                        continue
                    row = fit_one(name, seed, budget, selected, val, parents, args.output)
                    result['runs'].append(row)
                    atomic_json(result_path, result)
                    print('FIT',name,seed,budget,'new' if row['new_fit'] else 'reused',flush=True)
        result['fitting_complete'] = True
        result['source_hashes_unchanged'] = {n:hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in design['source_sha256'].items()}
        atomic_json(result_path, result)
    else:
        result = json.loads(result_path.read_text())
        if not result.get('fitting_complete'):
            raise RuntimeError('all planned fits must finish before case evaluation')
        evaluation_path = args.output / 'fresh-cases.json'
        if evaluation_path.exists() and not args.resume:
            raise RuntimeError('use --resume for an interrupted evaluation')
        evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else dict(design=design,runs=[])
        completed = {(r['model'],r['seed'],r['episodes_per_field'],r['updates']) for r in evaluation['runs']}
        base = TrackingConfig(**design['scene'])
        for row in result['runs']:
            model = make_model(row['model'], 'demeaned', 'full')
            for updates in design['update_endpoints']:
                key = (row['model'], row['seed'], row['episodes_per_field'], updates)
                if key in completed:
                    continue
                model.load_state_dict(torch.load(args.output/row['endpoints'][str(updates)]['checkpoint'],map_location='cpu',weights_only=True));model.eval()
                rows = evaluate(model, base, design['fresh_behavior_seeds'])
                evaluation['runs'].append(dict(model=row['model'],seed=row['seed'],episodes_per_field=row['episodes_per_field'],updates=updates,
                                               behavior=rows,mean_tracking=tracking_mean(rows)))
                atomic_json(evaluation_path,evaluation)
            print('CASES',row['model'],row['seed'],row['episodes_per_field'],flush=True)
        evaluation['complete'] = True
        atomic_json(evaluation_path,evaluation)


if __name__=='__main__':
    main()
