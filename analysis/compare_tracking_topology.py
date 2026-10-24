"""Fixed-data topology learning curves; supervised imitation, never RL.

Preflight is a separately recorded trainability check. The budget experiment
uses nested whole-episode subsets, fixed updates, crossed graph/training seeds,
and no learner-specific aggregation or checkpoint selection.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from train_tracking_network import collect, dataset_digest, evaluate, validation, tracking_mean
from thermotaxis.connectome import CORE, load_connectome
from thermotaxis.tracking_learning import TrackingNetwork, TrackingNetworkConfig
from thermotaxis.tracking_circuit import CircuitTrackingNetwork
from thermotaxis.isothermal_scene import TrackingConfig


MODELS = ('real8', 'rewired8-11', 'rewired8-22', 'rewired8-33', 'dense8', 'gru4', 'gru16')
SOURCES = ('analysis/compare_tracking_topology.py', 'analysis/train_tracking_network.py',
           'src/thermotaxis/tracking_circuit.py', 'src/thermotaxis/tracking_learning.py',
           'src/thermotaxis/isothermal_scene.py', 'src/thermotaxis/connectome.py')


def make_model(name, encoding, interface='restricted'):
    config = TrackingNetworkConfig(encoding=encoding)
    if name.startswith('gru'):
        return TrackingNetwork(replace(config, hidden_size=int(name[3:])))
    topology = 'rewired' if name.startswith('rewired') else 'dense' if name == 'dense8' else 'anatomical'
    graph_seed = int(name.split('-')[1]) if '-' in name else 0
    return CircuitTrackingNetwork(load_connectome().select(CORE), config, topology,
                                  graph_seed, tau_s=.3, interface=interface)


def fit(model, training, val, updates, seed):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    curves = []
    model.train()
    for update in range(updates + 1):
        if update % 128 == 0 or update == updates:
            curves.append(dict(update=update, validation=validation(model, val)))
        if update == updates:
            break
        index = rng.integers(0, len(training['y']), 64)
        prediction = model(torch.from_numpy(training['x'][index]))
        target = torch.from_numpy(training['y'][index])
        loss = ((prediction - target) / model.config.max_turn_rate_rad_s).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    model.eval()
    return curves


def subset(data, episode_budget):
    # collect() concatenates full episodes, field-major then seed-major.
    counts = [row['labels'] for row in data['episodes']]
    offsets = np.cumsum([0] + counts)
    selected = []
    seen = {'linear': 0, 'radial': 0}
    episodes = []
    for i, row in enumerate(data['episodes']):
        if seen[row['field']] < episode_budget:
            selected.extend(range(offsets[i], offsets[i + 1]))
            episodes.append(row)
        seen[row['field']] += 1
    if min(seen.values()) < episode_budget:
        raise ValueError('insufficient complete episodes')
    result = {k: data[k][selected] for k in ('x', 'y', 'times', 'fields')}
    result['episodes'] = episodes
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['preflight', 'budgets'], default='preflight')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--encoding', choices=['raw', 'demeaned'], default='demeaned')
    parser.add_argument('--interface', choices=['restricted', 'full'], default='restricted')
    parser.add_argument('--seeds', nargs='+', type=int, default=[7])
    parser.add_argument('--budgets', nargs='+', type=int, default=[64])
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--data-cache', type=Path, default=Path('results/tracking-topology-data-2026-10-24.npz'))
    args = parser.parse_args()
    if args.output.exists() or args.updates < 1 or not args.budgets or not 1 <= min(args.budgets) <= max(args.budgets) <= 64:
        parser.error('unused output path, positive updates, and episode budgets in [1,64] required')
    if any(len(x) != len(set(x)) for x in (args.seeds, args.budgets, args.models)):
        parser.error('duplicate experiment factors')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    base = TrackingConfig()
    training_scene = replace(base, duration_s=60., initial_error_fraction=.8, initial_heading_jitter_rad=.7)
    design = dict(status='exploratory_supervised_topology_' + args.mode,
        scene=asdict(base), training_scene=asdict(training_scene), models=args.models,
        encoding=args.encoding, interface=args.interface, training_seeds=args.seeds,
        episodes_per_field=args.budgets, updates=args.updates,
        optimizer=dict(name='Adam', lr=.003, batch=64, clipping=1.),
        training_episode_seeds=list(range(1000, 1064)), validation_episode_seeds=list(range(3000, 3008)),
        behavior_seeds=[8101, 8102], behavior_fields=['linear', 'radial'],
        initialization='torch seed per model/training seed; same chemical parameter vector for real/rewired',
        subset_rule='nested full episodes, first B seeds in each field; same optimizer updates at every budget',
        checkpoint_rule='fixed final update; no model-specific aggregation or best-checkpoint choice',
        topology_ablation='chemical layout only; node-labelled degrees, loops, gap, I/O and dynamics fixed',
        tau_s=.3, source_sha256={n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in SOURCES},
        versions=dict(torch=torch.__version__, numpy=np.__version__),
        limitations=['engineering AIY phase input and RIA output; no biological dynamics claim',
            'GRU/dense capacities differ; only real/rewired parameter matched',
            'demeaning retains mean and is not an established AFD mechanism',
            'fixed updates confound fewer trajectories with more repeated exposure',
            'preflight uses a single training seed; graph seeds and training seeds are distinct factors',
            'behavior cohort is exploratory; not RL sample efficiency or soft-body evidence'])
    (args.output / 'design.json').write_text(json.dumps(design, indent=2) + '\n')
    for name in SOURCES:
        target = args.output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    config = TrackingNetworkConfig()
    # Cache bytes are accepted only when the fixed scene/episode fingerprint matches.
    fingerprint = hashlib.sha256(json.dumps(dict(scene=asdict(training_scene),
        train=design['training_episode_seeds'], val=design['validation_episode_seeds'],
        sources={n:design['source_sha256'][n] for n in SOURCES if n != SOURCES[0] and n != SOURCES[2]}),
        sort_keys=True).encode()).hexdigest()
    if args.data_cache.exists():
        with np.load(args.data_cache, allow_pickle=False) as cache:
            if str(cache['fingerprint']) != fingerprint:
                raise RuntimeError('cache fingerprint mismatch')
            training = {k: cache['train_' + k].copy() for k in ('x', 'y', 'times', 'fields')}
            val = {k: cache['val_' + k].copy() for k in ('x', 'y', 'times', 'fields')}
            training['episodes'] = json.loads(str(cache['train_episodes']))
            val['episodes'] = json.loads(str(cache['val_episodes']))
        print('Loaded fixed teacher data', flush=True)
    else:
        print('Generating fixed teacher data', flush=True)
        training = collect(training_scene, config, design['training_episode_seeds'])
        val = collect(training_scene, config, design['validation_episode_seeds'])
        args.data_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.data_cache, fingerprint=np.array(fingerprint),
            train_episodes=np.array(json.dumps(training['episodes'])), val_episodes=np.array(json.dumps(val['episodes'])),
            **{'train_' + k: training[k] for k in ('x', 'y', 'times', 'fields')},
            **{'val_' + k: val[k] for k in ('x', 'y', 'times', 'fields')})
    report = dict(design=design, data_fingerprint=fingerprint, full_training_data=dataset_digest(training),
                  validation_data=dataset_digest(val), datasets={}, runs=[],
                  baselines={name: evaluate(name, base, design['behavior_seeds']) for name in ('expert', 'blind')})
    for budget in args.budgets:
        data = subset(training, budget)
        report['datasets'][str(budget)] = dict(digest=dataset_digest(data), episodes=data['episodes'])
        for name in args.models:
            for seed in args.seeds:
                started = time.monotonic()
                torch.manual_seed(seed)
                model = make_model(name, args.encoding, args.interface)
                metadata = model.structural_metadata() if isinstance(model, CircuitTrackingNetwork) else dict(
                    parameters=sum(p.numel() for p in model.parameters()), hidden_size=model.config.hidden_size)
                initial = validation(model, val)
                initial_behavior = evaluate(model, base, design['behavior_seeds'])
                curves = fit(model, data, val, args.updates, seed)
                behavior = evaluate(model, base, design['behavior_seeds'])
                filename = f'{name}-{seed}-episodes{budget}.pt'
                torch.save(model.state_dict(), args.output / filename)
                row = dict(model=name, seed=seed, episodes_per_field=budget, labels=len(data['y']),
                    metadata=metadata, initial_validation=initial, initial_behavior=initial_behavior,
                    curves=curves, behavior=behavior,
                    mean_tracking=tracking_mean(behavior), checkpoint=filename,
                    wall_time_s=time.monotonic() - started)
                report['runs'].append(row)
                (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
                print(name, seed, budget, 'tracking', round(row['mean_tracking'], 4),
                      'val_mse', round(curves[-1]['validation']['all']['normalized_mse'], 5),
                      'seconds', round(row['wall_time_s'], 1), flush=True)
    report['source_hashes_unchanged'] = {n: hashlib.sha256(Path(n).read_bytes()).hexdigest() == h
                                         for n, h in design['source_sha256'].items()}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
