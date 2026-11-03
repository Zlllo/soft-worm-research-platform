"""Posthoc failure localization; immutable archived actors and paired interventions.

Run from the checkout root with PYTHONPATH=src. This is mechanism diagnosis on
previously inspected data, not independent confirmation of a topology benefit.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from thermotaxis.connectome import CORE
from thermotaxis.navigation import NavigationConfig
from thermotaxis.navigation_window_policy import WindowActorCritic
from navigation_feasibility_replication import evaluate
from navigation_preflight import summary


ROOT = Path('results/navigation-diagnosis-2026-11-02')
ASSETS = Path('docs/reports/assets/navigation-study-2026-10-25')
PARENT = Path('results/navigation-factorial-2026-10-25')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_actor(model, x):
    """Independent observable replay, checked against the unchanged actor."""
    if model.is_gru:
        raise ValueError('node tracing requires a rate circuit')
    chemical = model.bias.new_zeros(8, 8).index_put((model.pre, model.post), model.chemical)
    g = model.gap_scale * torch.sigmoid(model.gap_logits)
    gap = model.bias.new_zeros(8, 8).index_put((model.gap_i, model.gap_j), g)
    gap = gap.index_put((model.gap_j, model.gap_i), g)
    if model.name == 'isolated8':
        chemical = chemical * torch.eye(8, device=x.device)
        gap = gap * 0
    if model.interface == 'full':
        injection = x[:, :, :6] @ model.input_weights
    else:
        injection = torch.cat([x[:, :, :4] @ model.input_weights[:4],
                               x[:, :, 4:6] @ model.input_weights[4:],
                               x.new_zeros(len(x), 41, 4)], dim=2)
    hidden = x.new_zeros(len(x), 8)
    states, drives = [], []
    alpha = 1 - math.exp(-.1 / .3)
    for t in range(41):
        drive = hidden @ chemical + model.bias + injection[:, t]
        updated = hidden + alpha * (torch.tanh(drive) - hidden + hidden @ gap - hidden * gap.sum(0))
        hidden = torch.where(x[:, t, 6:7] > 0, updated, hidden)
        states.append(hidden)
        drives.append(drive)
    return model.actor(hidden[:, model.outputs]).squeeze(-1), torch.stack(states, 1), torch.stack(drives, 1)


def representation(model, x):
    captured = []
    hook = model.actor.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        model.actor_mean(x)
    finally:
        hook.remove()
    return captured[0]


def temperature_perturbation(x, delta_K):
    """Update centered and mean channels consistently, ignoring padding."""
    mask = x[:, :, 6]
    delta = torch.as_tensor(delta_K, dtype=x.dtype, device=x.device).expand_as(mask) / .05
    average = (delta * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
    y = x.clone()
    y[:, :, 0] += (delta - average) * mask
    y[:, :, 1] += average * mask
    return y


def stimuli():
    phase = torch.arange(41).float()[None, :] * (.1 * 2 * math.pi * .5)
    phase = phase + torch.arange(16).float()[:, None] * (2 * math.pi / 16)
    phase = phase.repeat(5, 1)
    x = torch.zeros(80, 41, 7)
    x[:, :, 1] = torch.tensor([-.10, -.03, 0., .03, .10]).repeat_interleave(16)[:, None] / .05
    x[:, :, 4] = phase.sin()
    x[:, :, 5] = phase.cos()
    x[:, :, 6] = 1
    rng = np.random.default_rng(110201)
    noise = torch.from_numpy(rng.normal(size=(80, 41)).astype('float32'))
    step = torch.zeros(80, 41)
    step[:, 21:] = 1
    return x, dict(level=torch.ones(80, 41), phase=x[:, :, 4],
                   antiphase=-x[:, :, 4], ramp=torch.linspace(-1, 1, 41)[None, :].expand(80, -1),
                   step=step, noise=noise)


def stage_summary(cases):
    n = len(cases)
    def rate(key):
        return sum(c[key] is not None for c in cases) / n
    failure = dict(no_pre_entry=0, entered_without_pre_stability=0,
                   pre_stable_without_post_stability=0, complete=0)
    for c in cases:
        if c['first_entry_s'] is None:
            failure['no_pre_entry'] += 1
        elif c['pre_stable_s'] is None:
            failure['entered_without_pre_stability'] += 1
        elif c['post_stable_s'] is None:
            failure['pre_stable_without_post_stability'] += 1
        else:
            failure['complete'] += 1
    return dict(n=n, first_entry=rate('first_entry_s'), pre_stable=rate('pre_stable_s'),
                post_stable=rate('post_stable_s'), complete=np.mean([c['complete_success'] for c in cases]).item(),
                failures=failure, post_tracking=np.mean([c['post_tracking'] for c in cases]).item())


def prepare():
    if ROOT.exists():
        raise RuntimeError('preserve existing diagnosis output')
    parent = json.loads((PARENT / 'design.json').read_text())
    specs = [s for s in parent['specs'] if s['encoding'] == 'raw']
    sources = [Path(__file__), Path('src/thermotaxis/navigation_window_policy.py'),
               Path('src/thermotaxis/navigation.py'), Path('analysis/navigation_window_ppo.py')]
    teacher = Path('results/navigation-feature-teacher-raw-2026-10-25.npz')
    checkpoints = {}
    for s in specs:
        for name in ['BC.pt', 'model-RL262144.pt']:
            p = PARENT / f'fit-{s["id"]:03d}' / name
            checkpoints[str(p)] = digest(p)
    plan = dict(status='posthoc_mechanistic_diagnosis', specs=specs, seeds=[7, 17, 27],
                amplitudes_K=[.001, .002, .005, .010], stimulus_seed=110201,
                teacher_sample_seed=110202, teacher_windows=1024,
                rescue=dict(names=['real8'], interfaces=['full', 'restricted'],
                            source='BC.pt', updates=1024, batch=256, lr=.003,
                            train_episode_indices=list(range(24)), diagnostic_episode_indices=list(range(24, 32)),
                            evaluation_seeds=list(range(71001, 71033)), field_turn_deg=15),
                limitations=['All teacher episodes were used in original fitting; episode split is not an independent test.',
                             'Existing evaluation cases reused for paired diagnosis, not final confirmation.',
                             'Readout intervention uses BC checkpoints trained under original60deg; evaluates both arms at15deg.',
                             'Synthetic response probes are engineering assays, not calibrated cell physiology.',
                             'No cell-function calibration or whole-circuit retraining in this batch.'],
                inputs_sha256={str(teacher):digest(teacher), **checkpoints},
                source_sha256={str(p):digest(p) for p in sources},
                versions=dict(torch=torch.__version__, numpy=np.__version__))
    write(ROOT / 'design.json', plan)


def check(plan):
    for path, expected in {**plan['source_sha256'], **plan['inputs_sha256']}.items():
        if digest(Path(path)) != expected:
            raise RuntimeError('input/source changed: ' + path)


def failures():
    rows = []
    for filename, label in [('results.json', 'original60deg'), ('feasibility-results.json', 'replication15deg')]:
        original = json.loads((ASSETS / filename).read_text())
        for run in original['runs']:
            s = run['spec']
            if s['encoding'] != 'raw':
                continue
            for endpoint in run['primary']:
                cases = endpoint['cases']
                for group, subset in [('all', cases), ('cold', [c for c in cases if c['initial_error_K'] < 0]),
                                      ('hot', [c for c in cases if c['initial_error_K'] > 0]),
                                      ('linear', [c for c in cases if c['field'] == 'linear']),
                                      ('radial', [c for c in cases if c['field'] == 'radial'])]:
                    if subset:
                        rows.append(dict(scenario=label, spec=s, budget=endpoint['learner_interactions'],
                                         group=group, **stage_summary(subset)))
    write(ROOT / 'failure-decomposition.json', dict(rows=rows, complete=True))
    print('Failure decomposition:', len(rows), flush=True)


def load_model(s, checkpoint):
    torch.manual_seed(s['seed'])
    model = WindowActorCritic(s['name'], s['interface'])
    if checkpoint != 'initial':
        model.load_state_dict(torch.load(PARENT / f'fit-{s["id"]:03d}' / checkpoint,
                                        map_location='cpu', weights_only=True))
    return model.eval()


def signals(plan):
    with np.load('results/navigation-feature-teacher-raw-2026-10-25.npz') as z:
        eligible = np.flatnonzero(z['x'][:, :, 6].sum(1) == 41)
        selected = np.random.default_rng(plan['teacher_sample_seed']).choice(eligible, plan['teacher_windows'], replace=False)
        teacher_x = torch.from_numpy(z['x'][selected].copy())
        teacher_y = torch.from_numpy(z['y'][selected].copy())
    x, patterns = stimuli()
    metrics, profiles = [], []
    with torch.no_grad():
        for s in plan['specs']:
            for checkpoint in ['initial', 'BC.pt', 'model-RL262144.pt']:
                model = load_model(s, checkpoint)
                pred = torch.tanh(model.actor_mean(teacher_x))
                item = dict(spec=s, checkpoint=checkpoint,
                            teacher_mse=float((pred-teacher_y).square().mean()),
                            teacher_saturation=float((pred.abs() > .95).float().mean()),
                            teacher_action_std=float(pred.std(unbiased=False)),
                            teacher_action_correlation=float(np.corrcoef(pred.numpy(), teacher_y.numpy())[0, 1])
                            if float(pred.std()) > 1e-8 else None)
                base = .3 * torch.tanh(model.actor_mean(x))
                if not model.is_gru:
                    replay, states, drives = trace_actor(model, x)
                    torch.testing.assert_close(replay, model.actor_mean(x), rtol=1e-5, atol=1e-6)
                    item.update(node_tanh_saturation=(torch.tanh(drives).abs() > .95).float().mean((0, 1)).tolist(),
                                readout_weight_norm=float(model.actor.weight.norm()))
                metrics.append(item)
                for amp in plan['amplitudes_K']:
                    for kind, pattern in patterns.items():
                        changed = temperature_perturbation(x, pattern * amp)
                        action = .3 * torch.tanh(model.actor_mean(changed))
                        row = dict(spec=s, checkpoint=checkpoint, amplitude_K=amp, pattern=kind,
                                   action_rms_delta_rad_s=float((action-base).square().mean().sqrt()))
                        if not model.is_gru:
                            _, altered, _ = trace_actor(model, changed)
                            row['node_final_rms_delta'] = (altered[:, -1]-states[:, -1]).square().mean(0).sqrt().tolist()
                        profiles.append(row)
            print('Signal probes:', s['id'], s['name'], s['interface'], s['seed'], flush=True)
    write(ROOT / 'signals.json', dict(neurons=list(CORE), teacher_indices=selected.tolist(),
                                      metrics=metrics, profiles=profiles, complete=True))


def rescue(plan):
    design = plan['rescue']
    with np.load('results/navigation-feature-teacher-raw-2026-10-25.npz') as z:
        xs, ys = z['x'].copy(), z['y'].copy()
    # Original storage is [time, episode] flattened, not contiguous episodes.
    episodes = np.arange(len(ys)) % 32
    train_ids = np.flatnonzero(np.isin(episodes, design['train_episode_indices']))
    diagnostic_ids = np.flatnonzero(np.isin(episodes, design['diagnostic_episode_indices']))
    config = replace(NavigationConfig(), field_turn_rad=math.pi/12)
    rows = []
    for s in plan['specs']:
        if s['name'] not in design['names'] or s['interface'] not in design['interfaces']:
            continue
        dest = ROOT / f'readout-{s["id"]:03d}.json'
        if dest.exists():
            rows.append(json.loads(dest.read_text()))
            continue
        model = load_model(s, design['source'])
        before = {k:v.detach().clone() for k,v in model.state_dict().items()}
        reps = []
        with torch.no_grad():
            for start in range(0, len(xs), 1024):
                reps.append(representation(model, torch.from_numpy(xs[start:start+1024])).detach())
        features = torch.cat(reps)
        targets = torch.from_numpy(ys)
        def mse(indices):
            with torch.no_grad():
                return float((torch.tanh(model.actor(features[indices]).squeeze(-1))-targets[indices]).square().mean())
        baseline = dict(train_mse=mse(train_ids), diagnostic_mse=mse(diagnostic_ids),
                        cases=evaluate(model, config, design['evaluation_seeds']))
        baseline['summary'] = summary(baseline['cases'])
        optimizer = torch.optim.Adam(model.actor.parameters(), lr=design['lr'])
        rng = np.random.default_rng(110300+s['seed'])
        for _ in range(design['updates']):
            idx = rng.choice(train_ids, design['batch'], replace=True)
            loss = (torch.tanh(model.actor(features[idx]).squeeze(-1))-targets[idx]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite readout loss')
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        unchanged = all(torch.equal(v, model.state_dict()[k]) for k,v in before.items() if not k.startswith('actor.'))
        if not unchanged:
            raise RuntimeError('readout intervention modified circuit')
        after = dict(train_mse=mse(train_ids), diagnostic_mse=mse(diagnostic_ids),
                     cases=evaluate(model, config, design['evaluation_seeds']))
        after['summary'] = summary(after['cases'])
        torch.save(model.state_dict(), ROOT / f'readout-{s["id"]:03d}.pt')
        row = dict(spec=s, baseline=baseline, readout_refit=after, circuit_unchanged=unchanged)
        write(dest, row); rows.append(row)
        print('Readout intervention:', s, baseline['summary']['complete_success'],
              after['summary']['complete_success'], flush=True)
    write(ROOT / 'readout-rescue.json', dict(design=design, runs=rows, complete=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'failures', 'signals', 'rescue'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == 'prepare':
        prepare()
        return
    plan = json.loads((ROOT / 'design.json').read_text())
    check(plan)
    if args.stage == 'failures':
        failures()
    elif args.stage == 'signals':
        signals(plan)
    else:
        rescue(plan)
    check(plan)


if __name__ == '__main__':
    main()
