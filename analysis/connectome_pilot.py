"""Exploratory A2C comparison. No biological or sample-efficiency claim from a pilot.

Run: PYTHONPATH=src python3 analysis/connectome_pilot.py --updates 48 --seeds 7 17 27
Input is ONLY noisy local T minus the fixed, shared preferred T (scaled).
Position, heading, true gradient and reward NEVER enter policy/critic observations.
This one-dimensional point task is not the final embodied thermotaxis experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import time

import numpy as np
import torch

from thermotaxis.connectome import AFD_AIY, CORE, candidate_audit, load_connectome
from thermotaxis.connectome_policy import CircuitPolicy, GenericGRUPolicy


@dataclass(frozen=True)
class Task:
    batch: int = 16
    steps: int = 180
    dt_s: float = 1.
    speed_m_s: float = .0002
    half_extent_m: float = .015
    gradient_K_m: float = 100.
    noise_K: float = .05
    comfort_K: float = .25
    initial_min_m: float = .008
    initial_max_m: float = .012

    def __post_init__(self):
        if self.batch <= 0 or self.steps <= 0 or self.dt_s <= 0:
            raise ValueError('positive batch, steps and dt required')
        if not 0 < self.initial_min_m <= self.initial_max_m < self.half_extent_m:
            raise ValueError('invalid initial range')
        if self.speed_m_s * self.dt_s >= 2 * self.half_extent_m:
            raise ValueError('step exceeds supported reflection distance')


class PointBatch:
    def __init__(self, task, seed):
        self.task = task
        # Independent initial/environment, sensor and action random streams.
        self.initial_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        self.sensor_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.action_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        self.reset()

    def reset(self):
        t = self.task
        self.x = self.initial_rng.uniform(t.initial_min_m, t.initial_max_m, t.batch)
        self.x *= self.initial_rng.choice([-1., 1.], t.batch)
        self.heading = self.initial_rng.choice([-1., 1.], t.batch)
        self.gradient = t.gradient_K_m * self.initial_rng.choice([-1., 1.], t.batch)

    def observation(self):
        measured_error = self.x * self.gradient + self.sensor_rng.normal(0, self.task.noise_K, self.task.batch)
        return torch.tensor(measured_error / 1.5, dtype=torch.float32)

    def step(self, reverse):
        t = self.task
        self.heading *= np.where(reverse, -1., 1.)
        self.x += self.heading * t.speed_m_s * t.dt_s
        upper, lower = self.x > t.half_extent_m, self.x < -t.half_extent_m
        self.x[upper] = 2 * t.half_extent_m - self.x[upper]
        self.x[lower] = -2 * t.half_extent_m - self.x[lower]
        self.heading[upper | lower] *= -1
        error = np.abs(self.x * self.gradient)
        return -error / 1.5, error


def evaluate(policy, task, seed):
    env = PointBatch(task, seed)
    hidden = torch.zeros(task.batch, policy.size) if isinstance(policy, torch.nn.Module) else None
    errors = []
    first_hit = np.full(task.batch, np.nan)
    fast = slow = None
    with torch.no_grad():
        for step in range(task.steps):
            obs = env.observation()
            if hidden is not None:
                probability, _, hidden = policy(obs, hidden)
                p = probability.numpy()
            else:
                err = np.abs(obs.numpy() * 1.5)
                if fast is None:
                    fast, slow = err.copy(), err.copy()
                else:
                    fast += (1 - np.exp(-task.dt_s / .5)) * (err - fast)
                    slow += (1 - np.exp(-task.dt_s / 5.)) * (err - slow)
                rate = np.clip(.1 - 4. * (slow - fast), .01, .5) if policy == 'memory' else np.full(task.batch, .1)
                p = -np.expm1(-task.dt_s * rate)
            _, error = env.step(env.action_rng.random(task.batch) < p)
            errors.append(error)
            hit = (error <= task.comfort_K) & np.isnan(first_hit)
            first_hit[hit] = (step + 1) * task.dt_s
    errors = np.asarray(errors)
    return dict(residence_fraction=float((errors <= task.comfort_K).mean()),
                mean_absolute_error_K=float(errors.mean()),
                entered_fraction=float(np.isfinite(first_hit).mean()),
                per_episode_residence=(errors <= task.comfort_K).mean(axis=0).tolist(),
                per_episode_mae_K=errors.mean(axis=0).tolist())


def train_one(graph, variant, seed, updates, task, eval_every=16, topology_seed=None):
    torch.manual_seed(seed)
    if topology_seed is None:
        topology_seed = seed + 10000
    model = GenericGRUPolicy(task.dt_s) if variant == 'generic_gru' else CircuitPolicy(graph, variant, topology_seed=topology_seed, dt_s=task.dt_s)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    environment = PointBatch(task, seed + 1000)
    curves = []
    start = time.monotonic()
    eval_task = replace(task, batch=64)

    def validation(update):
        metrics = evaluate(model, eval_task, 90001)
        curves.append(dict(update=update, interactions=update * task.batch * task.steps, **metrics))

    validation(0)
    for update in range(1, updates + 1):
        environment.reset()
        hidden = torch.zeros(task.batch, model.size)
        logs, values, rewards, entropies = [], [], [], []
        for _ in range(task.steps):
            probability, value, hidden = model(environment.observation(), hidden)
            dist = torch.distributions.Bernoulli(probs=probability)
            action = torch.tensor(environment.action_rng.random(task.batch) < probability.detach().numpy(), dtype=torch.float32)
            reward, _ = environment.step(action.numpy().astype(bool))
            logs.append(dist.log_prob(action)); entropies.append(dist.entropy())
            values.append(value); rewards.append(torch.tensor(reward, dtype=torch.float32))
        values = torch.stack(values)
        rewards = torch.stack(rewards)
        # Finite-horizon episode ends here; no hidden state or bootstrap across episodes.
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(task.batch)
        following = torch.zeros(task.batch)
        for index in reversed(range(task.steps)):
            delta = rewards[index] + .99 * following - values[index].detach()
            gae = delta + .99 * .95 * gae
            advantages[index] = gae
            following = values[index].detach()
        targets = advantages + values.detach()
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss = -(torch.stack(logs) * normalized).mean() + .5 * (values - targets).square().mean() - .01 * torch.stack(entropies).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite training loss')
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if update % eval_every == 0 or update == updates:
            validation(update)
            print(f'{variant} seed={seed} update={update} validation_residence={curves[-1]["residence_fraction"]:.3f}', flush=True)
    # Pilot holdouts only. Looking at these retires them from future confirmatory tests.
    holdouts = {label: evaluate(model, replace(eval_task, **changes), 91001)
                for label, changes in {'same_distribution': {}, 'more_noise': {'noise_K': .1},
                                       'shallower_gradient': {'gradient_K_m': 70.}}.items()}
    return model, dict(variant=variant, seed=seed, topology_seed=topology_seed,
                       parameter_count=model.parameter_count(), rewiring=model.rewiring,
                       curves=curves, holdouts=holdouts, wall_time_s=time.monotonic() - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=48)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    parser.add_argument('--circuit', choices=['afd_aiy4', 'classic8'], default='classic8')
    parser.add_argument('--variants', nargs='+', choices=['anatomical', 'rewired', 'dense', 'generic_gru'], default=['anatomical', 'rewired', 'dense', 'generic_gru'])
    parser.add_argument('--output', type=Path, default=Path('results/connectome-pilot-2026-10-24'))
    args = parser.parse_args()
    if args.updates < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('positive updates and distinct seeds required')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'design.json').exists():
        parser.error('output already contains a design; choose a new output directory')
    task = Task()
    graph = load_connectome().select(AFD_AIY if args.circuit == 'afd_aiy4' else CORE)
    design = dict(status='exploratory_pilot', biological_fidelity='anatomical connectivity only; learned signed chemical efficacy; simplified rate dynamics',
        task=asdict(task), circuit=args.circuit, variants=args.variants, seeds=args.seeds, updates=args.updates,
        observation=['noisy_local_temperature_minus_fixed_preference'],
        readout='engineering hypothesis: AIY pair (4 cells) or RIA pair (8 cells)',
        comparison='chemical topology ablation; gap topology held fixed; dense baseline has same input/readout bottlenecks; generic_gru has none',
        limitations=['no spiking/cell-specific adaptation', 'no learned cultivation-temperature memory', 'one topology per training seed; pilot not crossed factorial',
                     '1D reflecting point model, no body mechanics', 'not parameter-matched dense baseline', 'all checkpoints retained; no best checkpoint selection'],
        optimizer=dict(name='Adam', lr=.003, gamma=.99, gae_lambda=.95, entropy_coefficient=.01, tau_s=2.),
        data=graph.provenance, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        versions=dict(torch=torch.__version__, numpy=np.__version__),
        code_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ['analysis/connectome_pilot.py', 'src/thermotaxis/connectome.py', 'src/thermotaxis/connectome_policy.py']})
    (args.output / 'design.json').write_text(json.dumps(design, indent=2) + '\n')
    for source in design['code_sha256']:
        target = args.output / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (args.output / 'audit.json').write_text(json.dumps(candidate_audit(), indent=2) + '\n')
    baselines = {name: evaluate(name, replace(task, batch=64), 90001) for name in ('blind', 'memory')}
    results = dict(design=design, baselines=baselines, runs=[])
    for variant in args.variants:
        for seed in args.seeds:
            model, result = train_one(graph, variant, seed, args.updates, task)
            results['runs'].append(result)
            torch.save(model.state_dict(), args.output / f'{variant}-{seed}.pt')
            (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
