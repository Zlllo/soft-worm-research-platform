"""Supervised expressivity/optimization probe, NOT an RL efficiency experiment.

Teacher trajectories and action probabilities come from the known two-timescale
controller. Networks receive the same noisy current temperature stream only.
No teacher internal state is supplied as input. No critic loss is used.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from connectome_pilot import PointBatch, Task, evaluate
from thermotaxis.connectome import CORE, load_connectome
from thermotaxis.connectome_policy import CircuitPolicy, GenericGRUPolicy


def teacher_data(task, seed):
    env = PointBatch(task, seed)
    inputs, probabilities = [], []
    fast = slow = None
    for _ in range(task.steps):
        obs = env.observation()
        err = np.abs(obs.numpy() * 1.5)
        if fast is None:
            fast, slow = err.copy(), err.copy()
        else:
            fast += (1 - np.exp(-task.dt_s / .5)) * (err - fast)
            slow += (1 - np.exp(-task.dt_s / 5.)) * (err - slow)
        rate = np.clip(.1 - 4. * (slow - fast), .01, .5)
        p = -np.expm1(-task.dt_s * rate)
        inputs.append(obs)
        probabilities.append(torch.tensor(p, dtype=torch.float32))
        env.step(env.action_rng.random(task.batch) < p)
    return torch.stack(inputs), torch.stack(probabilities)


def predict(model, inputs):
    h = torch.zeros(inputs.shape[1], model.size)
    probabilities = []
    for obs in inputs:
        p, _, h = model(obs, h)
        probabilities.append(p)
    return torch.stack(probabilities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=256)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path, default=Path('results/connectome-memory-probe-2026-10-24'))
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error('updates must be positive')
    if (args.output / 'design.json').exists():
        parser.error('choose a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    graph = load_connectome().select(CORE)
    task = Task()
    source_paths = ['analysis/connectome_memory_probe.py', 'analysis/connectome_pilot.py',
                    'src/thermotaxis/connectome.py', 'src/thermotaxis/connectome_policy.py']
    design = dict(status='supervised_memory_probe_not_RL', seed=args.seed, updates=args.updates,
        teacher='fixed fast=.5s slow=5s absolute noisy error; rate clip(.1-4*(slow-fast),.01,.5)',
        training_episodes=192, validation_episodes=64, batch=32, sequence_steps=task.steps,
        input='noisy temperature minus fixed preference, scaled by 1.5K',
        loss='MSE on reversal probability / .4; no critic gradient', lr=.003,
        trajectory_seeds=dict(training=82001, validation=82002, closed_loop=90001),
        variants=['anatomical','dense','generic_gru'],
        limitations=['single training seed', 'offline teacher-state distribution',
                     'training targets and preprocessing supply strong task information',
                     'closed-loop evaluation may differ from teacher trajectories',
                     'not an estimate of RL learning efficiency'],
        code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths})
    (args.output / 'design.json').write_text(json.dumps(design, indent=2)+'\n')
    for p in source_paths:
        target = args.output / 'source' / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    train_x, train_y = teacher_data(replace(task,batch=192),82001)
    val_x, val_y = teacher_data(replace(task,batch=64),82002)
    constant = float(((val_y-train_y.mean())/.4).square().mean())
    report = dict(design=design, constant_predictor_validation_mse=constant, runs=[])
    for variant in design['variants']:
        torch.manual_seed(args.seed)
        model = GenericGRUPolicy() if variant=='generic_gru' else CircuitPolicy(graph,variant)
        opt = torch.optim.Adam([p for n,p in model.named_parameters() if not n.startswith('critic.')],lr=.003)
        rng = np.random.default_rng(args.seed)
        curves=[]
        start=time.monotonic()
        for update in range(args.updates+1):
            if update%64==0 or update==args.updates:
                with torch.no_grad():
                    mse=float(((predict(model,val_x)-val_y)/.4).square().mean())
                curves.append(dict(update=update, validation_mse=mse))
                print(f'{variant} update={update} val_mse={mse:.5f} constant={constant:.5f}',flush=True)
            if update==args.updates:
                break
            index=rng.choice(192,32,replace=False)
            p=predict(model,train_x[:,index])
            loss=((p-train_y[:,index])/.4).square().mean()
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        result=dict(variant=variant, parameter_count=model.parameter_count(),curves=curves,
                    closed_loop=evaluate(model,replace(task,batch=64),90001),wall_time_s=time.monotonic()-start)
        report['runs'].append(result)
        torch.save(model.state_dict(),args.output/f'{variant}.pt')
        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
