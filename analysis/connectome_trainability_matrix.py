"""Predeclared supervised controls for input, capacity and explicit memory.

Training uses 192 teacher-generated episodes, never teacher internal states.
These probes distinguish optimization from task feasibility, not RL efficiency.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from connectome_pilot import Task, evaluate
from connectome_memory_probe import teacher_data, predict
from thermotaxis.learning_probes import PROBES, make_probe


def scalar_metrics(metrics):
    return {k: v for k, v in metrics.items() if not k.startswith('per_episode_')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    parser.add_argument('--variants', nargs='+', choices=PROBES, default=list(PROBES))
    parser.add_argument('--output', type=Path, default=Path('results/trainability-matrix-2026-10-24'))
    args = parser.parse_args()
    if args.updates < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('positive updates and distinct training seeds required')
    if (args.output / 'design.json').exists():
        parser.error('output has an existing design; use a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    task = Task()
    paths = ['analysis/connectome_trainability_matrix.py', 'analysis/connectome_memory_probe.py',
             'analysis/connectome_pilot.py', 'src/thermotaxis/learning_probes.py',
             'src/thermotaxis/connectome.py', 'src/thermotaxis/connectome_policy.py']
    design = dict(status='supervised_trainability_controls_not_RL_efficiency', task=asdict(task),
        variants=args.variants, seeds=args.seeds, updates=args.updates, optimizer='Adam', lr=.003,
        batch=32, training_episodes=192, validation_episodes=64, sequence_steps=task.steps,
        initial_actor='zero GRU head; filter gain=0 bias=.255; all initial rates=.255/s',
        differences=dict(signed_gru16='capacity control relative to signed_gru2',
            absolute_gru16='only replace signed input by absolute input relative to signed_gru16',
            fixed_error_filters='positive control: teacher time constants known, error-filter features, clipped linear rate readout',
            learned_error_filters='same filter class but learn taus from .8s and 3s'),
        validation_selection='final budget model only; no best-checkpoint selection',
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        seeds_description=dict(teacher_training=82001, teacher_validation=82002, behavior_validation=90001,
                               exploratory_holdouts=93001),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        versions=dict(torch=torch.__version__,numpy=np.__version__),
        limitations=['supervised teacher supplies known task mechanism', 'explicit filters are not AFD models',
                     'capacity and filter model comparisons do not isolate a single feature',
                     'input sign ablation has equal architecture and parameter count',
                     'teacher trajectories are reused, not independent environment samples per update',
                     'new holdouts are exploratory once inspected'])
    (args.output / 'design.json').write_text(json.dumps(design,indent=2)+'\n')
    for path in paths:
        target=args.output/'source'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    train_x,train_y=teacher_data(replace(task,batch=192),82001)
    val_x,val_y=teacher_data(replace(task,batch=64),82002)
    constant=float(((val_y-train_y.mean())/.4).square().mean())
    teacher=scalar_metrics(evaluate('memory',replace(task,batch=64),90001))
    report=dict(design=design,constant_predictor_mse=constant,teacher_behavior=teacher,runs=[])
    for variant in args.variants:
        for seed in args.seeds:
            torch.manual_seed(seed)
            model=make_probe(variant)
            optimizer=torch.optim.Adam(model.parameters(),lr=.003)
            rng=np.random.default_rng(seed)
            start=time.monotonic()
            curves=[]
            for update in range(args.updates+1):
                if update%128==0 or update==args.updates:
                    with torch.no_grad():
                        predicted=predict(model,val_x)
                        mse=float(((predicted-val_y)/.4).square().mean())
                    curves.append(dict(update=update,validation_mse=mse))
                    print(f'{variant} seed={seed} update={update} mse={mse:.6f}',flush=True)
                if update==args.updates:
                    break
                index=rng.choice(192,32,replace=False)
                predicted=predict(model,train_x[:,index])
                loss=((predicted-train_y[:,index])/.4).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            metrics=evaluate(model,replace(task,batch=64),90001)
            holdouts={name:scalar_metrics(evaluate(model,replace(task,batch=64,**change),93001))
                for name,change in {'new_random_draws':{},'more_noise':{'noise_K':.1},
                                    'shallower_gradient':{'gradient_K_m':70.}}.items()}
            learned={k:v.detach().tolist() for k,v in model.named_parameters()} if 'filters' in variant else None
            run=dict(variant=variant,seed=seed,parameters=sum(p.numel() for p in model.parameters()),
                     curves=curves,behavior_validation=metrics,exploratory_holdouts=holdouts,
                     fitted_filter_parameters=learned,wall_time_s=time.monotonic()-start)
            if 'filters' in variant:
                run['fitted_taus_s']=model.taus_s.detach().tolist()
            report['runs'].append(run)
            torch.save(model.state_dict(),args.output/f'{variant}-{seed}.pt')
            (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
            print(f'{variant} seed={seed} residence={metrics["residence_fraction"]:.4f}',flush=True)


if __name__=='__main__':
    main()
