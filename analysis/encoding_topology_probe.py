"""Cross sensory encoding with chemical topology; exploratory supervised study.

Design, code snapshots and full budget are saved before any training.
No architecture is given absolute error, teacher states, position or gradient.
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
from thermotaxis.connectome import CORE, load_connectome
from thermotaxis.encoding_policies import ARCHITECTURES, make_encoded_policy


def trim(metrics):
    return {k:v for k,v in metrics.items() if not k.startswith('per_episode_')}


def response_probe(model):
    """Identical current observation with opposite recent temperature histories."""
    result = {}
    with torch.no_grad():
        for side in (-1., 1.):
            labels={}
            for label, start in [('toward',side*.8),('away',side*.1)]:
                series = [start]*30 + np.linspace(start,side*.4,30).tolist()
                h=torch.zeros(1,model.size)
                for value in series:
                    p,_,h=model(torch.tensor([value],dtype=torch.float32),h)
                labels[label]=float(p.item())
            result['cold' if side<0 else 'hot']=labels
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--updates',type=int,default=512)
    parser.add_argument('--seeds',nargs='+',type=int,default=[7,17,27])
    parser.add_argument('--topology-seeds',nargs='+',type=int,default=[10001,10002,10003])
    parser.add_argument('--architectures',nargs='+',choices=ARCHITECTURES,default=list(ARCHITECTURES))
    parser.add_argument('--encodings',nargs='+',choices=['raw','adaptive'],default=['raw','adaptive'])
    parser.add_argument('--output',type=Path,default=Path('results/encoding-topology-2026-10-24'))
    args=parser.parse_args()
    if args.updates<1 or len(set(args.seeds))!=len(args.seeds) or len(set(args.topology_seeds))!=len(args.topology_seeds):
        parser.error('positive updates and distinct seeds required')
    if (args.output/'design.json').exists():
        parser.error('output already contains a design')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    task=Task()
    graph=load_connectome().select(CORE)
    sources=['analysis/encoding_topology_probe.py','analysis/connectome_memory_probe.py','analysis/connectome_pilot.py',
             'src/thermotaxis/encoding_policies.py','src/thermotaxis/connectome_policy.py','src/thermotaxis/connectome.py']
    runs=[dict(architecture=arch,encoding=encoding,topology_seed=topo,seed=seed)
          for arch in args.architectures
          for encoding in args.encodings
          for topo in (args.topology_seeds if arch=='rewired' else [0])
          for seed in args.seeds]
    design=dict(status='exploratory_supervised_encoding_by_topology',task=asdict(task),runs=runs,
        updates=args.updates,training_episodes=192,validation_episodes=64,batch=32,
        optimizer=dict(name='Adam',lr=.003,clip_gradient_norm=1.),
        input=dict(level='signed local noisy temperature deviation / 1.5K',
            auxiliary='(fast signed temperature - slow signed temperature)/0.05K, or zero in raw condition',
            fast_tau_s=.5,slow_tau_s=5.,dt_s=1.,filters_computed_in_both_conditions=True),
        controlled_comparison='Within each architecture/topology/seed, only auxiliary channel masking changes',
        teacher='previous known absolute-error two-timescale controller; states never input',
        data_seeds=dict(train=82001,validation=82002,behavior_validation=90001,exploratory_holdouts=94001),
        initialization='zero actor head: all models begin with rate .255/s',
        checkpoint_rule='final fixed-budget model; retain all curves, no best checkpoint selection',
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        data=graph.provenance,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        versions=dict(torch=torch.__version__,numpy=np.__version__),
        limitations=['not reinforcement learning or RL sample-efficiency evidence',
            'same signed filter form across models; not calibrated AFD physiology',
            'chemical topology ablation keeps real gap/input/output assumptions',
            'generic GRU has different dynamics and readout; GRU2 is only near parameter matched',
            'all rewired topologies crossed with training seeds, not selected for success',
            '64 behavior episodes are not 64 independent trained models',
            'holdouts are exploratory and retired after inspection'])
    (args.output/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    for p in sources:
        target=args.output/'source'/p
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
    train_x,train_y=teacher_data(replace(task,batch=192),82001)
    val_x,val_y=teacher_data(replace(task,batch=64),82002)
    report=dict(design=design,constant_predictor_mse=float(((val_y-train_y.mean())/.4).square().mean()),
        teacher_behavior=trim(evaluate('memory',replace(task,batch=64),90001)),runs=[])
    for spec in runs:
        torch.manual_seed(spec['seed'])
        model=make_encoded_policy(graph,**spec_model_arguments(spec))
        optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        rng=np.random.default_rng(spec['seed'])
        curves=[]
        start=time.monotonic()
        for update in range(args.updates+1):
            if update%128==0 or update==args.updates:
                with torch.no_grad():
                    mse=float(((predict(model,val_x)-val_y)/.4).square().mean())
                curves.append(dict(update=update,validation_mse=mse))
                print(f'{spec} update={update} mse={mse:.6f}',flush=True)
            if update==args.updates:
                break
            index=rng.choice(192,32,replace=False)
            prediction=predict(model,train_x[:,index])
            loss=((prediction-train_y[:,index])/.4).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        metrics=evaluate(model,replace(task,batch=64),90001)
        holdouts={label:trim(evaluate(model,replace(task,batch=64,**change),94001))
                  for label,change in {'new_random_draws':{},'more_noise':{'noise_K':.1},
                                       'shallower_gradient':{'gradient_K_m':70.}}.items()}
        run=dict(spec,parameters=sum(p.numel() for p in model.parameters()),rewiring=model.rewiring,
                 curves=curves,behavior_validation=metrics,exploratory_holdouts=holdouts,
                 history_response=response_probe(model),wall_time_s=time.monotonic()-start)
        report['runs'].append(run)
        stem=f'{spec["architecture"]}-{spec["encoding"]}-topo{spec["topology_seed"]}-seed{spec["seed"]}'
        torch.save(model.state_dict(),args.output/f'{stem}.pt')
        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'FINISHED {stem} residence={metrics["residence_fraction"]:.4f}',flush=True)


def spec_model_arguments(spec):
    return {k:spec[k] for k in ('architecture','encoding','topology_seed')}


if __name__=='__main__':
    main()
