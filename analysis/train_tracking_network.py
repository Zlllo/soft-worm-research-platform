"""Ordinary finite-history neural controllers: imitation then bounded aggregation.

This is supervised learning with expert queries, NOT reinforcement learning.
Episode-level data splits, fixed budgets and causal input permissions are logged.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from thermotaxis.isothermal_scene import (TrackingConfig,RegressionControlConfig,
    PhaseRegressionController,BlindTrackingController,run_tracking)
from thermotaxis.tracking_learning import (TrackingNetworkConfig,TrackingNetwork,
    NeuralTrackingController,ImitationRecorder)


VARIANTS={'gru_raw':('gru','raw'),'gru_demeaned':('gru','demeaned'),'window_mlp':('mlp','raw')}


def expert_for(config):
    return PhaseRegressionController(RegressionControlConfig(preferred_K=config.preferred_K,
        comfort_scale_K=config.comfort_half_width_K,speed_m_s=config.speed_m_s,
        head_forward_offset_m=config.length_m/2,head_lateral_amplitude_m=config.head_lateral_amplitude_m,
        window_s=2/config.head_frequency_Hz,max_turn_rate_rad_s=config.max_turn_rate_rad_s))


def collect(base,network_config,seeds,model=None,beta=1.,mixture_seed=0):
    xs=[];ys=[];times=[];fields=[];episodes=[]
    for field_id,field in enumerate(('linear','radial')):
        config=replace(base,field=field)
        for seed in seeds:
            recorder=ImitationRecorder(expert_for(config),network_config,model,beta,
                seed=mixture_seed+seed+field_id*100000)
            result=run_tracking(config,recorder,seed)
            x,y,t=recorder.dataset()
            xs.append(x);ys.append(y);times.append(t);fields.append(np.full(len(y),field_id))
            episodes.append(dict(field=field,seed=seed,labels=len(y),
                metrics=result['metrics'],expert_actions=recorder.expert_actions,beta=beta))
    return dict(x=np.concatenate(xs),y=np.concatenate(ys),times=np.concatenate(times),
                fields=np.concatenate(fields),episodes=episodes)


def dataset_digest(data):
    return {k:dict(shape=list(data[k].shape),sha256=hashlib.sha256(data[k].tobytes()).hexdigest())
            for k in ('x','y','times','fields')}


def validation(model,data):
    predictions=[]
    with torch.no_grad():
        for offset in range(0,len(data['y']),512):
            x=torch.from_numpy(data['x'][offset:offset+512])
            predictions.append(model(x).numpy())
    prediction=np.concatenate(predictions)
    scale=model.config.max_turn_rate_rad_s
    result={}
    groups={'all':np.ones(len(prediction),dtype=bool),'linear':data['fields']==0,'radial':data['fields']==1,
            'startup':data['times']<4.,'full_window':data['times']>=4.,
            'large_command':np.abs(data['y'])>.02}
    for label,mask in groups.items():
        if mask.any():
            target=data['y'][mask]
            result[label]=dict(n=int(mask.sum()),normalized_mse=float(np.mean(((prediction[mask]-target)/scale)**2)),
                zero_command_mse=float(np.mean((target/scale)**2)),
                target_rms_rad_s=float(np.sqrt(np.mean(target**2))),
                prediction_rms_rad_s=float(np.sqrt(np.mean(prediction[mask]**2))))
    return result


def fit(model,optimizer,data,validation_data,updates,rng,stage):
    curves=[]
    model.train()
    for update in range(updates+1):
        if update%128==0 or update==updates:
            metrics=validation(model,validation_data)
            curves.append(dict(stage=stage,update=update,metrics=metrics))
            print(stage,update,'val_mse',round(metrics['all']['normalized_mse'],6),
                  'zero',round(metrics['all']['zero_command_mse'],6),flush=True)
        if update==updates:
            break
        index=rng.integers(0,len(data['y']),size=64)
        x=torch.from_numpy(data['x'][index]);y=torch.from_numpy(data['y'][index])
        prediction=model(x)
        loss=((prediction-y)/model.config.max_turn_rate_rad_s).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite neural tracking loss')
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    model.eval()
    return curves


def evaluate(model,base,seeds):
    rows=[]
    for field in ('linear','radial'):
        for seed in seeds:
            controller=(expert_for(replace(base,field=field)) if model=='expert' else
                        BlindTrackingController() if model=='blind' else NeuralTrackingController(model))
            row=run_tracking(replace(base,field=field),controller,seed)
            rows.append(dict(field=field,seed=seed,metrics=row['metrics'],initial_state=row['initial_state']))
    return rows


def tracking_mean(rows):
    return float(np.mean([x['metrics']['tracking_fraction'] for x in rows]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--updates',type=int,default=512)
    parser.add_argument('--seeds',nargs='+',type=int,default=[7,17,27])
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    parser.add_argument('--output',type=Path,default=Path('results/neural-tracking-2026-10-24'))
    args=parser.parse_args()
    if args.updates<1 or len(set(args.seeds))!=len(args.seeds):
        parser.error('positive updates and distinct training seeds required')
    if (args.output/'design.json').exists():
        parser.error('choose an unused output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    base=TrackingConfig()
    training_scene=replace(base,duration_s=60.,initial_error_fraction=.8,initial_heading_jitter_rad=.7)
    sources=['analysis/train_tracking_network.py','src/thermotaxis/tracking_learning.py','src/thermotaxis/isothermal_scene.py']
    design=dict(status='supervised_neural_tracking_with_conditional_data_aggregation_not_RL',
        scene=asdict(base),training_scene=asdict(training_scene),network_defaults=asdict(TrackingNetworkConfig()),
        variants=args.variants,training_seeds=args.seeds,updates=args.updates,
        optimizer=dict(name='Adam',lr=.003,batch=64,clip_grad_norm=1.),
        episode_splits=dict(train=list(range(1000,1064)),offline_validation=list(range(3000,3008)),
            closed_loop_validation=[4001,4002],final_exploratory=[5001,5002,5003,5004],transfer=[6001,6002]),
        input='right-aligned 41-sample causal local-temperature/phase window; availability mask; GRU h0=0 for every decision',
        label='same-tick expert requested angular rate, not realized angular velocity or future target',
        aggregation=dict(trigger='stage1 validation mean tracking < .8',
            rounds=3,expert_mixture_betas=[.5,.25,0.],updates_per_round=128,
            episodes_per_field_per_round=4,seed_rule='7000+100*round+episode index',
            label_rule='expert observes exactly the learner-visited history; no geometry truth input',
            checkpoint_rule='fixed final stage, no best-checkpoint selection'),
        transfer_conditions=dict(more_noise={'noise_std_K':.01},steeper_gradient={'gradient_K_m':70.},
                                 slower_response={'turn_response_tau_s':1.2}),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        versions=dict(torch=torch.__version__,numpy=np.__version__),
        limitations=['expert imitation and extra labels, not RL sample efficiency',
            'history horizon informed by expert window', 'aggregation uses model-specific data and expert-query budgets',
            'training starts broader in position/heading than basic maintenance evaluation',
            'GRU16 and MLP parameter counts differ; no biological topology compared',
            'validation controls aggregation trigger; final exploratory seeds inspected only after fitting',
            'all final cases remain exploratory, not a definitive generalization test'])
    (args.output/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    for p in sources:
        target=args.output/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    print('Generating expert training episodes',flush=True)
    default=TrackingNetworkConfig()
    training=collect(training_scene,default,design['episode_splits']['train'])
    offline_val=collect(training_scene,default,design['episode_splits']['offline_validation'])
    report=dict(design=design,training_data=dataset_digest(training),validation_data=dataset_digest(offline_val),
        training_episodes=training['episodes'],validation_episodes=offline_val['episodes'],runs=[],
        final_baselines={name:evaluate(name,base,design['episode_splits']['final_exploratory']) for name in ('expert','blind')})
    for variant in args.variants:
        for seed in args.seeds:
            torch.manual_seed(seed)
            arch,encoding=VARIANTS[variant]
            model=TrackingNetwork(replace(default,architecture=arch,encoding=encoding))
            optimizer=torch.optim.Adam(model.parameters(),lr=.003)
            rng=np.random.default_rng(seed)
            started=time.monotonic()
            run=dict(variant=variant,seed=seed,parameter_count=sum(p.numel() for p in model.parameters()),
                curves=fit(model,optimizer,training,offline_val,args.updates,rng,'offline_imitation'))
            validation_rows=evaluate(model,base,design['episode_splits']['closed_loop_validation'])
            run['stage1_closed_loop_validation']=validation_rows
            run['stage1_validation_tracking']=tracking_mean(validation_rows)
            print(variant,seed,'stage1 tracking',run['stage1_validation_tracking'],flush=True)
            torch.save(model.state_dict(),args.output/f'{variant}-{seed}-stage1.pt')
            run['aggregation_rounds']=[]
            data=training
            if run['stage1_validation_tracking']<.8:
                for round_index,beta in enumerate(design['aggregation']['expert_mixture_betas']):
                    model.eval()
                    new=collect(training_scene,model.config,
                        range(7000+100*round_index,7004+100*round_index),model,beta,
                        mixture_seed=seed*100000+round_index*1000)
                    data={k:np.concatenate((data[k],new[k])) for k in ('x','y','times','fields')}
                    new_info=dict(round=round_index,beta=beta,episodes=new['episodes'],
                                  dataset=dataset_digest(new),total_available_labels=len(data['y']))
                    run['curves'].extend(fit(model,optimizer,data,offline_val,128,rng,f'aggregation_{round_index}'))
                    run['aggregation_rounds'].append(new_info)
            run['total_unique_expert_labels']=len(data['y'])
            run['final_offline_validation']=validation(model,offline_val)
            run['final_behavior']=evaluate(model,base,design['episode_splits']['final_exploratory'])
            run['transfer']={name:evaluate(model,replace(base,**changes),design['episode_splits']['transfer'])
                             for name,changes in design['transfer_conditions'].items()}
            run['wall_time_s']=time.monotonic()-started
            torch.save(model.state_dict(),args.output/f'{variant}-{seed}-final.pt')
            report['runs'].append(run)
            (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
            print('FINISHED',variant,seed,'final tracking',tracking_mean(run['final_behavior']),
                  'labels',run['total_unique_expert_labels'],flush=True)
    report['source_hashes_unchanged']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in design['source_sha256'].items()}
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE',args.output,flush=True)


if __name__=='__main__':
    main()
