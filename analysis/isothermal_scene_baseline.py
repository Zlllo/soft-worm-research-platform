"""Predeclared paired checks of the 2D active-sampling scene; no neural training."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from thermotaxis.isothermal_scene import (TrackingConfig,RegressionControlConfig,
    BlindTrackingController,PhaseRegressionController,run_tracking)


CONDITIONS=('blind','head_phase','center_phase','head_quarter_offset','head_scrambled')


def run_case(config,condition,seed,record=False):
    control_config=RegressionControlConfig(preferred_K=config.preferred_K,
        comfort_scale_K=config.comfort_half_width_K,speed_m_s=config.speed_m_s,
        head_forward_offset_m=0. if condition=='center_phase' else config.length_m/2,
        head_lateral_amplitude_m=config.head_lateral_amplitude_m,
        window_s=2/config.head_frequency_Hz,max_turn_rate_rad_s=config.max_turn_rate_rad_s)
    control=BlindTrackingController() if condition=='blind' else PhaseRegressionController(control_config)
    phase={'head_quarter_offset':'quarter_offset','head_scrambled':'scrambled'}.get(condition,'correct')
    result=run_tracking(config,control,seed,sensor_location='center' if condition=='center_phase' else 'head',
                        phase_mode=phase,record_trajectory=record)
    result['condition']=condition
    result['controller_config']=asdict(control_config) if condition!='blind' else None
    return result


def aggregate(rows):
    groups={}
    for row in rows:
        key=(row['config']['field'],row['config']['noise_std_K'],row['config']['turn_response_tau_s'],row['condition'])
        groups.setdefault(key,[]).append(row)
    summary=[]
    for key,group in groups.items():
        metrics={}
        for name in group[0]['metrics']:
            values=np.array([r['metrics'][name] for r in group])
            metrics[name]=dict(mean=float(values.mean()),minimum=float(values.min()),maximum=float(values.max()))
        summary.append(dict(field=key[0],noise_K=key[1],response_tau_s=key[2],condition=key[3],n=len(group),metrics=metrics))
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(101,113)))
    parser.add_argument('--output',type=Path,default=Path('results/isothermal-scene-2026-10-24'))
    args=parser.parse_args()
    if not args.seeds or len(set(args.seeds))!=len(args.seeds):
        parser.error('distinct seeds required')
    if (args.output/'design.json').exists():
        parser.error('choose an unused output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    base=TrackingConfig()
    specs=[dict(field=field,noise_std_K=noise,turn_response_tau_s=.3,condition=condition,seed=seed)
           for field in ('linear','radial') for noise in (0.,.002,.01)
           for condition in CONDITIONS for seed in args.seeds]
    response_specs=[dict(field=field,noise_std_K=.002,turn_response_tau_s=tau,condition='head_phase',seed=seed)
           for field in ('linear','radial') for tau in (0.,1.2,3.) for seed in args.seeds]
    convergence_specs=[dict(field=field,seed=seed,physics_dt_s=dt)
           for field in ('linear','radial') for seed in args.seeds[:2] for dt in (.02,.01,.005)]
    sources=['analysis/isothermal_scene_baseline.py','src/thermotaxis/isothermal_scene.py']
    design=dict(status='kinematic_scene_validation_no_neural_training',config=asdict(base),
        control=asdict(RegressionControlConfig()),paired_cases=specs,response_cases=response_specs,
        convergence_cases=convergence_specs,
        observation_contract='tracking_local_temperature_and_motor_phase_v2',
        measures=dict(primary='center-temperature comfort AND tangent heading error <=25deg while prescribed forward motion',
            secondary=['head temperature error','longest tracking duration','along-contour travel','absolute turn'],
            evaluation_only=['true position','heading','gradient orientation']),
        random_streams=['environment field orientation','initial state','temperature noise','reported phase intervention'],
        sampling_signal_K=base.head_lateral_amplitude_m*base.gradient_K_m,
        signal_noise_description='0.006K maximum lateral thermal amplitude; scan .002/.01K measurement noise',
        selection='all cases and fixed gains retained; no controller tuning after outcomes',
        construction_checks='Before preregistration, 10 noiseless seed100 smoke checks; no gains changed. Main seeds101..112.',
        limitations=['prescribed kinematic motion, not RFT/elastic/crawling physics',
            'starts near preference: maintenance task, not target seeking',
            'phase-regression controller has explicit task/geometry priors',
            'center sensor lacks the assumed head-sweep temperature modulation',
            'fixed phase and scramble are frozen-controller interventions, not retrained learning comparisons',
            'time trend is only an approximate longitudinal gradient in static slowly changing geometry',
            'no claim that head sampling or phase is necessary for every possible controller',
            'global temperature waves not evaluated in this batch'],
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),numpy=np.__version__)
    (args.output/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    for p in sources:
        target=args.output/'source'/p
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
    result=dict(design=design,runs=[],response_runs=[],convergence_runs=[])
    start=time.monotonic()
    for block,cases in [('runs',specs),('response_runs',response_specs)]:
        for index,spec in enumerate(cases):
            parameters={k:spec[k] for k in ('field','noise_std_K','turn_response_tau_s')}
            config=replace(base,**parameters)
            record=(block=='runs' and spec['noise_std_K']==.002 and spec['seed']==args.seeds[0])
            row=run_case(config,spec['condition'],spec['seed'],record)
            if record:
                stem=f'{config.field}-{spec["condition"]}'
                np.savez_compressed(args.output/f'{stem}.npz',
                    trajectory=row.pop('trajectory'),samples=row.pop('sensor_samples'),commands=row.pop('control_commands'))
                row['trace_file']=stem+'.npz'
            result[block].append(row)
            if (index+1)%len(args.seeds)==0:
                last=result[block][-len(args.seeds):]
                mean=np.mean([r['metrics']['tracking_fraction'] for r in last])
                print(block,index+1,'/',len(cases),spec['field'],spec['condition'],
                    'noise',spec['noise_std_K'],'tau',spec['turn_response_tau_s'],'tracking',round(float(mean),4),flush=True)
                (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for spec in convergence_specs:
        config=replace(base,field=spec['field'],physics_dt_s=spec['physics_dt_s'])
        result['convergence_runs'].append(run_case(config,'head_phase',spec['seed']))
    result['summary']=aggregate(result['runs']+result['response_runs'])
    result['wall_time_s']=time.monotonic()-start
    result['source_hashes_unchanged']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in design['source_sha256'].items()}
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',args.output,'seconds',round(result['wall_time_s'],1),flush=True)


if __name__=='__main__':
    main()
