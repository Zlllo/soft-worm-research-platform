"""A fixed physical readout of temperature direction from ideal ego-motion.

No body simulation, neuronal simulation, neural-network fitting or parameter
selection is performed. Run from the project/extracted archive root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np


ROOT=Path('results/physical-thermal-readout-2026-11-13')
SOURCE=Path('results/calcium-body-transfer-2026-11-13/cases')
CASES=['wave_r1_f0.5', 'wave_r1_f2', 'heavy_drag_r1_f2']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def scalar(item):
        if isinstance(item,np.generic):
            return item.item()
        raise TypeError(f'Unsupported JSON value {type(item).__name__}')
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,default=scalar)+'\n')


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    protocol=dict(date='2026-11-13', stage='physical-thermal-readout',
        parent_review='Approved by root before formal metrics; 12 conditions, four paired seeds, zero new simulations/training.',
        cases=CASES, motion_modes=['matched', 'delay_0.6s'], noise_mK=[.1, 1.],
        noise_seeds=[3201,3202,3203,3204], field_angles_rad=(np.pi/8+np.arange(8)*np.pi/4).tolist(),
        gradient_K_per_m=40., unknown_background_K=[-.04,.04], sensor_base_K=293.15,
        sensor_dt_s=.1, history_s=4., endpoint_s=[12.,20.], endpoint_step_s=.5,
        ideal_motion='Relative head displacements, expressed in fixed axes of current end-of-window body. No absolute world coordinates enter OLS.',
        axes='Forward axis from tail to head at window end; normal is its counterclockwise 90-degree rotation. Ego-motion is ideal, without estimation noise.',
        delayed_motion='Complete past head-displacement history t-0.6 s, translated relative to its own delayed endpoint, rotated into the CURRENT endpoint axes. Temperature stays current.',
        estimator='Unregularized centered OLS, no hyperparameter fitting: temperature = unknown intercept + X_ego @ g_ego.',
        source_roles='Actual positions and true gradients used only by sensor/label generator; estimator accepts temperature vector and relative ego-motion matrix only.',
        weak_target_abs_fraction=.1, rank_relative_tolerance=1e-12,
        minimum_singular_value_m=1e-12, minimum_conditional_normal_energy_m2=1e-24,
        unavailable='Report rank/energy failures and coverage; do not assign a favorable class. Correct/all-eligible counts unavailable as unresolved, not correct.',
        checks=dict(noiseless_relative_gradient_error=1e-8, rigid_invariance_relative_error=1e-8,
                    zero_delay_max_error=1e-14, analytic_covariance_relative_error=1e-10,
                    finite_and_causal=True),
        grouping='No fitted parameters; all three whole trajectories and all field directions fixed before comparison. Overlapping windows and four noise seeds are descriptive paired evaluations, not independent biological repetitions.',
        theory='Gaussian sign-accuracy prediction applies only to matched input, based on known noise and true gradient magnitude for validation; true gradient is never an OLS input.',
        budget=dict(input_conditions=12, paired_noise_seeds=4, new_body_simulations=0,
                    new_neuron_simulations=0, training=0, parameter_search=0))
    dump(root/'protocol.json',protocol)
    provenance={}
    for name in CASES:
        source=SOURCE/name/'trace.npz'
        target=root/'frozen_inputs'/name/'trace.npz'
        target.parent.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(source,target)
        provenance[name]=dict(original_path=str(source),sha256=digest(source),
                              local_path=str(target.relative_to(root)))
    dump(root/'inputs.json',provenance)
    rng=np.random.default_rng(20261113)
    base_noise=np.stack([np.random.default_rng(seed).normal(size=(8,201)) for seed in protocol['noise_seeds']])
    backgrounds=rng.uniform(-.04,.04,size=(4,8))
    np.savez_compressed(root/'paired_sensor_inputs.npz', base_noise=base_noise, backgrounds_K=backgrounds)
    locks={str(Path(__file__).resolve().relative_to(Path.cwd())):digest(__file__)}
    dump(root/'source-lock.json',locks)
    dump(root/'protocol-lock.json',dict(protocol_sha256=digest(root/'protocol.json'),
             inputs_sha256=digest(root/'inputs.json'),source_lock_sha256=digest(root/'source-lock.json'),
             paired_sensor_inputs_sha256=digest(root/'paired_sensor_inputs.npz')))
    print('Prepared twelve fixed conditions; no formal metrics evaluated.',flush=True)


def verify(root):
    lock=json.loads((root/'protocol-lock.json').read_text())
    for name,key in [('protocol.json','protocol_sha256'),('inputs.json','inputs_sha256'),
                     ('source-lock.json','source_lock_sha256'),('paired_sensor_inputs.npz','paired_sensor_inputs_sha256')]:
        assert digest(root/name)==lock[key],name
    for path,expected in json.loads((root/'source-lock.json').read_text()).items():
        assert digest(path)==expected,path
    for value in json.loads((root/'inputs.json').read_text()).values():
        assert digest(root/value['local_path'])==value['sha256']


def body_axes(pose):
    forward=pose[0,:2]-pose[-1,:2]
    forward=forward/np.linalg.norm(forward)
    normal=np.array([-forward[1],forward[0]])
    return np.column_stack([forward,normal])


def ego_history(pose, indices, endpoint, delay=0):
    frame=body_axes(pose[endpoint])
    past=indices-delay
    assert np.all(past>=0) and np.all(past<=endpoint)
    return (pose[past,0,:2]-pose[endpoint-delay,0,:2])@frame


def estimate(temperature_K, displacement_m, protocol):
    """Input boundary: no field gradient, world pose or labels are accepted."""
    x=np.asarray(displacement_m)-np.asarray(displacement_m).mean(axis=0)
    y=np.asarray(temperature_K)-np.asarray(temperature_K).mean()
    singular=np.linalg.svd(x,compute_uv=False)
    tol=protocol['rank_relative_tolerance']*singular[0] if singular[0]>0 else 0
    rank=int(np.count_nonzero(singular>tol))
    along_energy=float(x[:,0]@x[:,0])
    residual=x[:,1]-x[:,0]*float(x[:,0]@x[:,1])/along_energy if along_energy>0 else x[:,1]
    normal_energy=float(residual@residual)
    available=bool(rank==2 and singular[-1]>protocol['minimum_singular_value_m']
                   and normal_energy>protocol['minimum_conditional_normal_energy_m2'])
    gradient=np.linalg.lstsq(x,y,rcond=protocol['rank_relative_tolerance'])[0] if available else None
    condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None
    return dict(gradient=gradient,available=available,rank=rank,condition=condition,
                normal_energy_m2=normal_energy,normal_excursion_um=math.sqrt(max(normal_energy,0)/len(x))*1e6,
                smallest_singular_m=float(singular[-1]))


def numerical_controls(protocol):
    # Known plane and rich, asymmetric, synthetic self-motion; no source data.
    t=np.linspace(0,4,41)
    x=np.column_stack([3e-6*t+2e-6*np.sin(t*3),5e-6*np.cos(t*2)+1e-6*np.sin(t)])
    truth=np.array([17.,-29.])
    y=293.17+x@truth
    base=estimate(y,x,protocol)
    error=float(np.linalg.norm(base['gradient']-truth)/np.linalg.norm(truth))
    shifted=estimate(y+17.,x+np.array([.003,-.002]),protocol)
    shift_error=float(np.linalg.norm(base['gradient']-shifted['gradient'])/np.linalg.norm(truth))
    # A coordinate rotation of world positions AND sensor field leaves ego inputs invariant.
    angle=.71
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    rotated_world=x@rotation.T
    rotated_field=rotation@truth
    world_temperature=293.17+rotated_world@rotated_field
    rotated_ego=rotated_world@rotation
    rotated=estimate(world_temperature,rotated_ego,protocol)
    rotation_error=float(np.linalg.norm(base['gradient']-rotated['gradient'])/np.linalg.norm(truth))
    rank_one=estimate(y,np.column_stack([t*1e-6,np.zeros_like(t)]),protocol)
    tiny=estimate(y,x*1e-10,protocol)
    xc=x-x.mean(0)
    covariance=np.linalg.inv(xc.T@xc)
    covariance_error=abs(covariance[1,1]-1/base['normal_energy_m2'])/covariance[1,1]
    checks=dict(noiseless_plane_recovered=error<=protocol['checks']['noiseless_relative_gradient_error'],
                unknown_background_and_translation_invariant=shift_error<=protocol['checks']['rigid_invariance_relative_error'],
                world_rotation_invariant=rotation_error<=protocol['checks']['rigid_invariance_relative_error'],
                rank_one_reported_unavailable=not rank_one['available'],
                tiny_motion_reported_unavailable=not tiny['available'],
                conditional_energy_matches_inverse_covariance=covariance_error<=protocol['checks']['analytic_covariance_relative_error'])
    return dict(checks=checks,noiseless_relative_error=error,translation_relative_error=shift_error,
                rotation_relative_error=rotation_error,covariance_relative_error=float(covariance_error))


def balanced(values, key):
    per_class=[]
    for sign in [-1,1]:
        group=[r for r in values if r['target_sign']==sign and r['eligible']]
        if key=='conditional':
            group=[r for r in group if r['available']]
        per_class.append(sum(bool(r['correct']) for r in group)/len(group) if group else None)
    return float(np.mean(per_class)) if all(x is not None for x in per_class) else None


def analyze(root, output):
    verify(root)
    p=json.loads((root/'protocol.json').read_text())
    controls=numerical_controls(p)
    assert all(controls['checks'].values()),controls
    paired=np.load(root/'paired_sensor_inputs.npz')
    rows=[]
    saved_temperature=[]
    saved_ego=[]
    saved_sensor_time=[]
    saved_reference_time=[]
    saved_gradient=[]
    geometry=[]
    null_checks=[]
    field_angles=np.array(p['field_angles_rad'])
    fields=p['gradient_K_per_m']*np.column_stack([np.cos(field_angles),np.sin(field_angles)])
    end_times=np.arange(12.,20.+.01,.5)
    for name in p['cases']:
        z=np.load(root/'frozen_inputs'/name/'trace.npz')
        time=z['time_s'];pose=z['pose']
        assert np.allclose(time,np.arange(2001)*.01,atol=1e-12,rtol=0)
        assert np.isfinite(pose).all()
        sensor_indices=np.arange(0,2001,10)
        head=pose[sensor_indices,0,:2]
        for endpoint_s in end_times:
            endpoint=round(endpoint_s/.01)
            indices=np.arange(endpoint-400,endpoint+1,10)
            sensor_window=indices//10
            frame=body_axes(pose[endpoint])
            x_true=ego_history(pose,indices,endpoint)
            # Explicit zero-delay identity and causal-index checks on every window.
            null_checks.append(float(abs(x_true-ego_history(pose,indices,endpoint,delay=0)).max()))
            for mode in p['motion_modes']:
                delay=0 if mode=='matched' else 60
                x=ego_history(pose,indices,endpoint,delay=delay)
                geom=estimate(np.zeros(len(x)),x,p)
                geometry.append(dict(case=name,mode=mode,endpoint_s=float(endpoint_s),
                     available=geom['available'],rank=geom['rank'],condition=geom['condition'],
                     normal_energy_m2=geom['normal_energy_m2'],normal_excursion_um=geom['normal_excursion_um'],
                     smallest_singular_m=geom['smallest_singular_m'],max_reference_time_s=float(time[indices[-1]-delay])))
                for noise_mK in p['noise_mK']:
                    sigma=noise_mK*1e-3
                    for seed_i,seed in enumerate(p['noise_seeds']):
                        for field_i,g_world in enumerate(fields):
                            # Sensor generator alone sees world positions and true field.
                            all_temperature=p['sensor_base_K']+paired['backgrounds_K'][seed_i,field_i]+head@g_world
                            all_temperature=all_temperature+sigma*paired['base_noise'][seed_i,field_i]
                            temperature=all_temperature[sensor_window]
                            fit=estimate(temperature,x,p)
                            truth=frame.T@g_world   # label/theory boundary, after OLS
                            eligible=bool(abs(truth[1])>=p['weak_target_abs_fraction']*p['gradient_K_per_m'])
                            predicted=None if fit['gradient'] is None else float(fit['gradient'][1])
                            available=bool(fit['available'] and predicted!=0)
                            correct=bool(predicted*truth[1]>0) if available and eligible else None
                            sigma_g=sigma/math.sqrt(fit['normal_energy_m2']) if fit['normal_energy_m2']>0 else None
                            theory=.5*(1+math.erf(abs(truth[1])/sigma_g/math.sqrt(2))) if mode=='matched' and sigma_g else None
                            rows.append(dict(case=name,mode=mode,noise_mK=noise_mK,seed=seed,
                                field_index=field_i,endpoint_s=float(endpoint_s),target_sign=1 if truth[1]>0 else -1,
                                true_normal_gradient_K_per_m=float(truth[1]),estimated_normal_gradient_K_per_m=predicted,
                                eligible=eligible,available=available,correct=correct,
                                normal_gradient_error_K_per_m=predicted-float(truth[1]) if available else None,
                                predicted_sigma_normal_K_per_m=sigma_g,
                                matched_gaussian_sign_accuracy=theory))
                            saved_temperature.append(temperature.copy())
                            saved_ego.append(x.copy())
                            saved_sensor_time.append(time[indices].copy())
                            saved_reference_time.append(time[indices-delay].copy())
                            saved_gradient.append(fit['gradient'].copy() if available else np.full(2,np.nan))
    assert max(null_checks)<=p['checks']['zero_delay_max_error']
    conditions=[]
    for name in p['cases']:
        for mode in p['motion_modes']:
            geo=[g for g in geometry if g['case']==name and g['mode']==mode]
            for noise_mK in p['noise_mK']:
                selected=[r for r in rows if r['case']==name and r['mode']==mode and r['noise_mK']==noise_mK]
                eligible=[r for r in selected if r['eligible']]
                resolved=[r for r in eligible if r['available']]
                by_seed=[]
                for seed in p['noise_seeds']:
                    seed_rows=[r for r in eligible if r['seed']==seed]
                    by_seed.append(dict(seed=seed,balanced_accuracy=balanced(seed_rows,'conditional'),
                                        correct_fraction_all_eligible=balanced(seed_rows,'all'),
                                        coverage=sum(r['available'] for r in seed_rows)/len(seed_rows)))
                errors=[r['normal_gradient_error_K_per_m'] for r in resolved]
                predicted=[r['matched_gaussian_sign_accuracy'] for r in resolved if r['matched_gaussian_sign_accuracy'] is not None]
                conditions.append(dict(case=name,mode=mode,noise_mK=noise_mK,
                    balanced_accuracy=balanced(eligible,'conditional'),
                    correct_fraction_all_eligible=balanced(eligible,'all'),
                    coverage=len(resolved)/len(eligible) if eligible else 0,
                    observations=len(selected),eligible=len(eligible),weak_target=len(selected)-len(eligible),
                    unresolved=len(eligible)-len(resolved),
                    normal_gradient_RMSE_K_per_m=math.sqrt(float(np.mean(np.square(errors)))) if errors else None,
                    predicted_matched_accuracy=float(np.mean(predicted)) if predicted else None,
                    normal_excursion_um_median=float(np.median([g['normal_excursion_um'] for g in geo])),
                    condition_number_median=float(np.median([g['condition'] for g in geo if g['condition'] is not None]))
                    if any(g['condition'] is not None for g in geo) else None,
                    predicted_sigma_normal_K_per_m_median=float(np.median([r['predicted_sigma_normal_K_per_m'] for r in eligible
                                                                                 if r['predicted_sigma_normal_K_per_m'] is not None])),
                    noise_seed_results=by_seed))
    output.mkdir(parents=True,exist_ok=True)
    with (output/'window-results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'estimator-inputs.npz',temperature_K=np.array(saved_temperature),
                        ego_displacement_m=np.array(saved_ego),sensor_time_s=np.array(saved_sensor_time),
                        reference_time_s=np.array(saved_reference_time),estimated_gradient_K_per_m=np.array(saved_gradient),
                        available=np.array([r['available'] for r in rows]),
                        true_normal_gradient_K_per_m=np.array([r['true_normal_gradient_K_per_m'] for r in rows]))
    dump(output/'geometry.json',geometry)
    dump(output/'controls.json',controls)
    summary=dict(protocol_sha256=digest(root/'protocol.json'),conditions=conditions,
        checks=dict(all_six_numeric_controls_passed=all(controls['checks'].values()),
                    source_and_input_hashes=True,zero_delay_identity=True,all_reference_times_causal=all(g['max_reference_time_s']<=g['endpoint_s'] for g in geometry),
                    exactly_twelve_conditions=len(conditions)==12,exactly_four_paired_noise_seeds=len(p['noise_seeds'])==4,
                    no_new_simulation_or_training=True),
        total_window_evaluations=len(rows),physical_trajectories=3,windows_per_trajectory=17,
        sensor_noise_pairing='Same base noise for case/mode/noise-level comparisons; levels only rescale the common Gaussian stream.',
        scope='Fixed physical estimator with ideal ego-odometry, static linear temperature fields and float64 sensors. No biological neural implementation or closed-loop navigation claim.',
        estimator_input_boundary='Only a 41-value temperature vector and a 41x2 ideal ego-displacement history.',
        uncertainty='Four sensor-noise seeds and overlapping windows are descriptive paired evaluations; no biological significance tests or independent-window confidence intervals.')
    dump(output/'summary.json',summary)
    verify(root)
    print(json.dumps(dict(checks=summary['checks'],conditions=conditions),indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','analyze','verify'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=Path('docs/reports/assets/physical-thermal-readout-2026-11-13'))
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.root)
    elif args.action=='verify':verify(args.root)
    else:analyze(args.root,args.output)
