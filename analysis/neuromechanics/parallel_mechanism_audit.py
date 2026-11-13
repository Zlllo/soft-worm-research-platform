"""Parent-process audit, using independent geometry and linear algebra.

No simulations or fits with selected hyperparameters are performed here.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


STAGE=Path('results/parallel-mechanism-stage-2026-11-13')
ASSETS=Path('docs/reports/assets')
APPROVED={
    'network-output-replay-2026-11-13':'8ef90f458470129052867c60b8887132d0b63b18e33b577e96b86b6ae2116dab',
    'passive-body-stiffness-2026-11-13':'bcdfe21f7598d09623e63eee443087bedd0132c274032cb248fdc963ec373326',
    'physical-thermal-readout-2026-11-13':'a41d4c77c594f6c47a8b72a30ace09e9bcfe128d8e0c858b20d95b369f46dc29',
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):return json.loads(Path(path).read_text())


def independent_curvature(pose):
    """Turning angle from adjacent tangent cross/dot, not angle unwrapping."""
    tangent=np.diff(pose[..., :2],axis=-2)
    left,right=tangent[..., :-1,:],tangent[...,1:,:]
    cross=left[...,0]*right[...,1]-left[...,1]*right[...,0]
    dot=np.sum(left*right,axis=-1)
    length=(np.linalg.norm(left,axis=-1)+np.linalg.norm(right,axis=-1))/2
    return np.arctan2(cross,dot)/length*1e-3


def independent_speed(z,start,end):
    selected=(z['time_s']>=start-1e-9)&(z['time_s']<=end+1e-9)
    t=z['time_s'][selected]
    v=z['velocity'][selected,:,:2].mean(1)
    heading=z['pose'][selected,0,:2]-z['pose'][selected,-1,:2]
    heading/=np.linalg.norm(heading,axis=1,keepdims=True)
    signed=np.sum(v*heading,axis=1)
    return float(np.trapezoid(signed,t)/(end-start)*1e6)


def compare(value,reference,*,atol=1e-9,rtol=1e-8):
    assert abs(value-reference)<=atol+rtol*abs(reference),(value,reference)
    return abs(value-reference)


def audit_network_replay():
    name='network-output-replay-2026-11-13';root=Path('results')/name
    summary=load(ASSETS/name/'summary.json')
    assert sha(root/'protocol.json')==APPROVED[name]==summary['protocol_sha256']
    errors=[]
    for record in summary['cases']:
        folder=root/'cases'/record['name']
        assert sha(folder/'trace.npz')==record['trace_sha256']
        z=np.load(folder/'trace.npz');scale=record['spec']['time_scale']
        t=z['time_s'];mask=(t>=4*scale-1e-9)&(t<=8*scale+1e-9)
        k=independent_curvature(z['pose'][mask])[:,4:-4]
        mu=np.trapezoid(k,t[mask],axis=0)/(4*scale)
        var=np.trapezoid((k-mu)**2,t[mask],axis=0)/(4*scale)
        speed=independent_speed(z,4*scale,8*scale)
        dynamic=float(np.sqrt(var.mean()))
        static=float(np.sqrt(np.mean(mu*mu)))
        errors.append(dict(name=record['name'],speed_error_um_s=compare(speed,record['late_signed_speed_um_s']),
                           dynamic_curvature_error_per_mm=compare(dynamic,record['dynamic_curvature_rms_per_mm']),
                           static_curvature_error_per_mm=compare(static,record['static_curvature_rms_per_mm'])))
    assert len(errors)==10 and len(list((root/'cases').glob('*/record.json')))==10
    return dict(protocol_sha256=summary['protocol_sha256'],new_body_runs=10,independent_metrics=errors,
                branch_checks=summary['checks'],failed_branch_checks=[k for k,v in summary['checks'].items() if not v])


def audit_stiffness():
    name='passive-body-stiffness-2026-11-13';root=Path('results')/name
    summary=load(ASSETS/name/'summary.json')
    assert sha(root/'protocol.json')==APPROVED[name]==summary['protocol_sha256']
    errors=[]
    for record in summary['cases']:
        folder=root/('reused' if record['scale']==1 else 'cases')/record['name']
        assert sha(folder/'trace.npz')==record['trace_sha256']
        z=np.load(folder/'trace.npz');t=z['time_s'];mask=(t>=8-1e-9)&(t<20-1e-9)
        k=independent_curvature(z['pose'][mask])[:,4:-4]
        # Integer number of cycles: Fourier projection independently replaces LS.
        coefficient=2*np.mean(k*np.exp(-2j*np.pi*record['frequency_Hz']*t[mask,None]),axis=0)
        amplitude=float(np.sqrt(np.mean(abs(coefficient)**2)))
        speed=independent_speed(z,8,20)
        errors.append(dict(name=record['name'],speed_error_um_s=compare(speed,record['signed_speed_um_s']),
                           curvature_amplitude_error_per_mm=compare(amplitude,record['curvature_harmonic_rms_per_mm'])))
    assert len(errors)==11 and len(list((root/'cases').glob('*/record.json')))==8
    return dict(protocol_sha256=summary['protocol_sha256'],new_body_runs=8,reused_cases=3,
                independent_metrics=errors,branch_checks=summary['checks'],
                failed_branch_checks=[k for k,v in summary['checks'].items() if not v])


def audit_thermal_readout():
    name='physical-thermal-readout-2026-11-13';root=Path('results')/name
    folder=ASSETS/name;summary=load(folder/'summary.json')
    assert sha(root/'protocol.json')==APPROVED[name]==summary['protocol_sha256']
    z=np.load(folder/'estimator-inputs.npz')
    x=z['ego_displacement_m'];x=x-x.mean(axis=1,keepdims=True)
    y=z['temperature_K'];y=y-y.mean(axis=1,keepdims=True)
    xx=np.sum(x[:,:,0]**2,axis=1);yy=np.sum(x[:,:,1]**2,axis=1)
    xy=np.sum(x[:,:,0]*x[:,:,1],axis=1)
    bx=np.sum(x[:,:,0]*y,axis=1);by=np.sum(x[:,:,1]*y,axis=1)
    determinant=xx*yy-xy*xy
    available=z['available']
    estimates=np.column_stack([(yy*bx-xy*by)/determinant,(xx*by-xy*bx)/determinant])
    expected=z['estimated_gradient_K_per_m']
    relative=abs(estimates[available]-expected[available])/np.maximum(1,abs(expected[available]))
    assert np.max(relative)<1e-7,float(np.max(relative))
    assert np.all(z['reference_time_s']<=z['sensor_time_s']+1e-12)
    with (folder/'window-results.csv').open() as handle:rows=list(csv.DictReader(handle))
    conditions=[]
    for condition in summary['conditions']:
        indices=[i for i,r in enumerate(rows) if r['case']==condition['case'] and r['mode']==condition['mode']
                 and float(r['noise_mK'])==condition['noise_mK'] and r['eligible']=='True' and available[i]]
        recalls=[]
        for sign in [-1,1]:
            idx=[i for i in indices if int(rows[i]['target_sign'])==sign]
            recalls.append(float(np.mean(estimates[idx,1]*z['true_normal_gradient_K_per_m'][idx]>0)))
        accuracy=float(np.mean(recalls))
        compare(accuracy,condition['balanced_accuracy'],atol=1e-12,rtol=0)
        conditions.append(dict(case=condition['case'],mode=condition['mode'],noise_mK=condition['noise_mK'],
                               recomputed_balanced_accuracy=accuracy))
    assert len(conditions)==12 and len(rows)==summary['total_window_evaluations']
    return dict(protocol_sha256=summary['protocol_sha256'],new_simulations=0,conditions=conditions,
                max_gradient_relative_error=float(np.max(relative)),reference_times_causal=True,
                independent_solver='Centered 2x2 normal-equation inverse; branch used SVD least squares.',
                branch_checks=summary['checks'],failed_branch_checks=[k for k,v in summary['checks'].items() if not v])


def main():
    baseline=load(STAGE/'baseline-lock.json')
    changed=[name for name,digest in baseline.items() if sha(name)!=digest]
    assert not changed,changed
    replay=audit_network_replay();stiffness=audit_stiffness();thermal=audit_thermal_readout()
    summary=dict(shared_baseline_files_unchanged=len(baseline),new_body_runs=18,new_neural_runs=0,
                 new_training_runs=0,branches=dict(network_replay=replay,passive_stiffness=stiffness,
                                                 physical_thermal_readout=thermal),
                 all_independent_recomputations_passed=True,
                 all_branch_checks_passed=not any(x['failed_branch_checks'] for x in [replay,stiffness,thermal]))
    (STAGE/'independent-audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='branches'},indent=2))


if __name__=='__main__':main()
