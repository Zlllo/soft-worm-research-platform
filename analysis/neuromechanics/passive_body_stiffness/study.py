"""Passive skeletal elasticity intervention under fixed prescribed calcium.

Only force coefficients of passive longitudinal and diagonal springs change.
All original global constants, active muscle parameters and damping stay fixed.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from thermotaxis.neuromechanics.anatomical_body import load_anatomical_body_wiring, PlanarMuscleProjection
from thermotaxis.neuromechanics.boyle_muscle_body import BoyleMuscleBody
from thermotaxis.neuromechanics.boyle_output import curvature


ROOT=Path('results/passive-body-stiffness-2026-11-13')
ASSETS=Path('docs/reports/assets/passive-body-stiffness-2026-11-13')
BASE=Path('results/full-network-body-interface-2026-11-05/build')
REFERENCE=Path('results/calcium-body-transfer-2026-11-13')
SUNDIALS=Path('results/neuromechanics-fastbody-2026-10-27/upstream/boyle2012/WormSim/Sundials')
SCRIPT=Path('analysis/neuromechanics/passive_body_stiffness/study.py')
CA_HALF=2.5e-7


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def compile_cpp(source, output, log):
    command=['g++','-O3','-Wno-deprecated','-I',str((SUNDIALS/'include').resolve()),
             str(source.resolve()),'-L',str((SUNDIALS/'lib').resolve()),'-lsundials_ida',
             '-lsundials_cvode','-lsundials_nvecserial','-lm','-o',str(output.resolve())]
    with log.open('w') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    return command


def build(folder, scale, dt):
    original=(BASE/'worm-muscle-worker.cc').read_text()
    meta=json.loads((BASE/'build.json').read_text())
    assert sha(BASE/'worker')==meta['binary_sha256']
    assert sha(BASE/'worm-muscle-worker.cc')==meta['generated_sha256']
    assert original.count('F_PE = k_PE*')==2
    assert original.count(')*k_DE - D_DE*')==2
    text=original.replace('F_PE = k_PE*','F_PE = PASSIVE_ELASTIC_SCALE*k_PE*')
    text=text.replace(')*k_DE - D_DE*',')*PASSIVE_ELASTIC_SCALE*k_DE - D_DE*')
    clock=f'#define DELTAT {dt:.17g}'
    text=text.replace('#define DELTAT 0.001',clock+f'\n#define PASSIVE_ELASTIC_SCALE {scale:.17g}')
    restored=text.replace(clock+f'\n#define PASSIVE_ELASTIC_SCALE {scale:.17g}','#define DELTAT 0.001')
    restored=restored.replace('F_PE = PASSIVE_ELASTIC_SCALE*k_PE*','F_PE = k_PE*')
    restored=restored.replace(')*PASSIVE_ELASTIC_SCALE*k_DE - D_DE*',')*k_DE - D_DE*')
    assert restored==original, 'Undeclared source change'
    folder.mkdir(parents=True,exist_ok=False)
    source=folder/'worm-muscle-worker.cc'
    source.write_text(text)
    (folder/'source.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True),text.splitlines(True),fromfile='frozen-base',tofile='passive-skeleton-variant')))
    command=compile_cpp(source,folder/'worker',folder/'build.log')
    probe_source=folder/'parameter-probe.cc'
    fields=['k_PE','D_PE','k_AE','D_AE','k_DE','D_DE','T_muscle','CL_agar','CN_agar','L_seg']
    fields += ['DELTAT','PASSIVE_ELASTIC_SCALE']
    stream='cout << setprecision(17) << "{";\n'
    for index,name in enumerate(fields):
        stream+=f'cout << "{"," if index else ""}\\\"{name}\\\":" << {name};\n'
    stream+='cout << "}" << endl;\n'
    probe_source.write_text('#define main original_worker_main\n#include "worm-muscle-worker.cc"\n'
                            '#undef main\nint main(){\n'+stream+'return 0;\n}\n')
    compile_cpp(probe_source,folder/'parameter-probe',folder/'probe-build.log')
    parameters=json.loads(subprocess.check_output([str((folder/'parameter-probe').resolve())],text=True))
    # Original NBAR macro is unparenthesized NSEG+1: 2.0*NBAR expands
    # to 2.0*48+1 (=97). Preserve the original model, do not repair it here.
    expected=dict(k_PE=.02,D_PE=.0005,k_AE=.4,D_AE=.05,k_DE=7.,D_DE=.07,
                  T_muscle=.1,CL_agar=3.2e-3/97,CN_agar=128e-3/97,L_seg=1e-3/48)
    for name,value in expected.items():
        assert abs(parameters[name]-value)<max(1e-18,1e-14*abs(value)), (name,parameters[name],value)
    assert parameters['PASSIVE_ELASTIC_SCALE']==scale and parameters['DELTAT']==dt
    write(folder/'parameters.json',dict(parameters=parameters,unchanged_expected=expected,
          effective_longitudinal_passive_stiffness_N_m=scale*parameters['k_PE'],
          effective_diagonal_passive_stiffness_N_m=scale*parameters['k_DE'],
          source_reversal_exact=True,
          original_macro_note='NBAR=NSEG+1 without parentheses; original 2.0*NBAR divisor evaluates to 97. Preserved unchanged.'))
    meta.update(dt_s=dt,passive_elastic_scale=scale,muscle_tau_s=.1,drag_scale=1.,
                generated_sha256=sha(source),binary_sha256=sha(folder/'worker'),
                build_command=command,base_generated_sha256=sha(BASE/'worm-muscle-worker.cc'),
                intervention='Only F_PE and elastic part of F_D multiplied; original globals and all damping/active terms unchanged')
    write(folder/'build.json',meta)


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    assert not (root/'protocol.json').exists() and not (root/'build').exists(), 'Refuse to replace a prepared study'
    for scale in [.5,2.]:
        for dt in [.001,.0005]:
            suffix='_fine' if dt==.0005 else ''
            build(root/'build'/f's{scale:g}{suffix}',scale,dt)
    specs=[]
    for scale in [.5,2.]:
        for frequency in [.5,1.,2.]:
            specs.append(dict(name=f's{scale:g}_f{frequency:g}',scale=scale,frequency=frequency,
                              dt_s=.001,variant=f's{scale:g}'))
    for scale in [.5,2.]:
        specs.append(dict(name=f's{scale:g}_f2_fine',scale=scale,frequency=2.,dt_s=.0005,
                          variant=f's{scale:g}_fine',paired_with=f's{scale:g}_f2'))
    reused=[]
    for frequency in [.5,1.,2.]:
        old=REFERENCE/'cases'/f'wave_r1_f{frequency:g}'
        record=json.loads((old/'record.json').read_text())
        assert sha(old/'trace.npz')==record['trace_sha256']
        spec=json.loads((old/'spec.json').read_text())
        assert spec==dict(name=f'wave_r1_f{frequency:g}',variant='normal',ratio=1.,frequency=frequency,pattern='travel')
        new=root/'reused'/f's1_f{frequency:g}'
        new.mkdir(parents=True,exist_ok=False)
        for name in ['trace.npz','record.json','spec.json']:
            shutil.copyfile(old/name,new/name)
        reused.append(dict(name=new.name,frequency=frequency,scale=1.,source=str(old),
                           trace_sha256=record['trace_sha256'],dt_s=.001))
    design=dict(date='2026-11-13',declared_before_execution=True,source='Boyle frozen direct-muscle worker',
                intervention='Both passive longitudinal and passive diagonal spring elastic force coefficients times lambda; all active/damping parameters fixed',
                scales=[.5,1.,2.],frequencies_Hz=[.5,1.,2.],duration_s=20.,metric_window_s=[8.,20.],
                calcium_peak_over_K=1.,calcium_half_mM=CA_HALF,drive_cap=.3,waves_per_body=1.5,
                muscle_tau_s=.1,original_directional_drag_ratio=40.,specs=specs,reused=reused,
                scope='Prescribed calcium, no neurons, no learning; model passive skeleton elasticity, not calibrated animal softness',
                run_budget=dict(new_main=6,new_half_step=2,total_new_body=8),
                no_adaptive_runs=True,
                numerical_checks=dict(speed_absolute_um_s=.2,speed_relative=.02,curvature_relative=.03,
                                      input_muscle_arrays_identical=True,finite_and_causal_required=True),
                prediction='No sign or optimum preselected. Report all scales and frequencies, including null or failed outcomes.')
    write(root/'protocol.json',design)
    paths=[SCRIPT,BASE/'build.json',BASE/'worm-muscle-worker.cc',BASE/'worker',
           Path('src/thermotaxis/neuromechanics/boyle_body.py'),
           Path('src/thermotaxis/neuromechanics/boyle_muscle_body.py'),
           Path('src/thermotaxis/neuromechanics/anatomical_body.py'),
           Path('src/thermotaxis/neuromechanics/muscle_mapping.py'),
           Path('src/thermotaxis/connectome.py'),Path('data/connectome/cook2019_herm.json'),
           Path('data/connectome/herm_neurons.json')]
    paths += [p for tree in ['build','reused'] for p in (root/tree).rglob('*') if p.is_file()]
    write(root/'source-lock.json',{str(p):sha(p) for p in sorted(paths)})
    write(root/'protocol-lock.json',dict(protocol_sha256=sha(root/'protocol.json'),source_lock_sha256=sha(root/'source-lock.json')))
    print('Prepared 6 main + 2 half-step runs, 3 reused controls',flush=True)


def verify(root):
    lock=json.loads((root/'protocol-lock.json').read_text())
    assert sha(root/'protocol.json')==lock['protocol_sha256']
    assert sha(root/'source-lock.json')==lock['source_lock_sha256']
    for name,digest in json.loads((root/'source-lock.json').read_text()).items():
        assert sha(name)==digest,name


def run_case(root,spec):
    folder=root/'cases'/spec['name'];folder.mkdir(parents=True,exist_ok=False)
    write(folder/'spec.json',spec)
    names=load_anatomical_body_wiring().muscle_names
    projection=PlanarMuscleProjection.rank_bins(names)
    phase=np.array([(int(n[3:])-.5)/24 for n in names])*2*np.pi*1.5
    side=np.array([1 if n[1]=='D' else -1 for n in names])
    rows=[];moments=np.zeros((4,95));started=time.monotonic()
    with BoyleMuscleBody(root/'build'/spec['variant']/'worker') as body:
        body.reset();dt=body.dt_s;count=round(20/dt);save_every=round(.01/dt)
        for tick in range(count+1):
            t=tick*dt
            ca=CA_HALF*(1+side*np.sin(2*np.pi*spec['frequency']*t-phase))/2
            activation=.3*ca/(ca+CA_HALF);command=projection.to_segments(activation)
            if tick%save_every==0:
                s=body.state
                rows.append(dict(time_s=t,pose=s.pose.copy(),velocity=s.velocity.copy(),
                                 speed=s.signed_speed_m_s,muscle=s.muscle.copy(),command=command,
                                 calcium_mM=ca,activation=activation,internal_time_s=s.internal_time_s))
            if tick==count:break
            if t>=8-1e-12:moments+=np.stack([ca,ca*ca,activation,activation*activation])*dt
            body.step(1,muscle_drive=command)
    arrays={k:np.array([row[k] for row in rows]) for k in rows[0]}
    arrays.update(input_moments=moments,muscle_names=np.array(names))
    np.savez_compressed(folder/'trace.npz',**arrays)
    write(folder/'record.json',dict(complete=True,wall_s=time.monotonic()-started,dt_s=dt,
                                  trace_sha256=sha(folder/'trace.npz'),protocol_sha256=sha(root/'protocol.json')))
    print(spec['name'],'complete',round(time.monotonic()-started,2),'s',flush=True)


def run(root):
    verify(root);design=json.loads((root/'protocol.json').read_text());failures=[]
    for spec in design['specs']:
        folder=root/'cases'/spec['name']
        if folder.exists():raise FileExistsError(folder)
        try:run_case(root,spec)
        except Exception as exc:
            failure=dict(name=spec['name'],error=repr(exc));failures.append(failure)
            write(folder/'failure.json',failure);print('FAILED',failure,flush=True)
    verify(root);write(root/'execution.json',dict(new_runs=len(design['specs']),failures=failures))
    if failures:raise RuntimeError(f'{len(failures)} failed cases retained')


def harmonic(t,x,f):
    phase=2*np.pi*f*t
    matrix=np.column_stack([np.ones(len(t)),np.cos(phase),np.sin(phase)])
    fit=np.linalg.lstsq(matrix,x.reshape(len(t),-1),rcond=None)[0]
    return fit[1]-1j*fit[2]


def metrics(folder,name,scale,frequency):
    record=json.loads((folder/'record.json').read_text())
    assert record['complete'] and record['trace_sha256']==sha(folder/'trace.npz')
    z=np.load(folder/'trace.npz');t=z['time_s'];late=(t>=8-1e-10)&(t<20-1e-10);inclusive=t>=8-1e-10
    assert len(t)==2001 and np.allclose(t,np.arange(2001)*.01,rtol=0,atol=1e-12)
    assert all(np.isfinite(z[k]).all() for k in z.files if z[k].dtype.kind in 'fi')
    assert np.all(z['internal_time_s']<=t+1e-10)
    k=curvature(z['pose']);kh=harmonic(t[late],k[late],frequency)*1e-3
    center=z['pose'][:,:,:2].mean(1)
    mh=harmonic(t[late],z['muscle'][late,:,0]-z['muscle'][late,:,1],frequency)
    local=(mh[:-1]+mh[1:])/2
    krms=float(np.sqrt(np.mean(abs(kh[4:-4])**2)))
    return dict(name=name,scale=scale,frequency_Hz=frequency,dt_s=record['dt_s'],
                trace_sha256=record['trace_sha256'],signed_speed_um_s=float(np.trapezoid(z['speed'][inclusive],t[inclusive])/12*1e6),
                net_displacement_um=float(np.linalg.norm(center[-1]-center[800])*1e6),
                curvature_harmonic_rms_per_mm=krms,
                curvature_per_muscle=krms/float(np.sqrt(np.mean(abs(local[4:-4])**2))),
                central_length_min_mm=float(np.linalg.norm(np.diff(z['pose'][:,:,:2],axis=1),axis=2).sum(1).min()*1e3),
                central_length_max_mm=float(np.linalg.norm(np.diff(z['pose'][:,:,:2],axis=1),axis=2).sum(1).max()*1e3))


def summarize(root,output,plots=False):
    verify(root);design=json.loads((root/'protocol.json').read_text());execution=json.loads((root/'execution.json').read_text())
    assert not execution['failures']
    cases=[metrics(root/'cases'/s['name'],s['name'],s['scale'],s['frequency']) for s in design['specs']]
    cases += [metrics(root/'reused'/s['name'],s['name'],1.,s['frequency']) for s in design['reused']]
    byname={c['name']:c for c in cases};comparisons=[]
    for f in [.5,1.,2.]:
        baseline=np.load(root/'reused'/f's1_f{f:g}'/'trace.npz')
        for scale in [.5,2.]:
            z=np.load(root/'cases'/f's{scale:g}_f{f:g}'/'trace.npz')
            values={k:bool(np.array_equal(baseline[k],z[k])) for k in ['time_s','calcium_mM','activation','command','muscle','input_moments']}
            comparisons.append(dict(scale=scale,frequency_Hz=f,identical_arrays=values,passed=all(values.values())))
    refinement=[]
    for spec in design['specs']:
        if 'paired_with' not in spec:continue
        fine=byname[spec['name']];coarse=byname[spec['paired_with']]
        speed=abs(coarse['signed_speed_um_s']-fine['signed_speed_um_s'])
        kerror=abs(coarse['curvature_harmonic_rms_per_mm']-fine['curvature_harmonic_rms_per_mm'])/max(fine['curvature_harmonic_rms_per_mm'],1e-9)
        limit=max(.2,.02*abs(fine['signed_speed_um_s']))
        refinement.append(dict(coarse=coarse['name'],fine=fine['name'],speed_error_um_s=speed,
                               speed_limit_um_s=limit,curvature_relative_error=kerror,
                               passed=bool(speed<=limit and kerror<=.03)))
    constants=[json.loads(p.read_text()) for p in sorted((root/'build').glob('*/parameters.json'))]
    checks=dict(all_eight_new_cases_complete=len(execution['failures'])==0 and execution['new_runs']==8,
                all_three_controls_reused_and_hashed=len(design['reused'])==3,
                all_finite_causal=True,only_declared_passive_force_terms_changed=all(c['source_reversal_exact'] for c in constants),
                all_active_damping_environment_constants_unchanged=all(
                    all(abs(c['parameters'][k]-v)<=max(1e-18,1e-14*abs(v)) for k,v in c['unchanged_expected'].items()) for c in constants),
                calcium_command_actual_muscle_identical=all(c['passed'] for c in comparisons),
                both_half_step_pairs_pass=all(c['passed'] for c in refinement))
    output.mkdir(parents=True,exist_ok=True)
    summary=dict(protocol_sha256=sha(root/'protocol.json'),checks=checks,cases=cases,
                 pointwise_comparisons=comparisons,refinement=refinement,parameter_audit=constants,
                 interpretation='Model passive skeleton elasticity under prescribed calcium; no neuronal autonomy, animal softness calibration or thermotaxis claim.')
    write(output/'summary.json',summary)
    print(json.dumps(dict(checks=checks,refinement=refinement),indent=2),flush=True)
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                             'axes.spines.right':False,'figure.facecolor':'#fbfaf7','axes.facecolor':'#fbfaf7',
                             'savefig.facecolor':'#fbfaf7'})
        fig,axes=plt.subplots(1,3,figsize=(14.8,4.7),constrained_layout=True)
        for scale,color,label in [(.5,'#d38350','0.5× passive elasticity'),(1.,'#258b8a','1× original elasticity'),(2.,'#6d6aa4','2× passive elasticity')]:
            values=[byname[f's{scale:g}_f{f:g}'] for f in [.5,1.,2.]]
            axes[0].plot([.5,1,2],[x['signed_speed_um_s'] for x in values],'o-',lw=2.4,color=color,label=label)
            axes[1].plot([.5,1,2],[x['curvature_harmonic_rms_per_mm'] for x in values],'o-',lw=2.4,color=color)
            axes[2].plot([.5,1,2],[x['curvature_per_muscle'] for x in values],'o-',lw=2.4,color=color)
        axes[0].set(title='A  Translation',ylabel='Forward speed (µm/s)')
        axes[1].set(title='B  Body bending',ylabel='Curvature amplitude (mm$^{-1}$)')
        axes[2].set(title='C  Mechanical transmission',ylabel='Curvature / muscle amplitude (mm$^{-1}$)')
        axes[0].legend(frameon=False,fontsize=9)
        for ax in axes:ax.set_xlabel('Drive frequency (Hz)');ax.set_xticks([.5,1,2]);ax.grid(alpha=.15)
        fig.suptitle('Identical muscle activity, different passive skeleton elasticity',fontsize=16,weight='bold')
        fig.savefig(output/'passive-stiffness-summary.png',dpi=180)
        fig.savefig(output/'passive-stiffness-summary.svg');plt.close(fig)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','verify','summarize'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=ASSETS);p.add_argument('--plots',action='store_true')
    a=p.parse_args()
    if a.action=='summarize':summarize(a.root,a.output,a.plots)
    else:{'prepare':prepare,'run':run,'verify':verify}[a.action](a.root)
