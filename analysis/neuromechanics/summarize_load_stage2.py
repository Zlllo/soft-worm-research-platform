"""Offline wave/geometry metrics; analytic phase never enters neural feedback."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from run_load_feedback_stage2 import REPO,check,write
from thermotaxis.neuromechanics.boyle_output import curvature


def signals(path,period):
    with np.load(path) as z:
        values={k:z[k].copy() for k in z.files}
    k=curvature(values['pose']);front=k[:,:5].mean(1);back=k[:,35:45].mean(1)
    analytic_front=hilbert(front-front.mean());analytic_back=hilbert(back-back.mean())
    lag=np.angle(analytic_back*np.conj(analytic_front))
    center=values['pose'][:,:,:2].mean(1);dt=float(np.median(np.diff(values['time_s'])))
    speed=np.linalg.norm(np.gradient(center,dt,axis=0),axis=1)
    count=max(3,round(period/dt));count+=1-count%2
    speed=np.convolve(speed,np.ones(count)/count,mode='same')
    return dict(**values,curvature=k,lag=lag,speed=speed)


def compare(run,nominal,onset,duration,period,design):
    t=run['time_s'];off=onset+duration;end=t[-1]-period
    selected=(t>=onset)&(t<=end);after=(t>=off)&(t<=end)
    phase_error=np.angle(np.exp(1j*(run['lag']-nominal['lag'])))
    speed_error=(run['speed']-nominal['speed'])/np.maximum(nominal['speed'],1e-8)
    good=(np.abs(phase_error)<=design['metrics']['recovery_lag_tolerance_rad'])&(abs(speed_error)<=design['metrics']['recovery_speed_tolerance_fraction'])
    hold=round(period/design['sample_dt_s']);recovery=None;length=0
    for i in np.flatnonzero(after):
        length=length+1 if good[i] else 0
        if length>=hold:
            recovery=float(t[i-hold+1]-off);break
    posterior_error=run['curvature'][:,24:]-nominal['curvature'][:,24:]
    scale=float(np.std(nominal['curvature'][:,24:]))
    return dict(lag_abs_integral_rad_s=float(np.trapezoid(abs(phase_error[selected]),t[selected])),
        lag_post_rms_rad=float(np.sqrt(np.mean(phase_error[after]**2))),
        speed_post_abs_fraction=float(np.mean(abs(speed_error[after]))),
        posterior_pointwise_rms_normalized=float(np.sqrt(np.mean(posterior_error[after]**2))/scale),
        recovery_s=recovery,recovery_cycles=None if recovery is None else recovery/period,
        neural_change_rms=float(np.sqrt(np.mean((run['neural'][selected]-nominal['neural'][selected])**2))),
        muscle_change_rms=float(np.sqrt(np.mean((run['muscle'][selected]-nominal['muscle'][selected])**2))),
        maximum_actual_vs_used_length_m=float(np.max(abs(run['actual_side']-run['used_side']))),
        evaluation_end_s=float(end)),phase_error,speed_error


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/body-load-stage2-2026-11-03'))
    p.add_argument('--output',type=Path,default=Path('docs/reports/assets/body-load-stage2-2026-11-03'))
    a=p.parse_args();root,out=a.root,a.output
    design=json.loads((root/'design.json').read_text());check(design)
    conditions=json.loads((root/'conditions.json').read_text());assert conditions['complete']
    clock=json.loads((root/'baseline-clock.json').read_text());period=clock['period_s']
    nominal=signals(root/'nominal-live/trajectory.npz',period)
    rows=[];cache={};errors={}
    for row in conditions['rows']:
        key=row['path']
        if key not in cache:cache[key]=signals(root/key/'trajectory.npz',period)
        metric,phase_error,speed_error=compare(cache[key],nominal,row['onset_s'],design['pulse_duration_s'],period,design)
        rows.append(dict(**row,**metric));errors[key]=(phase_error,speed_error)
        if row['feedback']=='replay':
            np.testing.assert_array_equal(cache[key]['neural'],nominal['neural'])
            np.testing.assert_array_equal(cache[key]['muscle'],nominal['muscle'])
    groups=[]
    for factor in design['factors']:
        for mode in design['feedback']:
            selected=[r for r in rows if r['factor']==factor and r['feedback']==mode]
            keys=['lag_abs_integral_rad_s','lag_post_rms_rad','speed_post_abs_fraction','neural_change_rms','muscle_change_rms']
            groups.append(dict(factor=factor,feedback=mode,**{k:float(np.mean([r[k] for r in selected])) for k in keys},
                recovery_count=sum(r['recovery_s'] is not None for r in selected),
                recovery_s_by_phase=[r['recovery_s'] for r in selected]))
    half_nominal=signals(root/'half-dt-nominal/trajectory.npz',period)
    numerical=[]
    for mode in design['feedback']:
        half=signals(root/f'half-dt-load-2-{mode}/trajectory.npz',period)
        half_metric,_,_=compare(half,half_nominal,clock['onsets_s'][0],design['pulse_duration_s'],period,design)
        coarse=next(r for r in rows if r['phase']==0 and r['factor']==2 and r['feedback']==mode)
        numerical.append(dict(feedback=mode,coarse=coarse,half_dt=half_metric,
            maximum_position_difference_m=float(np.max(abs(half['pose'][:,:,:2]-cache[coarse['path']]['pose'][:,:,:2])))))
    result=dict(status='completed_load_replay_assay',design=design,period_s=period,rows=rows,groups=groups,
        nominal_replay_identity=conditions['nominal_replay_identity'],numerical=numerical,
        simulated_runs=len(list(root.glob('*/run.json'))),
        total_run_wall_s=sum(json.loads(p.read_text())['wall_s'] for p in root.glob('*/run.json')),
        maximum_geometry_error_m=max(json.loads(p.read_text())['max_geometry_error_m'] for p in root.glob('*/run.json')),
        note='48 condition records include reused nominal runs; 8 pulse phases are initial-condition robustness, not biological replicates')
    out.mkdir(parents=True,exist_ok=True);write(out/'summary.json',result);write(root/'summary.json',result)
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),constrained_layout=True)
    for mode,color in [('live','#2d6bad'),('replay','#c98024')]:
        run=cache[f'phase-00-load-2-{mode}'];t=run['time_s'];onset=clock['onsets_s'][0]
        phase_error,speed_error=errors[f'phase-00-load-2-{mode}']
        axes[0,0].plot(t,phase_error,label=mode,color=color)
        axes[0,1].plot(t,speed_error*100,label=mode,color=color)
        for j,factor in enumerate([1.5,2.]):
            vals=[r['lag_abs_integral_rad_s'] for r in rows if r['feedback']==mode and r['factor']==factor]
            axes[1,0].scatter(np.full(8,j+(-.08 if mode=='live' else .08)),vals,s=22,alpha=.65,color=color)
        vals=[r['lag_abs_integral_rad_s'] for r in groups if r['feedback']==mode and r['factor']>1]
        axes[1,0].plot(range(2),vals,'o-',label=mode,color=color)
        for j,factor in enumerate([1.5,2.]):
            r=next(r for r in groups if r['factor']==factor and r['feedback']==mode)
            axes[1,1].plot(j,r['neural_change_rms'],'o',color=color,label=mode if j==0 else None)
    for ax in axes[0]:ax.axvspan(onset,onset+2,color='gray',alpha=.18);ax.set_xlim(3,20-period)
    axes[0,0].set(xlabel='Time (s)',ylabel='Change in posterior/head phase lag (rad)',title='A  Prespecified phase 0, double anterior drag')
    axes[0,1].set(xlabel='Time (s)',ylabel='Cycle-smoothed speed deviation (%)',title='B  Speed relative to matched nominal gait')
    axes[1,0].set(xticks=[0,1],xticklabels=['1.5 x','2 x'],ylabel='Absolute lag-error integral (rad s)',title='C  All 8 pulse phases, means and individual values')
    axes[1,1].set(xticks=[0,1],xticklabels=['1.5 x','2 x'],ylabel='Neural-drive RMS change',title='D  Replay clamps actuation despite changed geometry')
    for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'load-feedback.png',dpi=170);fig.savefig(out/'load-feedback.svg');plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(9,7),constrained_layout=True,sharex=True)
    for ax,name in zip(axes,['nominal-live','phase-00-load-2-live','phase-00-load-2-replay']):
        run=nominal if name=='nominal-live' else cache[name]
        image=ax.imshow(run['curvature'].T*1e-3,origin='upper',aspect='auto',extent=[0,20,1,0],cmap='RdBu_r',vmin=-12,vmax=12)
        ax.set(ylabel='Head to tail',title=name);ax.axvline(clock['onsets_s'][0],color='black',lw=.8)
        ax.axvline(clock['onsets_s'][0]+2,color='black',lw=.8)
    axes[-1].set_xlabel('Time (s)');fig.colorbar(image,ax=list(axes),label='Curvature (mm^-1)')
    fig.savefig(out/'curvature.png',dpi=170);fig.savefig(out/'curvature.svg');plt.close(fig)
    print(json.dumps(dict(period_s=period,groups=groups,numerical=numerical),indent=2),flush=True)


if __name__=='__main__':main()
