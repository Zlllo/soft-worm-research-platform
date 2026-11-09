"""Recompute all preregistered gates from raw NPZ, without rerunning NEURON."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'results/b-motor-convergence-2026-11-09'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def write(p,d): p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def rms(x,axis=None): return np.sqrt(np.mean(np.square(x),axis=axis))


def shape(z):
    xy=z['pose'][:,:,:2]*1e6
    r=xy-xy.mean(1,keepdims=True)
    axis=xy[:,0]-xy[:,-1];axis/=np.linalg.norm(axis,axis=1,keepdims=True)
    return r[:,:,0]*(-axis[:,None,1])+r[:,:,1]*axis[:,None,0]


def load(name):
    d=read(ROOT/name/'record.json')
    assert d['complete'] and sha(ROOT/name/'trace.npz')==d['trace_sha256']
    provenance=read(ROOT/name/'provenance.json')
    assert provenance['protocol_sha256']==sha(ROOT/'protocol.json')
    assert provenance['inputs_sha256']==sha(ROOT/'inputs.json')
    z=dict(np.load(ROOT/name/'trace.npz',allow_pickle=False))
    plan=read(ROOT/'connection-plan.json')
    assert z['neuron_names'].tolist()==plan['neurons']
    required=dict(model='prototype',replay=None,gap_nS=.1,feedback_gain=100,feedback_clip=1,
        duration=8.,seed=13,avb_on=True,calcium_scale=1,perturb=False,nmj_nS=.9)
    assert all(d['spec'][k]==v for k,v in required.items())
    assert d['spec']['dt_ms'] in [.025,.0125,.00625]
    for k,v in z.items():
        if np.issubdtype(v.dtype,np.number): assert np.isfinite(v).all(),(name,k)
    assert z['neuron_voltage_mV'].shape==(801,302)
    assert z['muscle_calcium_mM'].shape==(801,95)
    assert z['drive'].shape==(801,48,2) and z['feedback_pA'].shape==(8001,18)
    assert (d['neurons'],d['muscles'],d['chemical_connections'],d['gap_pairs'])==(302,95,4665,1192)
    assert np.max(abs(z['time_s']-np.arange(801)*.01))<1e-8
    assert np.array_equal(z['exchange_time_s'],np.arange(8001)*.001)
    assert np.all(z['external_drive']==0)
    assert d['max_gap_pair_current_imbalance_nA']<1e-12
    assert z['neuron_voltage_mV'].min()>=-100 and z['neuron_voltage_mV'].max()<=60
    strain=(z['side_length']-z['side_length'][0])/z['side_length'][0]
    sensed=((strain[:,:,1]-strain[:,:,0])/2)@z['feedback_weights'].T*np.where(z['feedback_sides']==0,1.,-1.)
    assert np.max(abs(sensed-z['sensed_strain']))<1e-12
    expected=np.clip(100*sensed,-1,1) if d['spec']['feedback']=='live' else np.zeros_like(sensed)
    assert np.max(abs(expected-z['feedback_pA'][::10]))<1e-12
    n=z['neuron_names'].tolist();ix=[n.index(x) for x in z['B_names']]
    z['B']=z['neuron_voltage_mV'][:,ix]
    z['strain']=strain
    z['body_frame_y_um']=shape(z)
    z['late']=z['time_s']>=4-1e-8
    v=z['B'][z['late']];t=z['time_s'][z['late']]
    peaks=[find_peaks(v[:,j],prominence=3)[0] for j in range(18)]
    assert all(len(p)>=5 for p in peaks), 'Loss of B rhythm'
    f=np.array([1/np.diff(t[p]).mean() for p in peaks])
    displacement=float(np.linalg.norm(z['pose'][-1,:,:2].mean(0)-z['pose'][400,:,:2].mean(0))*1e6)
    speed=float(z['speed'][z['late']].mean()*1e6)
    assert abs(speed-d['late_mean_speed_um_s'])<1e-10
    assert abs(displacement-d['late_centroid_displacement_um'])<1e-10
    m=dict(dt_ms=d['spec']['dt_ms'],feedback=d['spec']['feedback'],B_names=z['B_names'].tolist(),
        B_peak_counts=[len(p) for p in peaks],B_frequency_Hz=f.tolist(),
        B_amplitude_mV=(np.percentile(v,95,axis=0)-np.percentile(v,5,axis=0)).tolist(),
        B_mean_mV=v.mean(0).tolist(),B_std_mV=v.std(0).tolist(),
        calcium_mean_mM=z['muscle_calcium_mM'][z['late']].mean(0).tolist(),
        calcium_std_mM=z['muscle_calcium_mM'][z['late']].std(0).tolist(),
        calcium_max_mM=float(z['muscle_calcium_mM'][z['late']].max()),
        drive_mean=float(z['drive'][z['late']].mean()),drive_mean_channel_std=float(z['drive'][z['late']].std(0).mean()),
        drive_max=float(z['drive'][z['late']].max()),
        body_frame_lateral_amplitude_um=float(np.ptp(z['body_frame_y_um'][z['late']],axis=0).max()),
        laboratory_lateral_amplitude_um=float(np.ptp(z['pose'][z['late'],:,1],axis=0).max()*1e6),
        side_strain_RMS=float(rms(strain[z['late']])),
        late_speed_um_s=speed,late_speed_RMS_um_s=float(rms(z['speed'][z['late']])*1e6),
        late_displacement_um=displacement,feedback_max_pA=float(abs(z['feedback_pA']).max()),
        trace_sha256=d['trace_sha256'],wall_s=d['wall_s'],
        propulsion_gate_passed=bool(speed>=10 and displacement>=20))
    return z,m


def compare(a,b,ma,mb,key):
    """a is coarser; b is reference. No shifts are applied to acceptance gates."""
    p=read(ROOT/'protocol.json')['acceptance'];g={};e={};details={}
    # Runs share intended exchange/sample ticks. Do not interpret sub-ns
    # accumulated NEURON floating-point clock drift as a phase difference.
    mask=a['late'];t=np.arange(801)*.01;va=a['B'][mask];vb=b['B'][mask]
    def gate(name,value,limit):
        e[name]=dict(value=float(value),limit=float(limit));g[name]=bool(value<=limit+1e-12)
    fa=np.array(ma['B_frequency_Hz']);fb=np.array(mb['B_frequency_Hz'])
    frequency=abs(fa-fb)/fb
    amp=abs(np.array(ma['B_amplitude_mV'])-mb['B_amplitude_mV'])/mb['B_amplitude_mV']
    vrms=rms(va-vb,axis=0)
    gate('B_frequency',frequency.max(),p['B_frequency_max_relative'])
    gate('B_amplitude',amp.max(),p['B_q95_q05_amplitude_max_relative'])
    gate('B_mean',abs(va.mean(0)-vb.mean(0)).max(),p['B_mean_max_abs_mV'])
    gate('B_trajectory',vrms.max(),p['B_unshifted_RMS_max_mV'])
    lags=[];signed_lags=[];aligned=[]
    for j in range(18):
        extended=t>=3.5-1e-8
        ta=t[extended][find_peaks(a['B'][extended,j],prominence=3)[0]]
        tb=t[extended][find_peaks(b['B'][extended,j],prominence=3)[0]]
        aa=ta[ta>=4-1e-8];bb=tb[tb>=4-1e-8]
        ab=tb[np.argmin(abs(aa[:,None]-tb[None,:]),axis=1)]-aa
        ba=ta[np.argmin(abs(bb[:,None]-ta[None,:]),axis=1)]-bb
        lags.append(float(max(abs(ab).max(),abs(ba).max())));signed_lags.append(float(np.median(ab)))
        # Diagnostic only: one constant shift, grid search +-250 ms, identical fixed interior window.
        errors=[]
        for shift in range(-25,26):
            ix=np.arange(425,775)
            errors.append(float(rms(a['B'][ix,j]-b['B'][ix+shift,j])))
        best=int(np.argmin(errors));aligned.append(dict(shift_s=(best-25)*.01,RMS_mV=errors[best],
            unshifted_same_window_RMS_mV=errors[25],fixed_window_s=[4.25,7.74]))
    gate('B_peak_phase',max(lags),p['B_peak_time_max_abs_s'])
    details.update(B_frequency_relative=frequency.tolist(),B_amplitude_relative=amp.tolist(),
        phase_clock='Canonical 10 ms recorded ticks; original neural clocks separately checked and retained',
        B_unshifted_RMS_mV=vrms.tolist(),B_max_abs_voltage_difference_mV=float(abs(va-vb).max()),
        all302_unshifted_RMS_mV=float(rms(a['neuron_voltage_mV'][mask]-b['neuron_voltage_mV'][mask])),
        B_bidirectional_peak_offset_max_s=lags,B_median_signed_peak_offset_s=signed_lags,
        B_constant_shift_diagnostic=aligned)
    for field,label,floor in [('muscle_calcium_mM','calcium',p['calcium_scale_floor_mM']),('drive','drive',p['drive_scale_floor'])]:
        x=a[field][mask].reshape(401,-1);y=b[field][mask].reshape(401,-1)
        scale=np.maximum(np.percentile(y,95,axis=0)-np.percentile(y,5,axis=0),floor)
        stats=np.stack([abs(x.mean(0)-y.mean(0)),abs(x.std(0)-y.std(0)),
            abs(np.percentile(x,5,axis=0)-np.percentile(y,5,axis=0)),
            abs(np.percentile(x,95,axis=0)-np.percentile(y,95,axis=0))])/scale
        trace=rms(x-y,axis=0)/scale
        gate(label+'_statistics',stats.max(),.05);gate(label+'_trajectory',trace.max(),.05)
        details[label+'_scaled_RMS_per_channel']=trace.tolist()
        details[label+'_scaled_stats_max_per_channel']=stats.max(0).tolist()
        details[label+'_absolute_RMS']=float(rms(x-y))
    dy=a['body_frame_y_um'][mask]-b['body_frame_y_um'][mask]
    gate('body_shape_trajectory',rms(dy),p['body_shape_RMS_max_um'])
    gate('body_shape_amplitude',abs(ma['body_frame_lateral_amplitude_um']-mb['body_frame_lateral_amplitude_um']),
        max(p['body_lateral_amplitude_abs_um'],p['body_lateral_amplitude_relative']*mb['body_frame_lateral_amplitude_um']))
    gate('side_strain_trajectory',rms(a['strain'][mask]-b['strain'][mask]),max(p['side_strain_RMS_abs'],p['side_strain_RMS_relative']*rms(b['strain'][mask])))
    gate('mean_speed',abs(ma['late_speed_um_s']-mb['late_speed_um_s']),p['speed_mean_abs_difference_um_s'])
    gate('speed_trajectory',rms(a['speed'][mask]-b['speed'][mask])*1e6,p['speed_trace_RMS_um_s'])
    gate('centroid_displacement',abs(ma['late_displacement_um']-mb['late_displacement_um']),p['centroid_displacement_abs_difference_um'])
    fm=a['exchange_time_s']>=4-1e-8
    gate('feedback_trajectory',rms(a['feedback_pA'][fm]-b['feedback_pA'][fm]),
        max(p['feedback_RMS_abs_pA'],p['feedback_RMS_relative']*rms(b['feedback_pA'][fm])))
    difference=dict(time_s=t,neuron_voltage_difference_mV=a['neuron_voltage_mV']-b['neuron_voltage_mV'],
        B_voltage_difference_mV=a['B']-b['B'],calcium_difference_mM=a['muscle_calcium_mM']-b['muscle_calcium_mM'],
        drive_difference=a['drive']-b['drive'],body_frame_y_difference_um=a['body_frame_y_um']-b['body_frame_y_um'],
        pose_difference_m=a['pose']-b['pose'],side_strain_difference=a['strain']-b['strain'],
        signed_speed_difference_um_s=(a['speed']-b['speed'])*1e6,
        exchange_time_s=a['exchange_time_s'],feedback_difference_pA=a['feedback_pA']-b['feedback_pA'])
    np.savez_compressed(ROOT/'differences'/f'{key}.npz',**difference)
    return dict(coarse_dt_ms=ma['dt_ms'],reference_dt_ms=mb['dt_ms'],feedback=ma['feedback'],
        passed=all(g.values()),gates=g,failed_gates=[k for k,v in g.items() if not v],errors=e,details=details)


def main():
    protocol=read(ROOT/'protocol.json');assert sha(ROOT/'protocol.json')==read(ROOT/'protocol-lock.json')['protocol_sha256']
    for path,d in read(ROOT/'inputs.json').items(): assert sha(ROOT/path)==d['sha256']
    for path,d in read(ROOT/'historical-baselines/sources.json').items(): assert sha(ROOT/path)==d['sha256']
    (ROOT/'differences').mkdir(exist_ok=True)
    metrics={};traces={}
    for mode in ['off','live']:
        for dt in protocol['steps_ms']:
            name=f'{mode}_dt{dt:g}'
            if (ROOT/name/'record.json').exists():traces[name],metrics[name]=load(name)
    comparisons={}
    for mode in ['off','live']:
        for lo,hi in [(.025,.0125),(.0125,.00625),(.025,.00625)]:
            na=f'{mode}_dt{lo:g}';nb=f'{mode}_dt{hi:g}'
            if na in traces and nb in traces:
                key=f'{mode}_{lo:g}_vs_{hi:g}'
                comparisons[key]=compare(traces[na],traces[nb],metrics[na],metrics[nb],key)
    historical={}
    for mode in ['off','live']:
        name=f'{mode}_dt0.025'
        if name in traces:
            old=ROOT/'historical-baselines'/f'nmj9_{mode}8'
            record=read(old/'record.json');assert sha(old/'trace.npz')==record['trace_sha256']
            with np.load(old/'trace.npz') as z:
                oldspec={k:v for k,v in record['spec'].items() if k!='name'}
                newspec={k:v for k,v in read(ROOT/name/'record.json')['spec'].items() if k!='name'}
                assert oldspec==newspec
                differences={k:float(np.max(abs(z[k]-traces[name][k]))) for k in z.files if np.issubdtype(z[k].dtype,np.number)}
                historical[mode]=dict(old_trace_sha256=record['trace_sha256'],same_spec_except_name=True,
                    max_abs_array_difference=differences,all_numeric_arrays_identical=all(x==0 for x in differences.values()))
    fine_required=any(not comparisons[f'{mode}_0.025_vs_0.0125']['passed'] for mode in ['off','live']) if len(metrics)>=4 else None
    latest_pair=(.0125,.00625) if len(metrics)==6 else (.025,.0125)
    final_keys=[f'{mode}_{latest_pair[0]:g}_vs_{latest_pair[1]:g}' for mode in ['off','live']]
    final=all(comparisons[k]['passed'] for k in final_keys) if all(k in comparisons for k in final_keys) else None
    effects={}
    for dt in protocol['steps_ms']:
        no=f'off_dt{dt:g}';nl=f'live_dt{dt:g}'
        if no in traces and nl in traces:
            off=traces[no];live=traces[nl];mask=off['late']
            effects[str(dt)]=dict(B_live_minus_off_RMS_mV=float(rms(live['B'][mask]-off['B'][mask])),
                B_live_minus_off_max_abs_mV=float(abs(live['B'][mask]-off['B'][mask]).max()),
                body_frame_live_minus_off_RMS_um=float(rms(live['body_frame_y_um'][mask]-off['body_frame_y_um'][mask])),
                speed_live_minus_off_um_s=metrics[nl]['late_speed_um_s']-metrics[no]['late_speed_um_s'])
    refinement_ratios={}
    if len(metrics)==6:
        for mode in ['off','live']:
            c=comparisons[f'{mode}_0.025_vs_0.0125']['errors']
            f=comparisons[f'{mode}_0.0125_vs_0.00625']['errors']
            refinement_ratios[mode]={k:(f[k]['value']/d['value'] if d['value']>1e-15 else None) for k,d in c.items()}
    domains=dict(B_rhythm=['B_frequency','B_amplitude','B_mean'],
        B_trajectory_and_peak_phase=['B_trajectory','B_peak_phase'],
        muscle_calcium_and_drive=['calcium_statistics','calcium_trajectory','drive_statistics','drive_trajectory'],
        body_shape=['body_shape_trajectory','body_shape_amplitude'],body_side_strain=['side_strain_trajectory'],
        near_zero_movement=['mean_speed','speed_trajectory','centroid_displacement'],feedback_current=['feedback_trajectory'])
    domain_pass={mode:{domain:all(comparisons[key]['gates'][g] for g in gates) for domain,gates in domains.items()}
        for mode,key in zip(['off','live'],final_keys) if key in comparisons}
    summary=dict(protocol_sha256=sha(ROOT/'protocol.json'),new_full_runs=len(metrics),checks_all_passed=True,
        checks=['Input/trace/protocol SHA','302/95/4665/1192 populations','18 B rhythms','801 state samples / 8001 exchange samples',
                'Clocks / finite states / voltage guards','No external drive','Gap current balance','Actual shape feedback reconstruction','Record statistics recomputed'],
        historical_baseline_consistency=historical,metrics=metrics,comparisons=comparisons,
        feedback_on_off_descriptive_differences=effects,error_ratios_finer_pair_over_coarser_pair=refinement_ratios,
        latest_pair_domains_passed=domain_pass,
        fine_required=fine_required,latest_pair_ms=latest_pair,latest_pair_passed=final,
        neural_step_convergence_accepted=bool(final and (len(metrics)==6 or fine_required is False)),
        boundary='Engineering prototype, no physiological fit; no sustained propulsion; neural dt only, fixed 1 ms exchange/body')
    write(ROOT/'summary.json',summary)
    print(json.dumps(dict(runs=len(metrics),fine_required=fine_required,latest_pair_passed=final,
        failed={k:v['failed_gates'] for k,v in comparisons.items()}),ensure_ascii=False))


if __name__=='__main__':main()
