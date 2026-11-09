"""Independent saved-data audit; --archive-only does not require native binaries."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))


def main(root,prototype,output,archive_only):
    d=read(root/'design.json')
    if not archive_only:
        from run_full302_closed_loop_rectified import check
        check(root)
    names=[s['name'] for s in d['specs']]+['bend_D_live-half-neural','bend_D_live-exchange2ms']
    traces={};records=[]
    mapping=read(root/'feedback-map.json');weights=np.array(mapping['weights']);sides=np.array(mapping['sides'])
    sampling_root=Path('results/full302-active-sampling-probe-2026-11-05')
    entries=[(name,root/name) for name in names]+[(name,sampling_root/name) for name in ['sampling_D_live','sampling_D_frozen']]
    if not archive_only:
        check(sampling_root)
    for name,folder in entries:
        record=read(folder/'record.json');assert record['complete']
        assert record['neurons']==302 and record['muscles']==95
        assert sha(folder/'trace.npz')==record['trace_sha256']
        with np.load(folder/'trace.npz') as z:values={k:z[k] for k in z.files}
        traces[name]=values;s=record['spec'];t=values['exchange_time_s'];dt=record['exchange_ms']/1000
        assert values['neuron_voltage_mV'].shape[1]==302 and values['muscle_voltage_mV'].shape[1]==95
        assert all(np.isfinite(v).all() for v in values.values())
        assert values['neuron_voltage_mV'].min()>=-100 and values['neuron_voltage_mV'].max()<=60
        np.testing.assert_allclose(np.diff(t),dt,rtol=0,atol=1e-12)
        warming=np.where((t>=.2-1e-12)&(t<1.2-1e-12),.05,0) if s['thermal']=='warming' else np.zeros(len(t))
        spatial=40*values['head_xy_m'][:,1] if s['thermal']=='warming' else np.zeros(len(t))
        np.testing.assert_allclose(values['actual_temperature_K'],293.15+spatial+warming,rtol=0,atol=1e-12)
        received=293.15+(np.full(len(t),spatial[0]) if s['frozen_head'] else spatial)+warming+values['noise_K']
        np.testing.assert_allclose(values['received_temperature_K'],received,rtol=0,atol=1e-12)
        filtered=threshold=293.15+spatial[0];expected=[]
        for i,temperature in enumerate(received):
            if i==0:current=0.
            else:
                filtered+=(1-np.exp(-dt/.1))*(temperature-filtered)
                current=5*float(np.clip((filtered-threshold)/.05,0,1))
                tau=45. if filtered>=threshold else 840.
                threshold+=(1-np.exp(-dt/tau))*(filtered-threshold)
            expected.append(current)
        np.testing.assert_allclose(values['AFD_current_pA'],expected,rtol=0,atol=1e-8)
        length=values['side_length'];strain=(length-length[0])/length[0]
        contrast=(strain[:,:,1]-strain[:,:,0])/2
        expected_fb=np.clip(500*(contrast@weights.T)*np.where(sides==0,1,-1)[None,:],0,5)
        indices=np.rint(values['time_s']/dt).astype(int)
        np.testing.assert_allclose(values['actual_feedback_candidate_pA'][indices],expected_fb,rtol=0,atol=1e-9)
        if s['feedback']=='live':np.testing.assert_array_equal(values['feedback_current_pA'],values['actual_feedback_candidate_pA'])
        elif s['feedback']=='off':assert not np.any(values['feedback_current_pA'])
        else:
            with np.load(root/s['replay']/'trace.npz') as ref:np.testing.assert_array_equal(values['feedback_current_pA'],ref['feedback_current_pA'])
        assert np.all(values['feedback_current_pA']>=0)
        records.append(record)
    identities={}
    for actual,replay in [('warm_live','warm_replay'),('bend_D_live','bend_D_replay')]:
        identities[actual]=all(np.array_equal(traces[actual][k],traces[replay][k]) for k in traces[actual])
        assert identities[actual]
    effects=[]
    for a,b in [('bend_D_live','bend_D_off'),('bend_V_live','bend_V_replay_D'),('warm_live','warm_off'),('warm_live','warm_frozen_head')]:
        x,y=traces[a],traces[b]
        effects.append(dict(a=a,b=b,neuron_voltage_rms_mV=rms(x['neuron_voltage_mV']-y['neuron_voltage_mV']),
            neuron_voltage_max_mV=float(abs(x['neuron_voltage_mV']-y['neuron_voltage_mV']).max()),
            max_point_difference_um=float(np.linalg.norm(x['pose'][:,:,:2]-y['pose'][:,:,:2],axis=2).max()*1e6)))
    motor=[]
    for name in ['motor_AVB_off','motor_AVB_live']:
        z=traces[name];mask=z['exchange_time_s']>=2;t=z['exchange_time_s'][mask];v=z['signed_speed_m_s'][mask]
        speed=float(np.sum((v[1:]+v[:-1])*.5*np.diff(t))/(t[-1]-t[0])*1e6)
        i=int(np.flatnonzero(z['time_s']>=2)[0]);axis=z['pose'][i,0,:2]-z['pose'][i,-1,:2];axis/=np.linalg.norm(axis)
        progress=float((z['pose'][-1,:,:2].mean(0)-z['pose'][i,:,:2].mean(0))@axis*1e6)
        motor.append(dict(name=name,last2s_mean_forward_um_s=speed,last2s_axis_progress_um=progress,
            passed=speed>=10 and progress>=20))
    temporal=prototype/'temporal-probe';td=read(temporal/'design.json')
    assert sha(temporal/'profiles.npz')==td['profiles_sha256']
    with np.load(temporal/'profiles.npz') as z:
        assert np.array_equal(np.sort(z['slow'][:-1]),np.sort(z['frontloaded'][:-1]))
    temporal_rows=[];temporal_traces={}
    for name in ['slow','frontloaded']:
        r=read(temporal/name/'record.json');assert r['complete'] and sha(temporal/name/'trace.npz')==r['trace_sha256']
        with np.load(temporal/name/'trace.npz') as z:values={k:z[k] for k in z.files}
        temporal_traces[name]=values
        temporal_rows.append(dict(name=name,integral_pA_s=r['current_integral_pA_s'],peak_pA=r['peak_current_pA'],
            maximum_calcium_mM=float(values['muscle_calcium_mM'].max()),
            maximum_drive=float(values['commanded_drive'].max()),
            max_point_difference_from_flat_um=float(np.linalg.norm(values['pose'][:,:,:2]-traces['flat_off']['pose'][:,:,:2],axis=2).max()*1e6)))
    numerical=[]
    for name in ['bend_D_live-half-neural','bend_D_live-exchange2ms']:
        a,b=traces['bend_D_live'],traces[name]
        numerical.append(dict(name=name,voltage_rms_difference_mV=rms(a['neuron_voltage_mV']-b['neuron_voltage_mV']),
            voltage_max_difference_mV=float(abs(a['neuron_voltage_mV']-b['neuron_voltage_mV']).max()),
            max_point_difference_um=float(np.linalg.norm(a['pose'][:,:,:2]-b['pose'][:,:,:2],axis=2).max()*1e6)))
    thermal=traces['warm_live'];spatial=40*(thermal['head_xy_m'][:,1]-thermal['head_xy_m'][0,1])
    max_sampling=float(abs(spatial).max())
    assert '225 passed, 14 warnings' in (root/'regression.log').read_text()
    a,b=traces['sampling_D_live'],traces['sampling_D_frozen']
    sampling=dict(max_head_displacement_um=float(np.linalg.norm(a['head_xy_m']-a['head_xy_m'][0],axis=1).max()*1e6),
        max_temperature_delivery_difference_K=float(abs(a['received_temperature_K']-b['received_temperature_K']).max()),
        max_AFD_current_difference_pA=float(abs(a['AFD_current_pA']-b['AFD_current_pA']).max()),
        max_neuron_voltage_difference_mV=float(abs(a['neuron_voltage_mV']-b['neuron_voltage_mV']).max()),
        externally_perturbed=True)
    assert sampling['max_temperature_delivery_difference_K']>1e-8
    assert sampling['max_AFD_current_difference_pA']>1e-6
    summary=dict(complete=True,full302_runs=17,neurons=302,muscles=95,replay_identities=identities,
        effects=effects,motor_gates=motor,temporal_probe=temporal_rows,numerical=numerical,
        active_sampling_verification=sampling,
        maximum_motion_induced_temperature_K=max_sampling,
        all_sampled_neuron_voltage_range_mV=[min(float(z['neuron_voltage_mV'].min()) for z in traces.values()),max(float(z['neuron_voltage_mV'].max()) for z in traces.values())],
        prototype_rejection=read(prototype/'prototype-validity.json'),project_tests=225,neural_fits=0,
        limits=['sensor transduction and mechanical field/gain are declared engineering assumptions',
                'no effective locomotion established; no thermotaxis training',
                'native1ms exchange and generic C1 parameters; fine spike-waveform convergence not claimed'])
    output.mkdir(parents=True,exist_ok=True);write(root/'summary.json',summary);write(output/'summary.json',summary)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for name in ['sampling_D_live','sampling_D_frozen']:
        z=traces[name];t=z['exchange_time_s'];warm=np.where((t>=.2-1e-12)&(t<1.2-1e-12),.05,0)
        axes[0,0].plot(t,(z['received_temperature_K']-293.15-warm)*1e6,label=name)
    axes[0,0].set(title='Head perturbation changes delivered temperature',ylabel='Spatial temperature term(uK)',xlabel='Time(s)')
    for name in ['bend_D_live','bend_V_live']:
        z=traces[name];j=mapping['names'].index('DB1')
        axes[0,1].plot(z['exchange_time_s'],z['feedback_current_pA'][:,j],label=name+' ->DB1')
    axes[0,1].set(title='One-sided B-cell feedback',ylabel='Current(pA)',xlabel='Time(s)')
    for name in ['slow','frontloaded']:
        z=temporal_traces[name]
        axes[1,0].plot(z['time_s'],z['muscle_calcium_mM'].max(1),label=name)
    axes[1,0].set(title='Same current values / integral, changed order',ylabel='Max model muscle calcium(mM)',xlabel='Time(s)')
    axes[1,0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    for name in ['motor_AVB_off','motor_AVB_live']:
        z=traces[name]
        axes[1,1].plot(z['exchange_time_s'],z['signed_speed_m_s']*1e6,label=name)
    axes[1,1].axhline(10,color='gray',linestyle='--',label='registered gate10um/s')
    axes[1,1].set(title='Tonic AVB motor probe: no sustained progression',ylabel='Signed forward speed(um/s)',xlabel='Time(s)')
    for ax in axes.flat:ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Full302 thermal/mechanical loops: implemented interfaces, motor gate not reached\nSigned-current prototype retained separately; no neural fitting')
    fig.tight_layout();fig.savefig(output/'closed-loop.png',dpi=160);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['effects','motor_gates','temporal_probe','numerical','maximum_motion_induced_temperature_K','all_sampled_neuron_voltage_range_mV']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/full302-closed-loop-rectified-2026-11-05'))
    p.add_argument('--prototype',type=Path,default=Path('results/full302-closed-loop-2026-11-05'))
    p.add_argument('--output',type=Path,default=Path('docs/reports/assets/full302-closed-loop-2026-11-05'))
    p.add_argument('--archive-only',action='store_true');a=p.parse_args();main(a.root,a.prototype,a.output,a.archive_only)
