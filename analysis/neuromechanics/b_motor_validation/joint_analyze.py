"""Frozen-criterion analysis of candidate runs and labeled historical control."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[3]/'results/full302-joint-numerics-2026-11-12'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def rms(x):return float(np.sqrt(np.mean(x*x)))


def helper():
    path=ROOT/'frozen-analysis/primary.py'
    assert sha(path)==read(ROOT/'sources.json')['frozen-analysis/primary.py']['sha256']
    spec=importlib.util.spec_from_file_location('joint_frozen_metric_functions',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=ROOT
    return m


def load(folder,new=True):
    d=read(folder/'record.json')
    if new:
        provenance=read(folder/'provenance.json')
        assert provenance['protocol_sha256']==sha(ROOT/'protocol.json')
        assert provenance['execution_lock_sha256']==sha(ROOT/'execution-lock.json')
    if not (folder/'trace.npz').exists():return None,dict(complete=False,valid=False,reason=d.get('failure','No saved trace'),spec=d['spec'])
    assert sha(folder/'trace.npz')==d['trace_sha256']
    z=dict(np.load(folder/'trace.npz',allow_pickle=False))
    finite=all(np.isfinite(v).all() for v in z.values() if np.issubdtype(v.dtype,np.number))
    if not d['complete'] or not finite:
        return z,dict(complete=False,valid=False,reason=d.get('failure','Nonfinite saved values'),spec=d['spec'],
            trace_sha256=d['trace_sha256'],saved_state_samples=len(z['time_s']),saved_duration_s=float(z['time_s'][-1]))
    plan=read(ROOT/'connection-plan.json');t=z['time_s'];late=t>=4-1e-8
    required=dict(model='prototype',replay=None,gap_nS=.1,feedback_gain=100,feedback_clip=1,duration=8.,seed=13,avb_on=True,calcium_scale=1,perturb=False,nmj_nS=.9)
    assert all(d['spec'][k]==v for k,v in required.items())
    assert z['neuron_names'].tolist()==plan['neurons'] and z['neuron_voltage_mV'].shape==(801,302)
    assert z['muscle_calcium_mM'].shape==(801,95) and z['drive'].shape==(801,48,2)
    assert z['feedback_pA'].shape==(8001,18) and len(z['B_names'])==18
    assert (d['neurons'],d['muscles'],d['chemical_connections'],d['gap_pairs'])==(302,95,4665,1192)
    assert abs(t-np.arange(801)*.01).max()<1e-8
    assert np.array_equal(z['exchange_time_s'],np.arange(8001)*.001)
    assert (z['external_drive']==0).all() and d['max_gap_pair_current_imbalance_nA']<1e-12
    assert z['neuron_voltage_mV'].min()>=-100 and z['neuron_voltage_mV'].max()<=60
    ix=[plan['neurons'].index(n) for n in z['B_names']];z['B']=z['neuron_voltage_mV'][:,ix];v=z['B'][late]
    strain=(z['side_length']-z['side_length'][0])/z['side_length'][0];z['strain']=strain
    sensed=((strain[:,:,1]-strain[:,:,0])/2)@z['feedback_weights'].T*np.where(z['feedback_sides']==0,1.,-1.)
    assert abs(sensed-z['sensed_strain']).max()<1e-12
    expected=np.clip(100*sensed,-1,1) if d['spec']['feedback']=='live' else np.zeros_like(sensed)
    assert abs(expected-z['feedback_pA'][::10]).max()<1e-12
    z['body_frame_y_um']=helper().shape(z);z['late']=late
    pp=[find_peaks(v[:,j],prominence=3)[0] for j in range(18)]
    frequency=[float(1/np.diff(t[late][p]).mean()) if len(p)>1 else None for p in pp]
    rhythm=all(len(p)>=5 for p in pp)
    displacement=float(np.linalg.norm(z['pose'][-1,:,:2].mean(0)-z['pose'][400,:,:2].mean(0))*1e6)
    speed=float(z['speed'][late].mean()*1e6)
    assert abs(speed-d['late_mean_speed_um_s'])<1e-10 and abs(displacement-d['late_centroid_displacement_um'])<1e-10
    m=dict(complete=True,valid=rhythm,reason=None if rhythm else 'Inherited minimum 5 prominent late peaks not retained in all 18 B cells',
        spec=d['spec'],dt_ms=d['spec']['dt_ms'],feedback=d['spec']['feedback'],B_names=z['B_names'].tolist(),
        B_peak_counts=[len(p) for p in pp],B_frequency_Hz=frequency,
        B_amplitude_mV=(np.percentile(v,95,axis=0)-np.percentile(v,5,axis=0)).tolist(),
        B_mean_mV=v.mean(0).tolist(),B_std_mV=v.std(0).tolist(),
        calcium_mean_mM=z['muscle_calcium_mM'][late].mean(0).tolist(),calcium_std_mM=z['muscle_calcium_mM'][late].std(0).tolist(),
        calcium_max_mM=float(z['muscle_calcium_mM'][late].max()),drive_mean=float(z['drive'][late].mean()),
        drive_mean_channel_std=float(z['drive'][late].std(0).mean()),drive_max=float(z['drive'][late].max()),
        body_frame_lateral_amplitude_um=float(np.ptp(z['body_frame_y_um'][late],axis=0).max()),
        laboratory_lateral_amplitude_um=float(np.ptp(z['pose'][late,:,1],axis=0).max()*1e6),side_strain_RMS=rms(strain[late]),
        late_speed_um_s=speed,late_speed_RMS_um_s=rms(z['speed'][late])*1e6,late_displacement_um=displacement,
        feedback_max_pA=float(abs(z['feedback_pA']).max()),trace_sha256=d['trace_sha256'],wall_s=d['wall_s'],
        propulsion_gate_passed=bool(speed>=10 and displacement>=20))
    return z,m


def main():
    p=read(ROOT/'protocol.json');assert sha(ROOT/'protocol.json')==read(ROOT/'protocol-lock.json')['protocol_sha256']
    assert p['acceptance']==read(ROOT/'parent-evidence/protocol.json')['acceptance']
    assert p['definitions']==read(ROOT/'parent-evidence/protocol.json')['definitions']
    for name,d in read(ROOT/'sources.json').items():assert sha(ROOT/name)==d['sha256']
    (ROOT/'differences').mkdir(exist_ok=True);m=helper();metrics={};traces={};historical={};old_traces={}
    for mode in ['off','live']:
        for dt in [.025,.0125,.00625]:
            name=f'{mode}_dt{dt:g}'
            if (ROOT/name/'record.json').exists():traces[name],metrics[name]=load(ROOT/name)
            old_traces[name],historical[name]=load(ROOT/'historical'/name,new=False)
    comparisons={}
    for mode in ['off','live']:
        for co,fi in [(.025,.0125),(.0125,.00625),(.025,.00625)]:
            a=f'{mode}_dt{co:g}';b=f'{mode}_dt{fi:g}';key=f'{mode}_{co:g}_vs_{fi:g}'
            if a in metrics and b in metrics:
                if metrics[a]['valid'] and metrics[b]['valid']:
                    comparisons[key]=m.compare(traces[a],traces[b],metrics[a],metrics[b],key)
                    comparisons[key]['comparable']=True
                else:comparisons[key]=dict(passed=False,comparable=False,feedback=mode,coarse_dt_ms=co,reference_dt_ms=fi,
                    gates={},errors={},details={},failed_gates=['trace_or_rhythm_validity'],
                    reason='No numerical pass can be inferred from incomplete data or loss of the inherited rhythm validity requirement')
    first=[f'{mode}_0.025_vs_0.0125' for mode in ['off','live']]
    initial_ready=all(k in comparisons for k in first)
    fine_required=any(not comparisons[k]['passed'] for k in first) if initial_ready else None
    if initial_ready:
        decision=dict(protocol_sha256=sha(ROOT/'protocol.json'),fine_required=fine_required,
            source_comparisons=first,failed_gates={k:comparisons[k]['failed_gates'] for k in first},new_full_conditions_observed=len(metrics))
        decision_path=ROOT/'refinement-decision.json'
        if decision_path.exists():
            old=read(decision_path);assert old['fine_required']==fine_required and old['failed_gates']==decision['failed_gates']
        else:write(decision_path,decision)
    latest=(.0125,.00625) if any(k.endswith('dt0.00625') for k in metrics) else (.025,.0125)
    final_keys=[f'{mode}_{latest[0]:g}_vs_{latest[1]:g}' for mode in ['off','live']]
    final_ready=all(k in comparisons for k in final_keys)
    accepted=bool(final_ready and all(comparisons[k]['passed'] for k in final_keys) and (fine_required is False or len(metrics)==6))
    changes={};ratios={};parent=read(ROOT/'parent-evidence/summary.json')
    for name,new in metrics.items():
        if not new['complete']:continue
        a=traces[name];b=old_traces[name];mask=a['late']
        assert {k:v for k,v in new['spec'].items() if k!='name'}=={k:v for k,v in historical[name]['spec'].items() if k!='name'}
        changes[name]=dict(B_voltage_RMS_mV=rms(a['B'][mask]-b['B'][mask]),
            all302_voltage_RMS_mV=rms(a['neuron_voltage_mV'][mask]-b['neuron_voltage_mV'][mask]),
            muscle_calcium_RMS_mM=rms(a['muscle_calcium_mM'][mask]-b['muscle_calcium_mM'][mask]),
            muscle_drive_RMS=rms(a['drive'][mask]-b['drive'][mask]),body_frame_shape_RMS_um=rms(a['body_frame_y_um'][mask]-b['body_frame_y_um'][mask]),
            speed_mean_change_um_s=new['late_speed_um_s']-historical[name]['late_speed_um_s'],
            displacement_change_um=new['late_displacement_um']-historical[name]['late_displacement_um'],
            scope='Same-parameter candidate minus old IMPLEMENTATION, not candidate timestep error; both generic-neuron and muscle density mechanisms changed')
    for key,c in comparisons.items():
        if c['comparable']:
            old=parent['comparisons'][key]
            ratios[key]={gate:dict(old_error=old['errors'][gate]['value'],candidate_error=value['value'],
                ratio=value['value']/old['errors'][gate]['value'] if old['errors'][gate]['value']>1e-15 else None,
                old_passed=old['gates'][gate],candidate_passed=c['gates'][gate]) for gate,value in c['errors'].items()}
    domains=dict(B_rhythm=['B_frequency','B_amplitude','B_mean'],B_voltage_phase=['B_trajectory','B_peak_phase'],
        muscle=['calcium_statistics','calcium_trajectory','drive_statistics','drive_trajectory'],
        body_shape=['body_shape_trajectory','body_shape_amplitude'],side_strain=['side_strain_trajectory'],
        near_zero_motion=['mean_speed','speed_trajectory','centroid_displacement'],feedback=['feedback_trajectory'])
    domain_pass={mode:{group:bool(comparisons[key]['comparable'] and all(comparisons[key]['gates'][g] for g in gates)) for group,gates in domains.items()}
        for mode,key in zip(['off','live'],final_keys) if key in comparisons}
    result=dict(protocol_sha256=sha(ROOT/'protocol.json'),candidate='joint_exp_direct',new_full_runs=len(metrics),
        complete_new_runs=sum(x['complete'] for x in metrics.values()),failed_conditions={k:d.get('reason') for k,d in metrics.items() if not d['valid']},
        checks=dict(acceptance_and_definitions_unchanged=True,all_saved_trace_and_source_hashes_verified=True,generated_and_zero_step_preflight_passed=True),
        metrics=metrics,comparisons=comparisons,historical_metrics=historical,historical_model_changes=changes,
        numerical_error_migration_ratios=ratios,latest_pair_domains_passed=domain_pass,
        fine_required=fine_required,latest_pair_ms=latest,latest_pair_passed=bool(final_ready and all(comparisons[k]['passed'] for k in final_keys)),
        neural_step_convergence_accepted=accepted,
        final_budget_complete=bool(initial_ready and (fine_required is False or len(metrics)==6)),
        scope='284 generic C1 neurons AND95 muscles; 18 B point prototypes unchanged; body/exchange1ms unchanged',
        propulsion_gate_passed={k:d.get('propulsion_gate_passed',False) for k,d in metrics.items()},
        no_next_stage_or_extra_repairs_authorized=True)
    write(ROOT/'summary.json',result)
    print(json.dumps(dict(runs=len(metrics),fine_required=fine_required,convergence_accepted=accepted,
        failed_conditions=result['failed_conditions'],failed={k:d['failed_gates'] for k,d in comparisons.items()}),ensure_ascii=False))


if __name__=='__main__':main()
