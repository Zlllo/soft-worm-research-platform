"""Independently recompute saved isolated-control and muscle evidence, no simulator."""
import json
import math
from pathlib import Path
import numpy as np
from integration_review_source import ROOT, REPO, ISOLATION, FROZEN, sha, write, number

GATES=[('ca_boyle','e_q'),('ca_boyle','f_q'),('k_fast','p_q'),('k_fast','q_q'),('k_slow','n_q')]
def rms(x):return float(np.sqrt(np.mean(np.square(x))))
def load(folder,protocol):
    record=json.loads((folder/'record.json').read_text())
    assert record['complete'] and sha(folder/'trace.npz')==record['trace_sha256']
    assert record['spec']['protocol_sha256']==sha(ISOLATION/'protocol.json')
    assert record['spec']['runner_sha256']==sha(REPO/'analysis/neuromechanics/b_motor_validation/integration_run.py')
    assert record['geometry']==dict(L_um=20.,diam_um=5.,cm_uF_cm2=1.)
    assert record['pool_rho']==2.38919e-12 and record['pool_decay_ms']==11.5943 and record['pool_rest_mM']==0
    with np.load(folder/'trace.npz') as z:
        cols=z['columns'].tolist();a=z['data'].copy()
    assert np.isfinite(a).all() and a.shape[1]==len(cols) and a.shape[0]==record['rows']
    values={k:a[:,j] for j,k in enumerate(cols)}
    t=values['time_ms'];assert abs(t[-1]-record['duration_ms'])<1e-8
    assert np.max(abs(t-np.arange(len(t))*record['sample_ms']))<1e-8
    assert abs(record['actual_neural_clock_ms']-record['duration_ms'])<1e-5
    assert record['actual_dt_ms']==record['spec']['dt_ms']
    for src,derived in [('cai_mM','drive_cai'),('concentration_mM','drive_STATE')]:
        c=np.maximum(values[src],0);expected=.3*c/(c+2.5e-7)
        assert np.max(abs(expected-values[derived]))<1e-15
    return values,record

def gate_control(z,r,p):
    t=z['time_ms'];v=z['voltage_mV'];dt=r['spec']['dt_ms'];result={}
    assert np.max(abs(v-p['voltage_mV']))<=p['voltage_guard_mV']
    for j,(mech,state) in enumerate(GATES):
        text=(FROZEN/'kernels'/f'{mech}.mod').read_text();prefix=state[0]
        tau=number(text,prefix+'_timeCourse_tau');mid=number(text,prefix+'_steadyState_midpoint')
        scale=number(text,prefix+'_steadyState_scale');rate=number(text,prefix+'_steadyState_rate')
        inf=rate/(1+math.exp(-(p['voltage_mV']-mid)/scale))
        x0=z[state][0];assert x0==p['initial_states'][j] and abs(x0-inf)>=p['initial_distance_min']
        expected=inf+(x0-inf)*np.exp(-t/tau)
        euler=inf+(x0-inf)*np.exp(np.arange(len(t))*math.log1p(-dt/tau))
        exact_error=float(abs(z[state]-expected).max());euler_error=float(abs(z[state]-euler).max())
        assert abs(r['gate_parameters'][mech+'.'+state]['tau_ms']-tau)<1e-12
        assert abs(r['gate_parameters'][mech+'.'+state]['inf']-inf)<1e-9
        result[state]=dict(initial=x0,steady=inf,tau_ms=tau,analytic_max_error=exact_error,
                           Euler_closed_form_max_error=euler_error,analytic_pass=exact_error<=p['max_state_abs_error'])
    return dict(max_voltage_deviation_mV=float(abs(v-p['voltage_mV']).max()),states=result,
                analytic_pass=all(x['analytic_pass'] for x in result.values()))

def pool_control(z,r,p):
    t=z['time_ms'];dt=r['spec']['dt_ms'];tau=r['pool_decay_ms'];rho=r['pool_rho'];rest=r['pool_rest_mM']
    measured=z['ica_mA_cm2'];expected=np.where(t<=50+.000001,-.002,np.where(t<=100+.000001,0.,.002))
    density_error=float(abs(measured[1:]-expected[1:]).max())
    assert density_error<=p['source_density_abs_tolerance_mA_cm2']
    assert np.max(abs(measured[1:]-z['source_ica_mA_cm2'][1:]))<=1e-12
    total=-measured*r['area_um2']*1e-8*1e6
    current_error=float(abs(total[1:]-z['pool_iCa_nA'][1:]).max())
    assert current_error<=p['total_current_abs_tolerance_nA']
    assert z['concentration_mM'][0]==p['initial_concentration_mM']
    assert z['cai_mM'][0]==p['initial_concentration_mM']
    S=-1e6*rho*(-.002);c_inf=rest+tau*S;c0=z['concentration_mM'][0]
    ref=c_inf+(c0-c_inf)*np.exp(-np.minimum(t,50)/tau)
    c50=c_inf+(c0-c_inf)*math.exp(-50/tau)
    ref=np.where(t<=50,ref,rest+(c50-rest)*np.exp(-(t-50)/tau))
    valid=t<=100+1e-8;stress=t>100+1e-8
    state_error=float(abs(z['concentration_mM'][valid]-ref[valid]).max())
    ion_error=float(abs(z['cai_mM'][valid]-ref[valid]).max())
    sync=abs(z['concentration_mM']-z['cai_mM'])
    # One-step algebra from the saved state and actual current, not a new integration.
    ss=rest+tau*(-1e6*rho*measured[1:]);before=z['concentration_mM'][:-1]
    uncut=ss+(before-ss)*math.exp(-dt/tau)
    euler=before+dt*(-1e6*rho*measured[1:]-(before-rest)/tau)
    expected_step=uncut if r['spec']['variant']=='both' else euler
    recurrence_error=float(abs(z['concentration_mM'][1:]-np.maximum(expected_step,0)).max())
    assert recurrence_error<1e-20
    zero=(t>=50-1e-8)&(t<=100+1e-8);i50=round(50/dt)
    zero_ref=z['concentration_mM'][i50]*np.exp(-(t[zero]-50)/tau)
    return dict(measured_density_max_error_mA_cm2=density_error,total_inward_current_max_error_nA=current_error,
        initial_total_current_cache_error_nA=float(abs(total[0]-z['pool_iCa_nA'][0])),source_mM_per_ms=S,
        inward_steady_mM=c_inf,reference_state_max_error_mM=state_error,reference_cai_max_error_mM=ion_error,
        analytic_STATE_pass=state_error<=p['state_max_abs_error_mM'],reference_sync_max_error_mM=float(sync[valid].max()),
        zero_input_decay_max_error_mM=float(abs(z['concentration_mM'][zero]-zero_ref).max()),
        stress_sync_max_error_mM=float(sync[stress].max()),stress_min_STATE_mM=float(z['concentration_mM'][stress].min()),
        stress_min_cai_mM=float(z['cai_mM'][stress].min()),clipped_saved_state_recurrence_max_error_mM=recurrence_error,
        exposed_cai_vs_unclipped_step_max_error_mM=float(abs(z['cai_mM'][1:]-expected_step).max()))

def pair(a,b,p):
    assert np.array_equal(a['time_ms'],b['time_ms']);m=a['time_ms']>=4000-1e-8
    out={};passed={}
    x=a['voltage_mV'][m];y=b['voltage_mV'][m]
    amp=lambda v:float(np.percentile(v,95)-np.percentile(v,5))
    out['voltage_RMS_mV']=rms(x-y);out['voltage_mean_abs_mV']=float(abs(x.mean()-y.mean()))
    out['voltage_amplitude_relative']=abs(amp(x)-amp(y))/max(amp(y),p['voltage_amplitude_floor_mV'])
    for k in ['voltage_RMS_mV','voltage_mean_abs_mV','voltage_amplitude_relative']:passed[k]=out[k]<=p[k]
    for field,floor,tolerance in [('concentration_mM',p['calcium_scale_floor_mM'],p['calcium_scaled_stats_and_RMS']),
                                 ('cai_mM',p['calcium_scale_floor_mM'],p['calcium_scaled_stats_and_RMS']),
                                 ('drive_cai',p['drive_scale_floor'],p['drive_scaled_stats_and_RMS']),
                                 ('drive_STATE',p['drive_scale_floor'],p['drive_scaled_stats_and_RMS'])]:
        x=a[field][m];y=b[field][m];scale=max(amp(y),floor)
        stats=lambda v:np.array([v.mean(),v.std(),np.percentile(v,5),np.percentile(v,95)])
        out[field+'_scaled_RMS']=rms(x-y)/scale
        out[field+'_scaled_statistics_max']=float(abs(stats(x)-stats(y)).max()/scale)
        passed[field+'_trajectory']=out[field+'_scaled_RMS']<=tolerance
        passed[field+'_statistics']=out[field+'_scaled_statistics_max']<=tolerance
    out['gate_state_RMS_per_gate']={state:rms(a[state][m]-b[state][m]) for _,state in GATES}
    passed['gate_state_RMS']=max(out['gate_state_RMS_per_gate'].values())<=p['gate_state_RMS']
    return dict(metrics=out,gates=passed,all_pass=all(passed.values()))

def main():
    p=json.loads((ISOLATION/'protocol.json').read_text());result=dict(protocol_sha256=sha(ISOLATION/'protocol.json'),controls={},main_runs={},pairs={},issues=[],pending=[])
    data={};records={}
    for case in ['gates','pool','main']:
        for variant in (p['variants'] if case=='main' else p['controls']['variants']):
            for dt in p['dt_ms']:
                name=f'{case}_{variant}_dt{dt:g}';folder=ISOLATION/'runs'/name
                if not (folder/'record.json').exists():result['pending'].append(name);continue
                z,r=load(folder,p);data[name]=z;records[name]=r
                if case=='gates':result['controls'][name]=gate_control(z,r,p['controls']['gates'])
                elif case=='pool':result['controls'][name]=pool_control(z,r,p['controls']['pool'])
                else:
                    assert z['voltage_mV'][0]==-45 and z['concentration_mM'][0]==0 and z['cai_mM'][0]==0
                    with np.load(ISOLATION/'presynaptic-input.npz') as source:inp=source['voltage_mV']
                    idx=np.rint(z['time_ms']/.00625).astype(int);stride=round(dt/.00625)
                    assert np.array_equal(z['peer_at_record_mV'],inp[idx])
                    assert np.array_equal(z['last_applied_peer_mV'][1:],inp[idx[1:]-stride])
                    caches={}
                    for _,state in GATES:
                        letter=state[0];power=r['gate_parameters'][next(mech for mech,s in GATES if s==state)+'.'+state]['power']
                        caches[letter]=dict(previous_state_power_max_error=float(abs(z['cache_'+letter][1:]-z['previous_'+letter][1:]**power).max()),
                            current_state_power_RMS_difference=rms(z['cache_'+letter][1:]-z[state][1:]**power),
                            initial_cache_difference=float(z['cache_'+letter][0]-z[state][0]**power))
                        assert caches[letter]['previous_state_power_max_error']<1e-12
                    result['main_runs'][name]=dict(trace_sha256=r['trace_sha256'],runner_sha256=r['spec']['runner_sha256'],rows=r['rows'],
                        input_endpoint_and_previous_interval_exact=True,caches=caches,
                        max_cai_STATE_difference_mM=float(abs(z['cai_mM']-z['concentration_mM']).max()),
                        voltage_range_mV=[float(z['voltage_mV'].min()),float(z['voltage_mV'].max())])
    for dt in p['dt_ms']:
        names=[f'main_{v}_dt{dt:g}' for v in p['variants']]
        if all(n in data for n in names):
            base=data[names[0]]
            for name in names[1:]:
                for key in ['peer_at_record_mV','last_applied_peer_mV','synapse_s']:
                    assert np.array_equal(base[key],data[name][key]),(name,key)
    for variant in p['variants']:
        for coarse,fine in [(.025,.0125),(.0125,.00625),(.025,.00625)]:
            a=f'main_{variant}_dt{coarse:g}';b=f'main_{variant}_dt{fine:g}'
            if a in data and b in data:result['pairs'][f'{variant}_{coarse:g}_vs_{fine:g}']=pair(data[a],data[b],p['acceptance'])
    result['raw_hashes']={n:r['trace_sha256'] for n,r in records.items()}
    result['reviewed_runs']=len(records)
    write(ROOT/'saved-data-review.json',result)
    print(json.dumps(dict(reviewed=len(records),pending=len(result['pending']),pairs=len(result['pairs']),issues=result['issues'])))

if __name__=='__main__':main()
