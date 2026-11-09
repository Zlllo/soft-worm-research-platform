"""Recompute isolated-control and muscle conclusions from saved data only."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[3]/'results/c1-muscle-integration-isolation-2026-11-09'
VARIANTS=['original','gates','pool','both'];STEPS=[.025,.0125,.00625]
GATES=[('ca_boyle','e_q','e'),('ca_boyle','f_q','f'),('k_fast','p_q','p'),('k_fast','q_q','q'),('k_slow','n_q','n')]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def number(text,n):return float(re.search(r'\b'+n+r'\s*=\s*([-+0-9.Ee]+)',text).group(1))
def rms(a,axis=None):return np.sqrt(np.mean(a*a,axis=axis))
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def load(case,v,dt):
    p=ROOT/'runs'/f'{case}_{v}_dt{dt:g}';d=read(p/'record.json')
    assert d['complete'] and d['trace_sha256']==sha(p/'trace.npz')
    assert d['spec']['protocol_sha256']==sha(ROOT/'protocol.json')
    with np.load(p/'trace.npz') as z:
        a=z['data'].copy();columns=z['columns'].tolist()
    x={n:a[:,i] for i,n in enumerate(columns)}
    assert np.isfinite(a).all() and d['rows']==len(a)
    assert abs(d['actual_dt_ms']-dt)<1e-15
    assert np.array_equal(x['time_ms'],np.arange(len(a))*d['sample_ms'])
    return x,d


def caches(x):
    state=np.array([x[n] for _,n,_ in GATES]).T
    previous=np.array([x['previous_'+p] for _,_,p in GATES]).T
    cached=np.array([x['cache_'+p] for _,_,p in GATES]).T
    powers=np.array([2,1,4,1,1])
    error_current=cached-state**powers;error_previous=cached-previous**powers
    return dict(max_abs_to_current_state_after_t0=float(abs(error_current[1:]).max()),
        RMS_to_current_state_after_t0=float(rms(error_current[1:])),
        max_abs_to_recorded_previous_neural_state=float(abs(error_previous[1:]).max()),
        cached_previous_state_identity_passed=bool(abs(error_previous[1:]).max()<1e-12),
        initialization_max_abs_cache_state_mismatch=float(abs(error_current[0]).max()))


def analytic_controls(protocol):
    output={};rho=number((ROOT/'frozen/CaPool.mod').read_text(),'rho');tau=number((ROOT/'frozen/CaPool.mod').read_text(),'decayConstant')
    for variant in ['original','both']:
        for dt in STEPS:
            x,d=load('gates',variant,dt);vtarget=protocol['controls']['gates']['voltage_mV'];t=x['time_ms']
            voltage=float(abs(x['voltage_mV']-vtarget).max());errors={};initialdist={}
            for mech,state,prefix in GATES:
                mod=(ROOT/'frozen'/f'{mech}.mod').read_text()
                tt=number(mod,prefix+'_timeCourse_tau');mid=number(mod,prefix+'_steadyState_midpoint');width=number(mod,prefix+'_steadyState_scale')
                inf=1/(1+np.exp(-(vtarget-mid)/width));initial=float(x[state][0]);initialdist[state]=abs(initial-inf)
                reference=inf+(initial-inf)*np.exp(-t/tt);errors[state]=float(abs(x[state]-reference).max())
                assert abs(d['gate_parameters'][f'{mech}.{state}']['tau_ms']-tt)<1e-12
                assert abs(d['gate_parameters'][f'{mech}.{state}']['inf']-inf)<1e-9
            assert min(initialdist.values())>=protocol['controls']['gates']['initial_distance_min']
            output[f'gates_{variant}_{dt:g}']=dict(max_abs_state_error=errors,initial_distance_to_inf=initialdist,
                voltage_max_abs_error_mV=voltage,passed=bool(max(errors.values())<=1e-9 and voltage<=1e-6),cache=caches(x))
            x,d=load('pool',variant,dt);t=x['time_ms'];initial=2e-7
            source=1e6*.002*rho;steady=source*tau
            c50=steady+(initial-steady)*np.exp(-50/tau)
            reference=np.where(t<=50,steady+(initial-steady)*np.exp(-t/tau),c50*np.exp(-(t-50)/tau))
            mask=t<=100;stress=t>100;zero=(t>=50)&(t<=100)
            previous=np.maximum(np.arange(len(t))-1,0)*dt
            intended=np.where(previous<50,-.002,np.where(previous<100,0.,.002))
            density_error=float(abs(x['ica_mA_cm2']-intended).max())
            assert np.max(abs(x['source_ica_mA_cm2']-intended))<1e-12
            area_cm2=d['area_um2']*1e-8
            actual_total=-x['ica_mA_cm2']*area_cm2*1e6
            total_error=float(abs(x['pool_iCa_nA'][1:]-actual_total[1:]).max())
            assert density_error<=1e-12 and total_error<=1e-12
            state_error=float(abs(x['concentration_mM'][mask]-reference[mask]).max())
            cai_error=float(abs(x['cai_mM'][mask]-reference[mask]).max())
            iz=int(round(50/dt));zero_ref=x['concentration_mM'][iz]*np.exp(-(t[zero]-50)/tau)
            output[f'pool_{variant}_{dt:g}']=dict(actual_ica_unique_mA_cm2=np.unique(x['ica_mA_cm2']).tolist(),
                area_um2=d['area_um2'],actual_iCa_nA_levels=np.unique(actual_total).tolist(),density_max_abs_error=density_error,
                total_current_max_abs_error_nA=total_error,STATE_max_abs_error_mM=state_error,cai_max_abs_error_mM=cai_error,
                zero_input_decay_max_abs_error_mM=float(abs(x['concentration_mM'][zero]-zero_ref).max()),
                positive_window_max_sync_error_mM=float(abs(x['cai_mM'][mask]-x['concentration_mM'][mask]).max()),
                STATE_min_mM=float(x['concentration_mM'].min()),cai_min_mM=float(x['cai_mM'].min()),
                outward_stress_max_sync_error_mM=float(abs(x['cai_mM'][stress]-x['concentration_mM'][stress]).max()),
                STATE_analytic_passed=bool(state_error<=1e-15),positive_window_cai_analytic_passed=bool(cai_error<=1e-15),
                STATE_nonnegative_passed=bool(x['concentration_mM'].min()>=0),
                stress_ion_sync_passed=bool(abs(x['cai_mM'][stress]-x['concentration_mM'][stress]).max()<=1e-15),
                source_and_units_passed=True)
    return output


def pair(a,b,protocol):
    late=a['time_ms']>=4000;gates={};errors={};p=protocol['acceptance']
    def add(name,value,limit):
        errors[name]=dict(value=float(value),limit=float(limit));gates[name]=bool(value<=limit+1e-12)
    av=a['voltage_mV'][late];bv=b['voltage_mV'][late]
    add('voltage_RMS_mV',rms(av-bv),p['voltage_RMS_mV'])
    add('voltage_mean_abs_mV',abs(av.mean()-bv.mean()),p['voltage_mean_abs_mV'])
    amp=lambda q:np.percentile(q,95)-np.percentile(q,5)
    add('voltage_amplitude_relative',abs(amp(av)-amp(bv))/max(amp(bv),p['voltage_amplitude_floor_mV']),p['voltage_amplitude_relative'])
    for field,label,floor in [('concentration_mM','STATE_calcium',1e-9),('cai_mM','cai',1e-9),('drive_cai','drive',.001)]:
        x=a[field][late];y=b[field][late];scale=max(amp(y),floor)
        add(label+'_scaled_RMS',rms(x-y)/scale,.05)
        stats=[abs(x.mean()-y.mean()),abs(x.std()-y.std()),abs(np.quantile(x,.05)-np.quantile(y,.05)),abs(np.quantile(x,.95)-np.quantile(y,.95))]
        add(label+'_scaled_stats',max(stats)/scale,.05)
    gate_err=[float(rms(a[state][late]-b[state][late])) for _,state,_ in GATES]
    add('gate_state_RMS',max(gate_err),p['gate_state_RMS'])
    return dict(passed=all(gates.values()),gates=gates,errors=errors,gate_RMS_per_state=gate_err,
        failed_gates=[k for k,v in gates.items() if not v])


def main():
    protocol=read(ROOT/'protocol.json');assert sha(ROOT/'protocol.json')==read(ROOT/'protocol-lock.json')['protocol_sha256']
    for path,d in read(ROOT/'sources.json').items():assert sha(ROOT/path)==d['sha256']
    assert sha(ROOT/'harness/integration_prepare_at_lock.py')==protocol['prepare_script_sha256']
    builds=read(ROOT/'build-evidence.json')
    for variant,d in builds.items():
        for path,digest in d['sha256'].items():assert sha(ROOT/'variants'/variant/path)==digest
    controls=analytic_controls(protocol);metrics={};comparisons={};input_hashes={};synapse_hashes={}
    assert sha(ROOT/'presynaptic-input.npz')==protocol['input_sha256']
    with np.load(ROOT/'presynaptic-input.npz') as z:input_reference=z['voltage_mV'][::4].copy();dense_input=z['voltage_mV'].copy()
    for variant in VARIANTS:
        data={}
        for dt in STEPS:
            x,d=load('main',variant,dt);data[dt]=x;name=f'{variant}_{dt:g}'
            assert len(x['time_ms'])==320001
            assert np.array_equal(x['peer_at_record_mV'],input_reference)
            ix=np.maximum(np.arange(len(input_reference))*4-round(dt/.00625),0)
            assert np.array_equal(x['last_applied_peer_mV'],dense_input[ix])
            for field,ca in [('drive_cai','cai_mM'),('drive_STATE','concentration_mM')]:
                val=np.maximum(x[ca],0);assert np.max(abs(x[field]-.3*val/(val+2.5e-7)))<1e-14
            input_hashes[name]=hashlib.sha256(x['peer_at_record_mV'].tobytes()).hexdigest()
            synapse_hashes[name]=hashlib.sha256(x['synapse_s'].tobytes()).hexdigest()
            late=x['time_ms']>=4000
            metrics[name]=dict(voltage_min_max_mV=[float(x['voltage_mV'][late].min()),float(x['voltage_mV'][late].max())],
                STATE_calcium_mean_mM=float(x['concentration_mM'][late].mean()),STATE_calcium_std_mM=float(x['concentration_mM'][late].std()),
                drive_mean=float(x['drive_cai'][late].mean()),drive_std=float(x['drive_cai'][late].std()),
                drive_max=float(x['drive_cai'][late].max()),synapse_s_min_max=[float(x['synapse_s'][late].min()),float(x['synapse_s'][late].max())],
                STATE_cai_sync_max_abs_mM=float(abs(x['concentration_mM']-x['cai_mM']).max()),
                STATE_min_mM=float(x['concentration_mM'].min()),cai_min_mM=float(x['cai_mM'].min()),
                cache=caches(x),trace_sha256=d['trace_sha256'],wall_s=d['wall_s'])
        for coarse,fine in [(.025,.0125),(.0125,.00625),(.025,.00625)]:
            comparisons[f'{variant}_{coarse:g}_vs_{fine:g}']=pair(data[coarse],data[fine],protocol)
    assert len(set(input_hashes.values()))==1
    for dt in STEPS:assert len({synapse_hashes[f'{v}_{dt:g}'] for v in VARIANTS})==1
    original=comparisons['original_0.0125_vs_0.00625'];combined=comparisons['both_0.0125_vs_0.00625']
    improvement={}
    for field in ['voltage_RMS_mV','STATE_calcium_scaled_RMS','cai_scaled_RMS','drive_scaled_RMS']:
        old=original['errors'][field]['value'];new=combined['errors'][field]['value'];limit=original['errors'][field]['limit']
        improvement[field]=dict(original_error=old,combined_error=new,ratio=new/old if old>1e-30 else None,
            strong_improvement=bool(new<=.5*old or old<=.1*limit))
    analytic_passed=all(v.get('passed',v.get('STATE_analytic_passed',False)) for k,v in controls.items() if '_both_' in k)
    code_passed=all(d['generated_states'][mech]['exponential_generated']==((variant in ['gates','both'] and mech!='CaPool') or (variant in ['pool','both'] and mech=='CaPool'))
        for variant,d in builds.items() for mech in ['ca_boyle','k_fast','k_slow','CaPool'])
    recommend=bool(code_passed and analytic_passed and combined['passed'] and all(d['strong_improvement'] for d in improvement.values()))
    summary=dict(protocol_sha256=sha(ROOT/'protocol.json'),main_conditions=12,analytic_control_conditions=12,new_full302_or_body_runs=0,
        checks=dict(source_hashes_verified=True,actual_input_arrays_identical_at_all_common_points=True,
            actual_left_interval_input_matches_dense_file=True,same_dt_synaptic_state_bitwise_identical_across_variants=True,
            code_update_forms_passed=code_passed,combined_analytic_STATE_controls_passed=analytic_passed),
        analytic_controls=controls,main_metrics=metrics,paired_step_comparisons=comparisons,
        input_array_hashes=input_hashes,synapse_gate_array_hashes=synapse_hashes,combined_over_original_improvement=improvement,
        combined_main_last_pair_passed=combined['passed'],recommend_next_full_system_validation= recommend,
        unresolved=dict(cache_previous_state_identity_observed=all(m['cache']['cached_previous_state_identity_passed'] for m in metrics.values()),
            outward_stress_cai_sync_failed=any(not c.get('stress_ion_sync_passed',True) for c in controls.values()),
            shared_C1_neuron_effects_not_tested=True,full_system_convergence_not_tested=True),
        implementation_notes=['Two initial observation setup failures before any fadvance preserved in setup-attempts; no scientific main reruns',
            'Pool generated negative coefficient uses -(1)/decayConstant; build checker spelling corrected without patch changes; locked preparer source preserved'])
    write(ROOT/'summary.json',summary)
    print(json.dumps(dict(code_passed=code_passed,analytic_passed=analytic_passed,main_conditions=12,
        combined_main_passed=combined['passed'],recommend_full_system= recommend,
        failed_last_pairs={v:comparisons[f'{v}_0.0125_vs_0.00625']['failed_gates'] for v in VARIANTS}),ensure_ascii=False))


if __name__=='__main__':main()
