"""Offline 2x2 outcome analysis; reuse byte-pinned old metric function."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]/'results/c1-current-timing-2026-11-10'
VARIANTS=['euler_cached','exp_cached','euler_direct','exp_direct'];STEPS=[.025,.0125,.00625]
FIELDS=['voltage_mV','e_q','f_q','p_q','q_q','n_q','concentration_mM','cai_mM','drive_cai']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def rms(a):return float(np.sqrt(np.mean(a*a)))
def stats(a):return dict(RMS=rms(a),signed_mean=float(a.mean()),max_abs=float(abs(a).max()))
def load(v,dt,extras=False):
    folder=ROOT/('reused-main' if v.endswith('cached') else 'main-runs')/f'{v}_dt{dt:g}'
    d=read(folder/'record.json');assert d['complete'] and sha(folder/'trace.npz')==d['trace_sha256']
    with np.load(folder/'trace.npz') as z:
        data=z['data'];x={n:data[:,i] for i,n in enumerate(z['columns'].tolist())}
        extra=None
        if extras:
            a=z['extras'];extra={n:a[:,i] for i,n in enumerate(z['extra_columns'].tolist())}
    assert len(data)==320001 and np.isfinite(data).all() and np.array_equal(x['time_ms'],np.arange(320001)*.025)
    return x,d,extra
def delta_summary(a,b,mask):return {field:stats(a[field][mask]-b[field][mask]) for field in FIELDS}


def main():
    protocol=read(ROOT/'protocol.json');lock=read(ROOT/'main-execution-lock.json')
    assert sha(ROOT/'protocol.json')==lock['protocol_sha256']
    assert sha(ROOT/'control-summary.json')==lock['control_summary_sha256']
    helper=ROOT/'legacy-code/integration_analyze.py'
    assert sha(helper)==read(ROOT/'sources.json')['legacy-code/integration_analyze.py']['sha256']
    spec=importlib.util.spec_from_file_location('frozen_previous_c1_analysis',helper)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    old_protocol=read(ROOT/'legacy/protocol.json')
    assert protocol['acceptance']==old_protocol['acceptance']
    input_sha={};syn_sha={};metrics={};pairs={};current_phase={};native={}
    with np.load(ROOT/'presynaptic-input.npz') as z:dense=z['voltage_mV'].copy()
    for v in VARIANTS:
        values={}
        for dt in STEPS:
            x,d,e=load(v,dt,not v.endswith('cached'));values[dt]=x;name=f'{v}_dt{dt:g}';late=x['time_ms']>=4000
            assert np.array_equal(x['peer_at_record_mV'],dense[::4])
            ix=np.maximum(np.arange(320001)*4-round(dt/.00625),0)
            assert np.array_equal(x['last_applied_peer_mV'],dense[ix])
            input_sha[name]=hashlib.sha256(x['peer_at_record_mV'].tobytes()).hexdigest()
            syn_sha[name]=hashlib.sha256(x['synapse_s'].tobytes()).hexdigest()
            for field,ca in [('drive_cai','cai_mM'),('drive_STATE','concentration_mM')]:
                q=np.maximum(x[ca],0);assert abs(x[field]-.3*q/(q+2.5e-7)).max()<1e-14
            metrics[name]=dict(voltage_range_mV=[float(x['voltage_mV'][late].min()),float(x['voltage_mV'][late].max())],
                voltage_std_mV=float(x['voltage_mV'][late].std()),STATE_calcium_mean_mM=float(x['concentration_mM'][late].mean()),
                STATE_calcium_std_mM=float(x['concentration_mM'][late].std()),drive_mean=float(x['drive_cai'][late].mean()),
                drive_std=float(x['drive_cai'][late].std()),drive_max=float(x['drive_cai'][late].max()),
                STATE_cai_sync_max_abs_mM=float(abs(x['concentration_mM']-x['cai_mM']).max()),
                STATE_min_mM=float(x['concentration_mM'].min()),cai_min_mM=float(x['cai_mM'].min()),
                trace_sha256=d['trace_sha256'],source='reused historical original' if v.endswith('cached') else 'new authorized direct',wall_s=d['wall_s'])
            native[name]=dict(V_mV=float(x['voltage_mV'][0]),STATE=[float(x[k][0]) for k in ['e_q','f_q','p_q','q_q','n_q']],
                caches=[float(x['cache_'+p][0]) for p in 'efpqn'],STATE_calcium_mM=float(x['concentration_mM'][0]),
                cai_mM=float(x['cai_mM'][0]),recorded_total_ica_mA_cm2=float(x['ica_mA_cm2'][0]),
                K_current_recorded_in_historical_trace=False if v.endswith('cached') else True)
            if e is not None:
                assert d['explicit_fcurrent_calls']==d['manual_rates_calls']==d['observer_hooks']==d['new_initialization_handlers']==0
                assert d['secondorder']==0 and d['cvode_active']==0 and d['actual_dt_ms']==dt
                assert np.isfinite(np.stack(list(e.values()))).all()
                assert max(abs(e['pre_last_time_ms'][1:]-(x['time_ms'][1:]-dt)))<1e-5
                errors={}
                for field,expr in [('ca',e['pre_last_e']**2*e['pre_last_f']*e['pre_last_h_fcond']),
                    ('fast',e['pre_last_p']**4*e['pre_last_q']),('slow',e['pre_last_n'])]:
                    mech={'ca':'ca_boyle','fast':'k_fast','slow':'k_slow'}[field]
                    rev=40 if field=='ca' else -60
                    predicted=d['gmax_S_cm2'][mech]*expr*(e['pre_last_voltage_mV']-rev)
                    actual=e['post_I_'+field+'_mA_cm2'];limit=np.maximum(1e-12,1e-9*abs(actual))
                    diff=abs(actual[1:]-predicted[1:])
                    errors[field]=dict(max_abs_error_mA_cm2=float(diff.max()),RMS_error_mA_cm2=rms(actual[1:]-predicted[1:]),
                        all_samples_within_control_formula_tolerance=bool(np.all(diff<=limit[1:])))
                current_phase[name]=dict(per_channel=errors,pre_last_snapshot_count=320000,
                    no_active_queries=True,scope='Stored post current agrees with last pre-advance V/gates/h; callback time not directly logged',
                    initial_I_mA_cm2={label:float(e['post_I_'+label+'_mA_cm2'][0]) for label in ['ca','fast','slow']})
            del e
        for co,fi in [(.025,.0125),(.0125,.00625),(.025,.00625)]:
            pairs[f'{v}_{co:g}_vs_{fi:g}']=legacy.pair(values[co],values[fi],old_protocol)
    assert len(set(input_sha.values()))==1
    for dt in STEPS:assert len({syn_sha[f'{v}_dt{dt:g}'] for v in VARIANTS})==1
    contrasts={};early={};interactions={}
    for dt in STEPS:
        x={v:load(v,dt)[0] for v in VARIANTS};mask=x['euler_cached']['time_ms']>=4000
        for a,b in [('euler_direct','euler_cached'),('exp_direct','exp_cached'),('exp_cached','euler_cached'),('exp_direct','euler_direct')]:
            key=f'{a}_minus_{b}_dt{dt:g}';contrasts[key]=delta_summary(x[a],x[b],mask)
            if a.endswith('direct') and b.endswith('cached'):
                early[key]={}
                for lo,hi in protocol['main']['initialization_report_windows_ms']:
                    times=x[a]['time_ms'];m=(times>=lo)&(times<=hi)
                    early[key][f'{lo:g}_to_{hi:g}_ms']=delta_summary(x[a],x[b],m)
        interactions[f'dt{dt:g}']={field:stats((x['exp_direct'][field][mask]-x['exp_cached'][field][mask])-
            (x['euler_direct'][field][mask]-x['euler_cached'][field][mask])) for field in FIELDS}
    improvement={}
    for integrator in ['euler','exp']:
        old=pairs[f'{integrator}_cached_0.0125_vs_0.00625'];new=pairs[f'{integrator}_direct_0.0125_vs_0.00625']
        results={}
        for field in ['voltage_RMS_mV','STATE_calcium_scaled_RMS','cai_scaled_RMS','drive_scaled_RMS']:
            a=old['errors'][field]['value'];b=new['errors'][field]['value'];tiny=a<=.1*old['errors'][field]['limit']
            results[field]=dict(cached_error=a,direct_error=b,direct_over_cached=b/a if a>1e-30 else None,
                cached_already_tiny=tiny,predeclared_improvement_requirement_passed=bool(b<=.5*a or tiny),actual_error_reduced=bool(b<a))
        improvement[integrator]=dict(fields=results,all_predeclared_improvement_requirements_passed=all(q['predeclared_improvement_requirement_passed'] for q in results.values()),
            direct_last_pair_all_gates_passed=new['passed'])
    init_diffs={}
    for integrator in ['euler','exp']:
        for dt in STEPS:
            c=native[f'{integrator}_cached_dt{dt:g}'];d=native[f'{integrator}_direct_dt{dt:g}']
            init_diffs[f'{integrator}_dt{dt:g}']=dict(STATE_initial_max_difference=float(abs(np.array(c['STATE'])-d['STATE']).max()),
                native_total_ica_direct_minus_cached_mA_cm2=d['recorded_total_ica_mA_cm2']-c['recorded_total_ica_mA_cm2'])
    result=dict(protocol_sha256=sha(ROOT/'protocol.json'),stage='six_new_main_complete',new_main_conditions=6,reused_main_conditions=6,
        new_control_conditions=12,new_full302_body_training_runs=0,
        checks=dict(all_raw_hashes_complete_and_finite=True,same_input_at_all_common_points=True,
            actual_left_interval_input_verified=True,same_dt_synapse_STATE_bitwise_identical_all_four=True,
            old_metric_function_reused_byte_pinned=True,main_no_extra_fcurrent_rates_observer=True),
        main_metrics=metrics,paired_step_comparisons=pairs,input_array_hashes=input_sha,synapse_gate_hashes=syn_sha,
        same_dt_factorial_contrasts=contrasts,same_dt_interactions=interactions,initialization_windows=early,
        natural_initial_states=native,native_readout_factor_effects=init_diffs,direct_passive_current_phase_checks=current_phase,
        improvement_by_integrator=improvement,
        limitations=['Historical cached secondorder=0 is source/default inference, not a logged historical value',
            'dt changes left-input hold interval; paired differences are not strict continuous-reference errors',
            'Startup-current and ongoing-readout contributions to late response are not uniquely decomposed',
            'Old fine-step records lack before-current V/cache/h; no invented current-call reconstruction for them',
            'h(cai), CaPool synchronization and NMJ current cache remain unchanged',
            'No full302/body/training run or full-system convergence claim'],
        no_more_conditions_authorized=True)
    write(ROOT/'summary.json',result)
    print(json.dumps(dict(new_main=6,reused_main=6,last_pair_failed={v:pairs[f'{v}_0.0125_vs_0.00625']['failed_gates'] for v in VARIANTS},
        improvement={k:d['all_predeclared_improvement_requirements_passed'] for k,d in improvement.items()},
        passive_current_formula_passed=all(d['per_channel'][m]['all_samples_within_control_formula_tolerance'] for d in current_phase.values() for m in ['ca','fast','slow']))))


if __name__=='__main__':main()
