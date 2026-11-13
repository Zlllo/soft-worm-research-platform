"""Analyze V2 raw controls offline; never loads a scientific runtime."""
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[3]/'results/c1-current-timing-2026-11-10'
VARIANTS=['euler_cached','exp_cached','euler_direct','exp_direct']
EXPECTED='8b9d11bda03f952c96d64d5b2b58f92496ecdcc008e5c471a51463664320f12a'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def longest(mask):
    best=run=0
    for value in mask:
        run=run+1 if value else 0;best=max(best,run)
    return best
def load(kind,v,voltage):
    folder=ROOT/'controls'/f'{kind}_{v}_v{voltage:g}';d=read(folder/'record.json')
    assert d['complete'] and sha(folder/'trace.npz')==d['trace_sha256']
    assert d['spec']['protocol_sha256']==EXPECTED and d['spec']['execution_lock_sha256']==sha(ROOT/'control-execution-lock.json')
    with np.load(folder/'trace.npz') as z:
        names=z['columns'].tolist();data=z['data'].copy();phases=z['phase'].copy();steps=z['step'].copy()
    assert np.isfinite(data).all()
    return dict(zip(names,data.T)),phases,steps,d
def formula_tolerance(values,p):return np.maximum(p['formula_current_abs_mA_cm2'],p['formula_current_relative']*np.abs(values))


def main():
    assert sha(ROOT/'protocol.json')==EXPECTED==read(ROOT/'protocol-lock.json')['protocol_sha256']
    p=read(ROOT/'protocol.json');tol=p['control_tolerances'];generated=read(ROOT/'generated-code-validation-v2.json');assert generated['passed']
    # C1 channels own internal reversals, not ca_ion's Nernst readout.
    internal_reversals={}
    for mech,key in [('ca_boyle','eca'),('k_fast','ek'),('k_slow','ek')]:
        text=(ROOT/'frozen/original'/f'{mech}.mod').read_text()
        internal_reversals[mech]=float(re.search(r'\b'+key+r'\s*=\s*([-+\d.Ee]+)',text).group(1))
    zero={};short={};natural={};success=True
    invariant_names=['time_ms','voltage_mV','concentration_mM','cai_mM','h_q','h_fcond',
        'e','f','p','q','n','cache_e','cache_f','cache_p','cache_q','cache_n']
    for v in VARIANTS:
        for voltage in [-20,-10]:
            x,phase,step,d=load('zero',v,voltage);assert phase.tolist()==['native_finitialize_return','algebra_before_explicit_fcurrent','algebra_after_explicit_fcurrent']
            assert d['executed_fadvance']==0 and d['explicit_fcurrent_calls']==1 and d['physical_observer_hooks']==0
            differences={name:float(abs(x[name][2]-x[name][1])) for name in invariant_names}
            inv_gates={name: value <= (0 if name=='time_ms' else tol['zero_step_concentration_mM'] if name in ['concentration_mM','cai_mM'] else tol['zero_step_voltage_mV'] if name=='voltage_mV' else tol['zero_step_STATE_and_cache_abs']) for name,value in differences.items()}
            m=d['metadata'];g=m['gmax_S_cm2'];powers={n:q['power'] for n,q in m['gate_parameters'].items()}
            stateCa=x['e'][2]**powers['e']*x['f'][2]**powers['f']*x['h_fcond'][2]
            cachedCa=x['cache_e'][2]*x['cache_f'][2]*x['h_fcond'][2]
            stateK=x['p'][2]**powers['p']*x['q'][2]**powers['q'];cachedK=x['cache_p'][2]*x['cache_q'][2]
            stateSlow=x['n'][2]**powers['n'];cachedSlow=x['cache_n'][2]
            result={};all_good=all(inv_gates.values())
            for mech,label,statefactor,cachefactor in [('ca_boyle','ca',stateCa,cachedCa),('k_fast','fast',stateK,cachedK),('k_slow','slow',stateSlow,cachedSlow)]:
                driving=x['voltage_mV'][2]-internal_reversals[mech]
                direct=g[mech]*statefactor*driving;cached=g[mech]*cachefactor*driving
                actual=float(x[f'I_{label}_mA_cm2'][2]);limit=float(formula_tolerance(actual,tol))
                selected=direct if v.endswith('direct') else cached
                alt=cached if v.endswith('direct') else direct
                matched=abs(actual-selected)<=limit;distinct=abs(selected-alt)>5*limit
                effective_reversal=float(x['voltage_mV'][2]-actual/x[f'gion_{label}_S_cm2'][2])
                result[label]=dict(actual_mA_cm2=actual,from_STATE_mA_cm2=float(direct),from_cache_mA_cm2=float(cached),
                    selected_formula_abs_error_mA_cm2=float(abs(actual-selected)),tolerance_mA_cm2=limit,
                    selected_formula_passed=bool(matched),alternative_distinguishable=bool(distinct),
                    observed_effective_channel_reversal_mV=effective_reversal,
                    source_internal_reversal_mV=internal_reversals[mech])
                all_good &= bool(matched and distinct and abs(effective_reversal-internal_reversals[mech])<1e-9)
            ion_sum_errors=dict(ca=float(abs(x['sum_ica_mA_cm2'][2]-x['I_ca_mA_cm2'][2])),
                k=float(abs(x['sum_ik_mA_cm2'][2]-x['I_fast_mA_cm2'][2]-x['I_slow_mA_cm2'][2])))
            all_good &= max(ion_sum_errors.values())<=tol['formula_current_abs_mA_cm2']
            key=f'{v}_v{voltage:g}'
            zero[key]=dict(passed=bool(all_good),invariant_abs_changes=differences,invariant_gates=inv_gates,
                currents=result,ionic_sum_abs_errors=ion_sum_errors,
                assigned_current_changes={n:float(x[n][2]-x[n][1]) for n in ['I_ca_mA_cm2','I_fast_mA_cm2','I_slow_mA_cm2','sum_ica_mA_cm2','sum_ik_mA_cm2']},
                trace_sha256=d['trace_sha256'])
            natural[key]=dict(voltage_mV=float(x['voltage_mV'][0]),STATE=[float(x[n][0]) for n in 'efpqn'],
                caches=[float(x['cache_'+n][0]) for n in 'efpqn'],concentration_mM=float(x['concentration_mM'][0]),cai_mM=float(x['cai_mM'][0]),
                currents_mA_cm2={n:float(x['I_'+n+'_mA_cm2'][0]) for n in ['ca','fast','slow']})
            success &= all_good
        x,phase,step,d=load('short',v,-10);assert d['executed_fadvance']==200 and d['explicit_fcurrent_calls']==0 and d['physical_observer_hooks']==0
        before=phase=='before_fadvance';after=phase=='after_fadvance';indices=np.arange(1,201)
        assert np.array_equal(step[before],indices) and np.array_equal(step[after],indices)
        m=d['metadata'];params=m['gate_parameters'];tau=params['p']['tau_ms'];dt=.025
        pinf=params['p']['rate']/(1+np.exp(-(-10-params['p']['midpoint_mV'])/params['p']['scale_mV']))
        qinf=params['q']['rate']/(1+np.exp(-(-10-params['q']['midpoint_mV'])/params['q']['scale_mV']))
        time=indices*dt
        ref=pinf+(.7-pinf)*((1-dt/tau)**indices if v.startswith('euler') else np.exp(-time/tau))
        p_error=float(abs(x['p'][after]-ref).max());q_error=float(abs(x['q'][after]-qinf).max())
        voltage_error=float(max(abs(x['voltage_mV'][after]+10).max(),abs(x['voltage_mV'][before]+10).max()))
        g=m['gmax_S_cm2']['k_fast'];rev=internal_reversals['k_fast'];drive=x['voltage_mV'][before]-rev
        stateI=g*x['p'][before]**params['p']['power']*x['q'][before]**params['q']['power']*drive
        cacheI=g*x['cache_p'][before]*x['cache_q'][before]*drive
        returnedI=g*x['p'][after]**params['p']['power']*x['q'][after]**params['q']['power']*(x['voltage_mV'][after]-rev)
        actual=x['I_fast_mA_cm2'][after];limit=formula_tolerance(actual,tol)
        selected=stateI if v.endswith('direct') else cacheI
        current_match=abs(actual-selected)<=limit
        informative=abs(stateI-cacheI)>5*limit
        long=longest((informative&current_match)[1:])
        # Predicted extra STATE generation, independently of post-step mismatch.
        p_history=np.r_[.7,ref];which=indices-1 if v.endswith('direct') else np.maximum(indices-2,0)
        analytic_current=g*p_history[which]**params['p']['power']*qinf*drive
        analytic_error=float(abs(actual-analytic_current).max())
        analytic_pass=bool(np.all(abs(actual-analytic_current)<=limit))
        herror=float(abs(x['h_fcond']-(1+(x['h_q']-1)*m['h_alpha'])).max())
        good=bool(p_error<=tol['short_gate_discrete_recurrence_max_abs'] and q_error<=tol['short_constant_q_max_abs'] and
            voltage_error<=tol['voltage_hold_mV'] and current_match.all() and analytic_pass and long>=5 and herror<=1e-12)
        short[v]=dict(passed=good,p_reference_max_error=p_error,q_constant_max_error=q_error,voltage_max_error_mV=voltage_error,
            current_selected_formula_max_error_mA_cm2=float(abs(actual-selected).max()),
            current_discrete_generation_max_error_mA_cm2=analytic_error,current_discrete_generation_passed=analytic_pass,
            first_step_hypothesis_difference_mA_cm2=float(abs(stateI[0]-cacheI[0])),
            first_step_aligned_formulas_equal=bool(abs(stateI[0]-cacheI[0])<=limit[0]),
            informative_steps_after_first=int(informative[1:].sum()),longest_consecutive_informative_matching_steps_after_first=long,
            post_endpoint_formula_max_error_mA_cm2=float(abs(actual-returnedI).max()),
            post_endpoint_formula_matching_step_count=int((abs(actual-returnedI)<=limit).sum()),
            h_algebra_max_error=herror,trace_sha256=d['trace_sha256'],
            first_four_steps=[dict(step=int(i+1),time_ms=float(time[i]),p_after=float(x['p'][after][i]),
                actual_I_mA_cm2=float(actual[i]),normal_pre_STATE_I=float(stateI[i]),cache_I=float(cacheI[i]),returned_STATE_I=float(returnedI[i])) for i in range(4)])
        success &= good
    init_effects={}
    for mode in ['euler','exp']:
        c=natural[f'{mode}_cached_v-10'];v=natural[f'{mode}_direct_v-10']
        init_effects[mode]=dict(STATE_initial_max_diff=float(np.max(abs(np.array(c['STATE'])-v['STATE']))),
            initial_current_direct_minus_cache_mA_cm2={n:v['currents_mA_cm2'][n]-c['currents_mA_cm2'][n] for n in ['ca','fast','slow']})
    clarification=dict(raw_field='record.metadata.reversal_mV.ca',raw_value=1000000.0,
        actual_recorded_quantity='seg.eca: ca_ion Nernst readout at zero intracellular concentration',
        not_the_internal_C1_channel_reversal=True,
        channel_internal_reversal_mV=internal_reversals,
        verification='Frozen MOD INITIAL / generated channel eca storage plus measured V-I/gion in every zero-step condition',
        raw_records_modified=False,new_scientific_runs_for_clarification=0)
    write(ROOT/'control-readout-clarification.json',clarification)
    summary=dict(protocol_sha256=EXPECTED,stage='v2_controls_complete_main_not_started',
        controls_completed=12,zero_step=8,short_controls=4,total_integrated_ms=20,explicit_fcurrent_calls=8,
        new_main_runs=0,new_full302_body=0,observer_conditions=0,retries=0,
        generated_code_passed=True,all_control_gates_passed=bool(success),
        zero_step_results=zero,short_K_results=short,natural_initialization=natural,native_initial_current_factor_effects=init_effects,
        readout_clarification=clarification,
        conclusion='Actual current readout depends on cached vs current available STATE powers; short aligned controls distinguish one extra cached generation from normal current-before-state splitting. Does not quantify startup share or fix h/CaPool/NMJ/full-system timing.',
        main_execution_authorized_by_this_result=False,remaining='Mainline must review actual controls before authorizing six new main conditions')
    write(ROOT/'control-summary.json',summary)
    print(json.dumps(dict(all_control_gates_passed=bool(success),zero=8,short=4,new_main=0,
        failed_zero=[k for k,d in zero.items() if not d['passed']],failed_short=[k for k,d in short.items() if not d['passed']])))


if __name__=='__main__':main()
