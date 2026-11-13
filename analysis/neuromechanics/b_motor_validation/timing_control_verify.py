"""Independent raw-control audit; no import of analyzer, NEURON or its APIs."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]/'results/c1-current-timing-2026-11-10'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,atol=1e-16):np.testing.assert_allclose(a,b,rtol=1e-9,atol=atol)
def raw(kind,v,V):
    folder=ROOT/'controls'/f'{kind}_{v}_v{V:g}';r=read(folder/'record.json')
    assert r['complete'] and sha(folder/'trace.npz')==r['trace_sha256']
    assert r['spec']['execution_lock_sha256']==sha(ROOT/'control-execution-lock.json')
    with np.load(folder/'trace.npz') as z:
        columns=z['columns'].tolist();a=z['data'].copy();phase=z['phase'].copy();step=z['step'].copy()
    return {k:a[:,i] for i,k in enumerate(columns)},phase,step,r


def main():
    s=read(ROOT/'control-summary.json');p=read(ROOT/'protocol.json');t=p['control_tolerances']
    assert s['protocol_sha256']==sha(ROOT/'protocol.json')==read(ROOT/'control-execution-lock.json')['protocol_sha256']
    gates=[];controls=0;steps=0;currents_checked=0
    for v in ['euler_cached','exp_cached','euler_direct','exp_direct']:
        for V in [-20,-10]:
            a,ph,ix,r=raw('zero',v,V);m=r['metadata'];reported=s['zero_step_results'][f'{v}_v{V:g}']
            assert r['executed_fadvance']==0 and r['explicit_fcurrent_calls']==1 and r['physical_observer_hooks']==0
            own_inv=True
            for name in reported['invariant_abs_changes']:
                difference=abs(a[name][2]-a[name][1]);close(difference,reported['invariant_abs_changes'][name])
                limit=0 if name=='time_ms' else 1e-15 if name in ['cai_mM','concentration_mM'] else 1e-12
                own_inv &= difference<=limit
            own=True
            for mech,label,letters,E in [('ca_boyle','ca','ef',40),('k_fast','fast','pq',-60),('k_slow','slow','n',-60)]:
                state_factor=1.;cache_factor=1.
                for q in letters:
                    state_factor*=a[q][2]**m['gate_parameters'][q]['power'];cache_factor*=a['cache_'+q][2]
                if label=='ca':state_factor*=a['h_fcond'][2];cache_factor*=a['h_fcond'][2]
                coefficient=m['gmax_S_cm2'][mech]*(a['voltage_mV'][2]-E)
                predicted=coefficient*(state_factor if v.endswith('direct') else cache_factor)
                alternative=coefficient*(cache_factor if v.endswith('direct') else state_factor)
                I=a['I_'+label+'_mA_cm2'][2];limit=max(1e-12,1e-9*abs(I))
                err=abs(I-predicted);item=reported['currents'][label]
                close(err,item['selected_formula_abs_error_mA_cm2'])
                assert item['selected_formula_passed']==bool(err<=limit)
                assert item['alternative_distinguishable']==bool(abs(predicted-alternative)>5*limit)
                close(a['voltage_mV'][2]-I/a['gion_'+label+'_S_cm2'][2],E,atol=1e-10)
                own &= err<=limit and abs(predicted-alternative)>5*limit
                currents_checked+=1
            assert reported['passed']==bool(own and own_inv)
            gates.append(reported['passed']);controls+=1
        a,ph,ix,r=raw('short',v,-10);m=r['metadata'];item=s['short_K_results'][v]
        pre=ph=='before_fadvance';post=ph=='after_fadvance'
        assert len(a['p'][post])==200 and r['executed_fadvance']==200 and r['explicit_fcurrent_calls']==0
        q=m['gate_parameters']['q'];qinf=q['rate']/(1+math.exp(-(-10-q['midpoint_mV'])/q['scale_mV']))
        par=m['gate_parameters']['p'];pinf=par['rate']/(1+math.exp(-(-10-par['midpoint_mV'])/par['scale_mV']))
        decay=1-.025/par['tau_ms'] if v.startswith('euler') else math.exp(-.025/par['tau_ms'])
        history=[.7]
        for _ in range(200):history.append(pinf+(history[-1]-pinf)*decay)
        p_error=max(abs(a['p'][post]-np.array(history[1:])))
        # Independent iterative recurrence vs analyzer's closed form may differ
        # by a few ULP; the actual preregistered state gate stays at 1e-9.
        close(p_error,item['p_reference_max_error'],atol=20*np.finfo(float).eps)
        assert p_error<=1e-9 and max(abs(a['q'][post]-qinf))<=1e-9
        matched=[];informative=[];actual=a['I_fast_mA_cm2'][post]
        for j in range(200):
            voltage=a['voltage_mV'][pre][j];coef=m['gmax_S_cm2']['k_fast']*(voltage+60)
            now=coef*a['p'][pre][j]**4*a['q'][pre][j]
            cache=coef*a['cache_p'][pre][j]*a['cache_q'][pre][j]
            predicted=now if v.endswith('direct') else cache;limit=max(1e-12,1e-9*abs(actual[j]))
            matched.append(abs(actual[j]-predicted)<=limit);informative.append(abs(now-cache)>5*limit)
            old_index=j if v.endswith('direct') else max(j-1,0)
            closed_current=coef*history[old_index]**4*qinf
            assert abs(actual[j]-closed_current)<=limit
            currents_checked+=1
        run=longest=0
        for yes in [bool(a and b) for a,b in zip(matched[1:],informative[1:])]:
            run=run+1 if yes else 0;longest=max(longest,run)
        assert longest==item['longest_consecutive_informative_matching_steps_after_first']
        assert longest>=5 and all(matched)
        assert not informative[0] and item['first_step_aligned_formulas_equal']
        close(max(abs(a['h_fcond']-(1+(a['h_q']-1)*m['h_alpha']))),item['h_algebra_max_error'])
        assert item['passed'];gates.append(item['passed']);controls+=1;steps+=200
    assert controls==12 and steps==800 and len(list((ROOT/'controls').glob('*/record.json')))==12
    assert not (ROOT/'main').exists()
    if (ROOT/'main-runs').exists():
        # A later, separately approved main stage may be co-packaged. This
        # control audit still checks only the original 12 control conditions.
        main_lock=read(ROOT/'main-execution-lock.json')
        approval=read(ROOT/'approvals/main-run-authorization.json')
        assert main_lock['protocol_sha256']==s['protocol_sha256']==approval['protocol_sha256']
        assert main_lock['main_conditions_authorized']==approval['main_new_run_cap']==6
        assert len(list((ROOT/'main-runs').glob('*/record.json')))<=6
    assert s['all_control_gates_passed']==all(gates) and s['new_main_runs']==0 and not s['main_execution_authorized_by_this_result']
    result=dict(independent_control_audit_passed=True,control_conditions_checked=controls,neural_steps_checked=steps,
        separate_current_formula_samples_checked=currents_checked,explicit_fcurrent_calls=8,
        analyzer_imported=False,NEURON_imported=False,control_summary_sha256=sha(ROOT/'control-summary.json'),
        control_execution_lock_sha256=sha(ROOT/'control-execution-lock.json'),new_main_runs=0)
    (ROOT/'control-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
