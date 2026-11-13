"""Narrow v1->v2 protocol delta review only; preserve all previous reviews."""
import json
from pathlib import Path
from timing_review_static import ROOT, A, sha, write

V1='4603a5b323c58e71ee3061633bec1abf049e0610cc7696423ec32542b922fad4'
V2='8b9d11bda03f952c96d64d5b2b58f92496ecdcc008e5c471a51463664320f12a'
def read(p):return json.loads(p.read_text())

def main():
    protected=[ROOT/n for n in ['review.json','static-review.json','protocol-review.json','protocol_review.json']]
    before={str(p):sha(p) for p in protected}
    oldpath=A/'protocol-history/v1-unexecuted/protocol.json';newpath=A/'protocol.json'
    assert sha(oldpath)==V1 and sha(newpath)==V2
    assert sha(A/'protocol-history/v2-unexecuted/protocol.json')==V2
    old=read(oldpath);new=read(newpath);lock=read(A/'protocol-lock.json');preflight=read(A/'protocol-preflight.json')
    assert lock['protocol_sha256']==V2 and lock['supersedes_protocol_sha256']==V1
    assert new['protocol_version']==2 and new['supersedes_protocol_sha256']==V1
    assert lock['new_scientific_runs']==lock['new_compilations']==0
    assert preflight['protocol_sha256']==V2 and preflight['scientific_runs']==preflight['compilations']==0
    unchanged=['design','protected','acceptance','factorial_analysis','reuse','source_manifest_sha256','prepare_script_sha256',
               'scientific_execution_authorized','compile_or_new_scientific_run_performed']
    assert all(new[k]==old[k] for k in unchanged)
    main_added=set(new['main'])-set(old['main'])
    assert main_added=={'passive_extra_fields','initialization_limit','legacy_decimation_limit','non_density_current'}
    assert all(new['main'][k]==v for k,v in old['main'].items())
    gen_added=set(new['generated_code_gate'])-set(old['generated_code_gate'])
    assert gen_added=={'no_observer_in_main','NMJ','accessible_current_readouts'}
    assert all(new['generated_code_gate'][k]==v for k,v in old['generated_code_gate'].items())
    # Carry forward the prior source-only patch audit, checking the proposal stayed identical.
    prior=read(ROOT/'protocol_review.json');assert prior['reviewed_protocol_sha256']==V1 and prior['status']=='complete'
    assert sha(A/'reuse-audit.json')==new['reuse']['audit_sha256']==old['reuse']['audit_sha256']
    assert sha(A/'sources.json')==new['source_manifest_sha256']
    assert (ROOT/'A-protocol-snapshot/proposal-source-hashes.json').read_bytes()==(A/'proposal-source-hashes.json').read_bytes()
    proposal=read(A/'proposal-source-hashes.json')
    for rel,h in proposal.items():assert sha(A/rel)==h
    c=new['controls'];zero=c['zero_step'];short=c['short_K_fast'];b=new['budgets'];tol=new['control_tolerances']
    assert c['total_conditions']==12 and c['observer_instrumented_kernels']==c['modified_observer_current_hooks']==0
    assert zero['conditions']==8 and len(zero['kernels'])==4 and zero['voltage_mV']==[-20,-10]
    assert zero['duration_ms']==zero['fadvance_calls']==0 and zero['explicit_fcurrent_calls']==1 and zero['fresh_process_each']
    assert short['conditions']==4 and len(short['kernels'])==4 and short['dt_ms']==.025
    assert short['duration_ms']==5 and short['neural_steps']==200 and short['record_every_neural_step']
    assert short['NMJ_inserted'] is False and short['no_explicit_fcurrent'] and short['no_observer_hooks']
    assert b['new_main']==b['reused_main']==6 and b['small_controls']==12 and b['observer_conditions']==0
    assert b['new_full302']==b['new_body']==b['new_training']==b['new_parameter_search']==0
    assert b['explicit_fcurrent_calls']==8 and b['main_fcurrent_calls']==b['short_trajectory_fcurrent_calls']==0
    for key in ['formula_current_abs_mA_cm2','formula_current_relative','voltage_hold_mV']:
        assert tol[key]==old['control_tolerances'][key]
    assert tol['short_gate_discrete_recurrence_max_abs']==old['control_tolerances']['gate_recurrence_max_abs']
    changed={}
    for key in sorted(set(old)|set(new)):
        if old.get(key)!=new.get(key):
            changed[key]=[k for k in sorted(set(old[key])|set(new[key])) if old[key].get(k)!=new[key].get(k)] if isinstance(old.get(key),dict) and isinstance(new.get(key),dict) else 'changed_or_added'
    result=dict(status='complete',decision='v2_delta_and_minimal_control_design_acceptable',
        scope='narrow delta/protocol review; previous static/source/data audit not repeated',
        selected_protocol_version=2,selected_protocol_sha256=V2,superseded_protocol_sha256=V1,
        mainline_decision='v2 with8 zero-step +4 K-fast replaces prior16-control decision; this file does not merge or add the two plans',
        prior_review_preserved=True,changed_fields=changed,unchanged_sections_checked=unchanged,
        proposal_source_files_still_match_prior_audit=len(proposal),
        findings=[],
        minimal_control_review=dict(
            zero_step='4 kernels x2 voltages, intentionally distinct STATE/cache values and actual unchanged h_fcond: one active fcurrent then discard, no advance. This separates formula dependence and records assigned-buffer changes.',
            short_K_fast='4 kernels at one dt, K-fast only, p0=.7 and q=q_inf, initial cache powers aligned once. Every step recorded, no extra current query. Closed-form p makes normal p_(j-1) current and extra cached p_(j-2) distinguishable after the first equal step.',
            intervention_scope='control-only STATE/cache alignment and zeroing other currents never enter main. No observer kernels or callback hooks remain.',
            formula_units='per-channel outward current in mA/cm2; existing negative i__ exposures need sign reversal; do not use summed ik for fastK',
            h_boundary='retain actual h_fcond, no h_q substitution or h(cai)/rates refresh',
            inference='one selected dt establishes that dependency for the isolated control; it is not a complete latency profile or a main-trajectory attribution fraction'),
        unchanged_main=dict(new=6,reused=6,dt_ms=[.025,.0125,.00625],duration_s=8,window_s=[4,8],full_system_new=0),
        reuse_review_carried_forward=prior['historical_reuse'],
        wording_interpretations=[
            dict(field='main.start_condition',note='This inherited v1 sentence still mentions observer noninterference controls. Under authoritative v2 controls/budgets, observer count is zero: its intended pre-main gate is generated-code checks plus zero-step invariants and short-K algebra/phase checks. Do not implement observer conditions to satisfy this stale phrase.'),
            dict(field='budgets.small_control_duration_ms_each',note='The inherited5ms scalar describes the4 short controls; the detailed zero_step_duration_ms_each=0 governs the other8 controls. Total planned integrated control time is20ms, not60ms.')],
        pending_execution_checks=['actual new compiled current/state functions','zero-step invariants and active-query formula discrimination',
            'short-K fixed-voltage/STATE/current phase and five-consecutive-informative-step checks','six new main outcomes and factorial contrasts'],
        scientific_execution_authorized_by_this_review=False,new_simulations=0,new_compilations=0,
        revision_not_result_driven_evidence='mainline explicitly reports0 scientific/compile runs; v1/v2 unexecuted history and protocol-preflight agree')
    assert all(sha(Path(path))==h for path,h in before.items())
    result['preserved_previous_review_sha256']=before
    write(ROOT/'protocol-review-v2.json',result)
    print(json.dumps(dict(status=result['status'],sha256=V2,findings=0,old_reviews_unchanged=True,small_controls=12,new_main=6,new_simulations=0)))

if __name__=='__main__':main()
