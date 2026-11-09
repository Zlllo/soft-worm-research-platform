"""Independent source/unit algebra review. Never imports or launches NEURON."""
import hashlib
import json
import math
from pathlib import Path
import re

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'results/c1-muscle-integration-review-2026-11-09'
FROZEN=REPO/'results/b-motor-convergence-2026-11-09'
ISOLATION=REPO/'results/c1-muscle-integration-isolation-2026-11-09'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def block(text,pattern):
    m=re.search(pattern,text)
    if not m:raise ValueError(pattern)
    start=text.index('{',m.start());depth=0
    for i in range(start,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[start+1:i]
    raise ValueError('unclosed block')
def number(text,name):
    return float(re.search(r'\b'+re.escape(name)+r'\s*=\s*([-+\d.eE]+)',text).group(1))

def main():
    ROOT.mkdir(exist_ok=True)
    inputs=json.loads((FROZEN/'inputs.json').read_text())
    paths=[FROZEN/'kernels'/n for n in ['ca_boyle.mod','k_fast.mod','k_slow.mod','CaPool.mod','GenericMuscleCell.hoc','GenericNeuronCell.hoc','neuron_to_neuron_exc_syn.mod']]
    paths += [FROZEN/'kernels/arm64'/f'{n}.cpp' for n in ['ca_boyle','k_fast','k_slow','CaPool']]
    sources={}
    for p in paths:
        rel=str(p.relative_to(FROZEN));digest=sha(p)
        assert digest==inputs[rel]['sha256'],rel
        sources[str(p.relative_to(REPO))]=digest
    for name in ['docs/c1-muscle-integration-isolation-2026-11-09.md',
                 'results/b-motor-convergence-2026-11-09/numerical-diagnostic.md',
                 'results/b-motor-convergence-2026-11-09/numerical-diagnostic-evidence.json']:
        sources[name]=sha(REPO/name)
    gates=[]
    for mech,letters in [('ca_boyle',['e','f']),('k_fast',['p','q']),('k_slow',['n'])]:
        mod=(FROZEN/'kernels'/f'{mech}.mod').read_text()
        cpp=(FROZEN/'kernels/arm64'/f'{mech}.cpp').read_text()
        state=block(cpp,r'static int states\s*\([^;\n]*\)\s*\{')
        rates=block(cpp,r'static int\s+rates\s*\([^;\n]*\)\s*\{')
        current=block(cpp,r'static double _nrn_current\s*\([^;\n]*\)\s*\{')
        initial=block(mod,r'\bINITIAL\s*\{')
        for letter in letters:
            q=f'{letter}_q';tau=number(mod,f'{letter}_timeCourse_tau')
            assert re.search(r'\b'+q+r'\s*=\s*'+letter+r'_inf',initial)
            assert f'{q} = {q} - dt*(- ( rate_{q} ) )' in state
            assert f'{letter}_fcond = ' in rates
            assert f'{letter}_fcond' in current and f'{letter}_fcond =' not in current
            gates.append(dict(mechanism=mech,state=q,tau_ms=tau,default_initial='steady state, so analytic tests must override after initialization',
                frozen_state_method='explicit Euler confirmed from generated states()',
                cached_power_in_rates_before_state_update=True,current_reads_cached_power_without_refresh=True,
                closed_form='x_inf+(x0-x_inf)*exp(-t/tau)',
                scalar_relaxation=[dict(dt_ms=dt,dt_over_tau=dt/tau,
                    Euler_fraction=dt/tau,exact_fraction=-math.expm1(-dt/tau),
                    relative_increment_error=(dt/tau)/(-math.expm1(-dt/tau))-1) for dt in [.025,.0125,.00625]]))
    pool=(FROZEN/'kernels/CaPool.mod').read_text();cpp=(FROZEN/'kernels/arm64/CaPool.cpp').read_text()
    state=block(cpp,r'static int states\s*\([^;\n]*\)\s*\{')
    assert 'concentration = concentration - dt*(- ( rate_concentration ) )' in state
    assert state.index('concentration = concentration -')<state.index('cai = concentration')
    nrnstate=block(cpp,r'static void nrn_state\s*\([^;\n]*\)\s*\{')
    assert nrnstate.index('states();')<nrnstate.index('if ( concentration < 0.0 )')<nrnstate.index('_ion_cai = cai')
    for cell in ['GenericMuscleCell','GenericNeuronCell']:
        hoc=(FROZEN/'kernels'/f'{cell}.hoc').read_text()
        assert all(f'insert {mech}' in hoc for mech in ['ca_boyle','k_fast','k_slow','CaPool'])
    rho=number(pool,'rho');tau=number(pool,'decayConstant');rest=number(pool,'restingConc')
    source_coefficient=-1e6*rho
    unit=dict(ica_units='mA/cm2, outward positive',surface_area_cm2='area_um2 * 1e-8',
        iCa_nA='-ica_mA_per_cm2 * surface_area_cm2 * 1e6',source_units='umol/cm3/ms = mM/ms',
        rho_value=rho,rho_units='umol/cm/nA/ms',rho_SI_equivalent_mol_per_m_A_s=rho*1e8,
        decay_ms=tau,resting_mM=rest,source_per_density=source_coefficient,
        source_expression='S_mM_per_ms = -1e6 * rho * measured_ica_mA_per_cm2',
        fixed_input_closed_form='C(t)=C_inf+(C0-C_inf)*exp(-t/decay); C_inf=rest+decay*S',
        exact_step='C_next=C_inf+(C_old-C_inf)*exp(-dt/decay)',
        clipping_boundary='unclipped analytic solution applies while C>=0; negative-input clipping and cai synchronization are separate diagnostics',
        illustrative_not_measured=[dict(ica_mA_per_cm2=i,source_mM_per_ms=source_coefficient*i,
            unclipped_steady_mM=rest+tau*source_coefficient*i) for i in [-.01,-.001,0.,.001]])
    expected=dict(variants=['frozen','gates_only','pool_only','combined'],allowed_changes='only derivative RHS for five gate states and/or CaPool.concentration',
        protected=['all PARAMETER/UNITS/NEURON/INITIAL blocks','rates() and current cache operations','CaPool cai assignment and negative clipping order',
                   'hoc geometry, capacitance and conductances','synaptic mechanism, conductance and actual input','initialization and concentration synchronization'],
        saved_data_checks=['non-steady actual first gate states, actual fixed voltage and constant tau',
            'CaPool measured ica converted using density units, not nominal point current',
            'STATE concentration compared to analytic recurrence independently of cai',
            'all four actual presynaptic inputs identical at common times; distinguish voltage input from output synaptic s/current',
            'dense data sufficient to resolve 0.100027ms gate; all state/cache/current/concentration/drive arrays saved',
            'preregistered duration, windows, floors and error gates unchanged after runs',
            'paired-step errors and original failures retained; no full-system inference from isolated muscle'])
    result=dict(status='static_source_review_complete_waiting_for_A_materials',scope='independent read-only; zero NEURON imports or scientific simulations',
        source_hashes=sources,verified_facts=dict(source_files_match_frozen_input_manifest=True,
            all_five_generated_gate_updates_are_Euler=True,CaPool_generated_update_is_Euler=True,
            gate_default_initial_values_are_steady_state=True,gate_power_caches_are_preupdate_and_not_refreshed_in_current=True,
            pool_cai_assignment_precedes_separate_negative_clip=True,shared_mechanisms_used_by_neurons_and_muscles=True),
        gates=gates,CaPool_units=unit,review_contract=expected,issues=[],
        pending=['A frozen new protocol and lock','four MOD variants and generated states()','fixed-voltage gate and measured-CaPool-input saved controls',
                 '12 isolated muscle datasets and dense applied inputs','A summary and conclusion boundary'])
    write(ROOT/'source-and-units.json',result)
    if not (ROOT/'review.json').exists():write(ROOT/'review.json',result)
    print(json.dumps(dict(source_files_verified=len(paths),gates=len(gates),rho_SI=unit['rho_SI_equivalent_mol_per_m_A_s'],A_protocol_available=(ISOLATION/'protocol.json').exists())))

if __name__=='__main__':main()
