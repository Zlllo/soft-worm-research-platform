"""Read-only independent protocol, input-file and compiled-source audit."""
import hashlib
import json
from pathlib import Path
import re
import ast
import difflib
import numpy as np
from integration_review_source import ROOT, REPO, FROZEN, ISOLATION, block, sha, write

GATES={'ca_boyle': [('e_q','e'),('f_q','f')], 'k_fast':[('p_q','p'),('q_q','q')], 'k_slow':[('n_q','n')]}
def compact(x):return re.sub(r'\s+','',x)
def cpp_block(text,name,kind='int'):
    return block(text,r'static '+kind+r'\s+'+re.escape(name)+r'\s*\([^;\n]*\)\s*\{')

def main():
    protocol=json.loads((ISOLATION/'protocol.json').read_text())
    assert sha(ISOLATION/'protocol.json')==json.loads((ISOLATION/'protocol-lock.json').read_text())['protocol_sha256']
    assert protocol['variants']==['original','gates','pool','both']
    assert protocol['dt_ms']==[.025,.0125,.00625] and protocol['main']['conditions']==12
    assert protocol['main']['record_ms']==.025 and protocol['main']['duration_ms']==8000
    assert protocol['main']['window_ms']==[4000,8000]
    assert sha(ISOLATION/'sources.json')==protocol['source_manifest_sha256']
    assert sha(ISOLATION/'presynaptic-input.npz')==protocol['input_sha256']
    assert sha(ISOLATION/'harness/IntegrationCaInput.mod')==protocol['helper_sha256']
    prepare_path=REPO/'analysis/neuromechanics/b_motor_validation/integration_prepare.py'
    locked_prepare=ISOLATION/'harness/integration_prepare_at_lock.py'
    if sha(prepare_path)!=protocol['prepare_script_sha256']:
        assert locked_prepare.exists() and sha(locked_prepare)==protocol['prepare_script_sha256']
        def prepare_ast(p):return ast.dump(next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='prepare'))
        assert prepare_ast(prepare_path)==prepare_ast(locked_prepare)
        (ROOT/'prepare-build-adjustments.diff').write_text(''.join(difflib.unified_diff(locked_prepare.read_text().splitlines(True),prepare_path.read_text().splitlines(True),fromfile='at_protocol_lock',tofile='current_build_resume')))
    source_map=json.loads((ISOLATION/'sources.json').read_text())
    for path,d in source_map.items():assert sha(ISOLATION/path)==d['sha256'],path
    with np.load(ISOLATION/'presynaptic-input.npz') as z:
        t=z['time_ms'];v=z['voltage_mV']
        assert np.array_equal(t,np.arange(1280001)*.00625)
        # Independently express the declared piecewise cubic blend.
        u=(t-250)/100
        w=np.where(t<=250,0.,np.where(t>=350,1.,3*u*u-2*u*u*u))
        expected=(1-w)*(-60)+w*(-32+6*np.sin(2*np.pi*1.9*t/1000))
        err=float(abs(v-expected).max());assert err<1e-12
        file_info=dict(shape=list(v.shape),time_step_ms=.00625,end_ms=float(t[-1]),
            voltage_range_mV=[float(v.min()),float(v.max())],independent_analytic_max_error_mV=err)
    protected={};variants={};pending=[];issues=[]
    files=sorted(p for p in (ISOLATION/'frozen').iterdir() if p.suffix in ['.mod','.hoc'])
    for variant in protocol['variants']:
        folder=ISOLATION/'variants'/variant
        if not folder.exists():pending.append(f'{variant}: directory absent during snapshot');continue
        changes=[];generated={};fingerprints={}
        for old in files:
            p=folder/old.name
            if not p.exists():pending.append(f'{variant}: {old.name} absent during snapshot');continue
            original=old.read_text();expected=original
            if variant in ['gates','both'] and old.stem in GATES:
                for state,prefix in GATES[old.stem]:
                    expected=re.sub(r'('+re.escape(state)+r"'\s*=\s*)rate_"+state+r'\b',
                        lambda m:m.group(1)+f'({prefix}_inf-{state})/{prefix}_tau',expected)
            if variant in ['pool','both'] and old.stem=='CaPool':
                expected=re.sub(r"(concentration'\s*=\s*)rate_concentration\b",
                    lambda m:m.group(1)+'(iCa/surfaceArea)*rho-(concentration-restingConc)/decayConstant',expected)
            actual=p.read_text()
            if actual!=expected:
                issues.append(dict(severity='error',file=str(p.relative_to(REPO)),message='source contains a change beyond the declared derivative-RHS substitution'))
            if actual!=original:changes.append(old.name)
            fingerprints[old.name]=sha(p)
            # Source blocks protecting units, initialization, caches and ionic ordering are byte-identical.
            if old.stem in ['ca_boyle','k_fast','k_slow','CaPool']:
                for label in ['NEURON','UNITS','PARAMETER','INITIAL','BREAKPOINT']:
                    assert block(actual,r'\b'+label+r'\s*\{')==block(original,r'\b'+label+r'\s*\{'),(variant,old.name,label)
                assert block(actual,r'PROCEDURE rates\s*\([^)]*\)\s*\{')==block(original,r'PROCEDURE rates\s*\([^)]*\)\s*\{')
                generated_path=folder/'arm64'/f'{old.stem}.cpp'
                if not generated_path.exists():pending.append(f'{variant}: generated {old.stem}.cpp absent');continue
                cpp=generated_path.read_text();baseline=(ISOLATION/'variants/original/arm64'/f'{old.stem}.cpp').read_text()
                state=cpp_block(cpp,'states')
                should_exp=(variant in ['gates','both'] and old.stem in GATES) or (variant in ['pool','both'] and old.stem=='CaPool')
                assert ('exp(' in state)==should_exp,(variant,old.stem,state)
                if should_exp:
                    variables=[('concentration','decayConstant')] if old.stem=='CaPool' else [(a,b+'_tau') for a,b in GATES[old.stem]]
                    for q,tau in variables:
                        update=next(line for line in state.splitlines() if re.search(r'\b'+q+r'\s*=',line))
                        assert 'exp(' in update and tau in update
                        assert ('-((1.0))/' if old.stem=='CaPool' else '-1.0') in compact(update)
                else:assert compact(state)==compact(cpp_block(baseline,'states'))
                same={}
                for fun,kind in [('rates','int'),('nrn_init','void'),('nrn_state','void')]+([] if old.stem=='CaPool' else [('_nrn_current','double')]):
                    same[fun]=compact(cpp_block(cpp,fun,kind))==compact(cpp_block(baseline,fun,kind))
                    assert same[fun],(variant,old.stem,fun)
                generated[old.stem]=dict(exponential=should_exp,generated_states_code=state,
                    nonstate_generated_functions_identical=same,cpp_sha256=sha(generated_path))
        helper=folder/'IntegrationCaInput.mod'
        if helper.exists():assert sha(helper)==protocol['helper_sha256']
        variants[variant]=dict(changed_files=changes,source_sha256=fingerprints,generated=generated)
    result=dict(protocol_sha256=sha(ISOLATION/'protocol.json'),input_sha256=protocol['input_sha256'],
        prepare_script_at_lock_sha256=protocol['prepare_script_sha256'],prepare_script_current_sha256=sha(prepare_path),
        scientific_prepare_function_unchanged_after_lock=True,
        input=file_info,variants=variants,issues=issues,pending=pending,
        static_review_complete=not pending and not issues,
        conclusions=dict(only_declared_derivative_RHS_changed=not issues,
            all_parameters_geometry_initialization_cache_and_sync_preserved=not issues and not pending,
            four_versions_actual_generated_state_functions_checked=not pending,
            source_is_fixed_prescribed_component_input_not_autonomous_activity=True,
            analytic_controls_original_and_combined_only='adequate for independent gate/pool checks because fixed voltage/state equations are uncoupled; intermediate variants still checked in generated code',
            input_hold_note='same dense source, but left-hold duration varies with dt; dt effects include discretization of prescribed input, not only density integration; same-dt variant comparison isolates the patch',
            input_identity_definition='presynaptic voltage and same-dt synaptic s must match; synaptic current may change with postsynaptic voltage and is an output, not a required identical current clamp'))
    write(ROOT/'protocol-and-patches.json',result)
    review=json.loads((ROOT/'source-and-units.json').read_text())
    review.update(status='new_protocol_and_patches_reviewed_waiting_for_saved_controls_and_main_data',
        protocol_review='protocol-and-patches.json',issues=issues,pending=pending+['saved control data','12 main trajectories and summary'])
    write(ROOT/'review.json',review)
    print(json.dumps(dict(variants=len(variants),pending=pending,issues=issues,input_error_mV=err)))

if __name__=='__main__':main()
