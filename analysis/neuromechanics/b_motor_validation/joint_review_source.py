"""Read-only source/protocol checks for the full-system candidate."""
import ast
import difflib
import json
import re
import numpy as np
from pathlib import Path
from joint_review_compute import REPO, OUT, A, OLD, read, save, digest, check

COMPONENT = REPO/'results/c1-current-timing-2026-11-10/proposed-variants/exp_direct'
if not COMPONENT.exists(): COMPONENT=A/'candidate-reference'
CHANGED = ('ca_boyle.mod','k_fast.mod','k_slow.mod','CaPool.mod')
REPLACEMENTS = {
    'ca_boyle.mod': [('fopen0 = e_fcond * f_fcond * h_fcond',
                     'fopen0 = (e_q ^ e_instances) * (f_q ^ f_instances) * h_fcond'),
                    ("e_q' = rate_e_q", "e_q' = (e_inf-e_q)/e_tau"),
                    ("f_q' = rate_f_q", "f_q' = (f_inf-f_q)/f_tau")],
    'k_fast.mod': [('fopen0 = p_fcond * q_fcond',
                   'fopen0 = (p_q ^ p_instances) * (q_q ^ q_instances)'),
                  ("p_q' = rate_p_q", "p_q' = (p_inf-p_q)/p_tau"),
                  ("q_q' = rate_q_q", "q_q' = (q_inf-q_q)/q_tau")],
    'k_slow.mod': [('fopen0 = n_fcond','fopen0 = n_q ^ n_instances'),
                  ("n_q' = rate_n_q", "n_q' = (n_inf-n_q)/n_tau")],
    'CaPool.mod': [("concentration' = rate_concentration",
                   "concentration' = (iCa/surfaceArea)*rho-(concentration-restingConc)/decayConstant")]
}


def function_dump(path, fname):
    tree=ast.parse(path.read_text())
    nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==fname]
    check(len(nodes)==1,f'{path}:{fname} unique definition')
    return ast.dump(nodes[0],include_attributes=False)


def cpp_function(text, signature):
    match=re.search(signature+r'[^;{]*\{',text)
    check(match is not None,'Missing generated function '+signature)
    start=match.end()-1;depth=0
    for i in range(start,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return re.sub(r'\s+','',text[start:i+1])
    raise AssertionError('Unclosed generated function')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report=dict(scope='Read-only; no simulator import, compilation, or scientific execution',
        review_script_sha256=digest(Path(__file__)),
        shared_density_population=dict(generic_C1_neurons=284,C1_muscles=95,B_prototype_neurons=18),
        candidate_available=(A/'protocol.json').exists(),checks={},pending=[])
    checks=report['checks']; evidence={}
    for filename in CHANGED:
        before=(OLD/'kernels'/filename).read_text();expected=before
        for src,dst in REPLACEMENTS[filename]:
            check(expected.count(src)==1,f'unique prescribed replacement {filename}:{src}')
            expected=expected.replace(src,dst)
        component=(COMPONENT/filename).read_text()
        check(expected==component,f'exact prescribed component changes: {filename}')
        checks['component_exact_allowed_change_'+filename]=True
        diff=''.join(difflib.unified_diff(before.splitlines(True),component.splitlines(True),
            fromfile='parent/'+filename,tofile='exp_direct/'+filename))
        (OUT/(filename+'.diff')).write_text(diff)
        evidence[filename]=dict(parent_sha256=digest(OLD/'kernels'/filename),
            exp_direct_sha256=digest(COMPONENT/filename),replacement_count=len(REPLACEMENTS[filename]))
        if (A/'kernels'/filename).exists():
            checks['candidate_matches_exp_direct_'+filename]=digest(A/'kernels'/filename)==digest(COMPONENT/filename)
    report['mechanism_evidence']=evidence
    report['retained_boundaries']=[
        'h_q is assigned from cai; h_fcond remains 1+(h_q-1)*h_alpha',
        'NMJ chemical current is cached; no NMJ current/readout rewrite',
        'CaPool cai assignment and later negative-concentration clipping remain in their original locations',
        'Initialization expressions are retained; direct STATE current readout can change native initial current',
        'Shared C1 density kernels affect 284 generic neurons plus 95 muscles; 18 B density gmax disabled',
        'Private channel reversal remains Ca 40 mV / K -60 mV; ionic buffer eca is not channel reversal']
    kernel=A/'kernels'
    if kernel.exists():
        for path in sorted((OLD/'kernels').iterdir()):
            if path.suffix not in ('.mod','.hoc') or path.name in CHANGED:continue
            if (kernel/path.name).exists():
                checks['unchanged_kernel_'+path.name]=digest(path)==digest(kernel/path.name)
            else:report['pending'].append('kernel '+path.name)
    else:report['pending'].append('candidate kernels')
    parent_protocol=read(OLD/'protocol.json')
    if (A/'protocol.json').exists():
        candidate=read(A/'protocol.json')
        keys=['duration_s','seed','feedback','nmj_nS','gap_nS','feedback_gain_pA_per_strain',
            'feedback_clip_pA','exchange_ms','recorded_ms','steps_ms','max_new_full_runs',
            'window_s','acceptance','definitions']
        report['protocol_values']={k:dict(parent=parent_protocol[k],candidate=candidate.get(k),
            unchanged=parent_protocol[k]==candidate.get(k)) for k in keys}
        checks['all_frozen_protocol_definitions_unchanged']=all(x['unchanged'] for x in report['protocol_values'].values())
        report['protocol_wording_review']={
            'refinement_rule':dict(parent=parent_protocol['refinement_rule'],candidate=candidate['refinement_rule'],
                assessment='Same 4+2 rule, now explicitly includes inherited validity/incomplete-run failures; no extra run or retry'),
            'reference':dict(parent=parent_protocol['reference'],candidate=candidate['reference'],
                assessment='Same finer adjacent reference plus coarse/fine diagnostic')}
        checks['same_4_plus_2_refinement_and_reference_semantics']=candidate['initial_cases']==4 and candidate['max_new_full_runs']==6 and 'BOTH modes' in candidate['refinement_rule'] and 'No retries' in candidate['refinement_rule']
        report['candidate_protocol_sha256']=digest(A/'protocol.json')
        if (A/'protocol-lock.json').exists():
            report['protocol_lock']=read(A/'protocol-lock.json')
            checks['protocol_lock_sha256']=report['protocol_lock']['protocol_sha256']==digest(A/'protocol.json')
    else:report['pending'].append('candidate protocol/lock')
    manifest=A/('sources.json' if (A/'sources.json').exists() else 'inputs.json')
    if manifest.exists():
        inputs=read(manifest);report['candidate_input_manifest']=manifest.name;report['candidate_inputs_sha256']=digest(manifest)
        report['candidate_input_checks']={p:(A/p).exists() and digest(A/p)==v['sha256'] for p,v in inputs.items()}
        checks['candidate_all_input_hashes']=all(report['candidate_input_checks'].values())
        checks['source_manifest_locked_in_protocol']=read(A/'protocol.json')['source_manifest_sha256']==digest(A/'sources.json')
        report['input_differences_vs_parent']={}
        for p,v in read(OLD/'inputs.json').items():
            if p in inputs and (A/p).exists() and digest(OLD/p)!=digest(A/p):
                report['input_differences_vs_parent'][p]=dict(parent_sha256=digest(OLD/p),candidate_sha256=digest(A/p))
    else:report['pending'].append('candidate inputs')
    # These functions encode the 18-B intrinsic current setup, connectivity changes,
    # seed/initialization, feedback projection and interventions. Exact AST checks
    # avoid accepting an innocuous-looking added call in the control path.
    driver=A/'runtime/driver.py'
    if driver.exists():
        for func in ('configure','field'):
            checks['driver_'+func+'_AST_unchanged']=function_dump(driver,func)==function_dump(OLD/'runtime/driver.py',func)
        (OUT/'driver.diff').write_text(''.join(difflib.unified_diff(
            (OLD/'runtime/driver.py').read_text().splitlines(True),driver.read_text().splitlines(True),
            fromfile='parent/driver.py',tofile='candidate/driver.py')))
        report['driver_calls']=[ast.unparse(n.func) for n in ast.walk(ast.parse(driver.read_text())) if isinstance(n,ast.Call)]
        checks['driver_has_no_explicit_fcurrent_or_rates']=not any(
            c.endswith(('.fcurrent','.rates')) for c in report['driver_calls'])
    else:report['pending'].append('candidate frozen driver')
    for relative in ('runtime/src/thermotaxis/neuromechanics/full302_c1.py',
                     'runtime/src/thermotaxis/neuromechanics/boyle_muscle_body.py',
                     'connection-plan.json','selected-parameters.json','runtime/body/worker'):
        if (A/relative).exists():checks['unchanged_'+relative]=digest(A/relative)==digest(OLD/relative)
        else:report['pending'].append(relative)
    if manifest.exists():
        shared_paths=[p for p in read(OLD/'inputs.json') if p.startswith('runtime/')]
        report['parent_runtime_file_comparison']={p:(A/p).exists() and digest(A/p)==digest(OLD/p) for p in shared_paths}
        checks['entire_parent_runtime_byte_identical']=all(report['parent_runtime_file_comparison'].values())
        runner=REPO/'analysis/neuromechanics/b_motor_validation/joint_run.py'
        if runner.exists():
            text=runner.read_text();tree=ast.parse(text)
            calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
            report['runner_review']=dict(sha256=digest(runner),calls=calls,
                assessment='PinnedNetwork adds constructor-only assertions; PinnedBody substitutes frozen binary path. Frozen driver is byte-identical. Failure serialization occurs only after an exception, with no integration-loop callbacks.')
            checks['runner_no_extra_current_rates_advance_or_hooks']=not any(
                c.endswith(('.fcurrent','.rates','.fadvance','.finitialize','.advance','.step','.FInitializeHandler')) for c in calls)
            if (A/'protocol.json').exists():checks['runner_matches_protocol_lock']=digest(runner)==read(A/'protocol.json')['runner_sha256']
    generated={};generated_pending=[]
    for filename in CHANGED:
        stem=Path(filename).stem;current=kernel/'arm64'/(stem+'.cpp');reference=A/'candidate-reference/arm64'/(stem+'.cpp')
        if not current.exists():generated_pending.append(stem);continue
        current_text=current.read_text();reference_text=reference.read_text()
        signatures=[r'static\s+int\s+states\s*\(',r'static\s+double\s+_nrn_current\s*\(',r'static\s+int\s+rates\s*\(',r'void\s+initmodel\s*\(']
        funcs={s:cpp_function(current_text,s)==cpp_function(reference_text,s) for s in signatures}
        generated[stem]=dict(sha256=digest(current),reference_sha256=digest(reference),function_matches=funcs)
        checks['generated_'+stem+'_state_current_rates_init_match_verified_component']=all(funcs.values())
    report['generated_code_comparison']=generated;report['generated_code_pending']=generated_pending
    untouched={}
    for mod in kernel.glob('*.mod') if kernel.exists() else []:
        if mod.name in CHANGED:continue
        current=kernel/'arm64'/(mod.stem+'.cpp');parent=A/'parent-kernel-generated'/(mod.stem+'.cpp')
        if not current.exists():continue
        matches={}
        for signature in [r'static\s+int\s+states\s*\(',r'static\s+double\s+_nrn_current\s*\(']:
            blocks=[]
            for path in (current,parent):
                try:blocks.append(cpp_function(path.read_text(),signature))
                except AssertionError:blocks.append(None)
            matches[signature]=blocks[0]==blocks[1]
        untouched[mod.stem]=matches
        checks['unchanged_generated_'+mod.stem]=all(matches.values())
    report['untouched_generated_functions']=untouched
    if (A/'execution-lock.json').exists():
        execution=read(A/'execution-lock.json')
        report['execution_lock_sha256']=digest(A/'execution-lock.json')
        checks['execution_protocol_lock']=execution['protocol_sha256']==digest(A/'protocol.json')
        checks['execution_all_runtime_hashes']=all(digest(A/p)==v for p,v in execution['runtime_sha256'].items())
        checks['execution_generated_validation_hash']=execution['generated_validation_sha256']==digest(A/'generated-code-validation.json')
        checks['execution_preflight_hash']=execution['preflight_sha256']==digest(A/'preflight.json')
        preflight=read(A/'preflight.json');report['A_zero_step_preflight']=preflight
        checks['preflight_population_default_and_zero_clocks']=all([
            preflight['actual_neurons']==302,preflight['actual_muscles']==95,
            preflight['actual_chemical']==4665,preflight['actual_gap_pairs']==1192,
            preflight['shared_active_C1_neurons']==284,preflight['shared_active_C1_muscles']==95,
            preflight['B_points']==18,preflight['B_density_gmax_zero'],preflight['secondorder']==0,
            preflight['neural_time_ms']==0,preflight['body']['time_s']==0,
            preflight['body']['internal_time_s']==0,preflight['body']['dt_s']==.001,
            preflight['neural_fadvance_calls']==0,preflight['body_steps']==0])
        expected=(-60+np.random.default_rng(13).normal(0,.2,18)).tolist()
        checks['preflight_seed13_independently_regenerated']=preflight['B_seed_initial_voltage_mV']==expected
        checks['preflight_AVB_gap_and_NMJ']=all(abs(v-.1)<1e-12 for v in preflight['AVB_total_per_B_nS'].values()) and preflight['B_NMJ_edges_checked']==132
        report['preflight_evidence_boundary']='A executed the zero-step setup. B audited its code, hashes, source population/configuration and regenerated seed; B did not independently load the simulator.'
    official=REPO/'results/c1-current-timing-review-2026-11-10'
    if not official.exists():official=OUT/'review-inputs/official'
    manifest=read(official/'official-sources.json')
    report['reused_official_sources']={p:dict(**manifest[p],locally_reverified_sha256=digest(official/'official-9.0.2'/p))
        for p in ['fadvance.cpp','treeset.cpp','hh.mod']}
    checks['saved_official_sources_intact']=all(v['sha256']==v['locally_reverified_sha256'] for v in report['reused_official_sources'].values())
    report['source_review_complete']=not report['pending']
    report['all_available_checks_pass']=all(checks.values())
    save(OUT/'source-protocol-review.json',report)
    print(json.dumps(dict(complete=report['source_review_complete'],available_checks_pass=report['all_available_checks_pass'],pending=report['pending']),ensure_ascii=False))


if __name__=='__main__': main()
