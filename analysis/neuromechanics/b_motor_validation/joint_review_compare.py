"""Cross-check independent results against saved summaries, never A code."""
import argparse
import json
import numpy as np
from pathlib import Path
from joint_review_compute import OUT, OLD, A, read, save, digest, check, compute, compare, name, STEPS, rms


def crosscheck(own, published):
    differences={};mismatches=[]
    for key,pair in own['comparisons'].items():
        other=published['comparisons'][key]
        for metric,values in pair['errors'].items():
            for field in ('value','limit'):
                d=abs(values[field]-other['errors'][metric][field])
                differences[f'{key}/{metric}/{field}']=d
                if d>1e-10:mismatches.append(dict(pair=key,metric=metric,field=field,
                    independent=values[field],published=other['errors'][metric][field],difference=d))
            check(values['passed']==other['gates'][metric], f'Gate disagreement {key}/{metric}')
        check(pair['passed']==other['passed'], f'Pair disagreement {key}')
    for k in ('new_full_runs','fine_required','neural_step_convergence_accepted'):
        if k=='neural_step_convergence_accepted' and own[k] is None:continue
        check(own[k]==published[k],f'Batch summary disagreement: {k}')
    return dict(passed=not mismatches,review_compare_script_sha256=digest(Path(__file__)),numeric_scalars_compared=len(differences),
        max_abs_numeric_difference=max(differences.values(),default=0),mismatches=mismatches,
        all_pair_and_gate_decisions_identical=True,A_functions_imported=False,
        candidate_final_status_pending=own['neural_step_convergence_accepted'] is None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',action='store_true');args=parser.parse_args()
    if args.parent:
        own=read(OUT/'parent-independent.json');published=read(OLD/'summary.json')
        result=crosscheck(own,published)
        result.update(independent_sha256=digest(OUT/'parent-independent.json'),
            published_sha256=digest(OLD/'summary.json'),purpose='Historical formula cross-check; not a new candidate baseline')
        save(OUT/'parent-crosscheck.json',result);print(json.dumps(result));return
    own=read(OUT/'independent-results.json')
    published=read(A/'summary.json')
    result=crosscheck(own,published)
    result.update(independent_sha256=digest(OUT/'independent-results.json'),A_summary_sha256=digest(A/'summary.json'))
    save(OUT/'A-summary-crosscheck.json',result)
    if (A/'refinement-decision.json').exists():
        decision=read(A/'refinement-decision.json')
        initial={k:v['failed_gates'] for k,v in own['comparisons'].items() if k.endswith('0.025_vs_0.0125')}
        check(decision['fine_required']==own['fine_required'],'Adaptive refinement decision mismatch')
        check(decision['failed_gates']==initial,'Adaptive refinement failure list mismatch')
        check(decision['new_full_conditions_observed']==4,'Refinement was not decided from four initial conditions')
        save(OUT/'refinement-decision-crosscheck.json',dict(passed=True,
            A_decision_sha256=digest(A/'refinement-decision.json'),independent_refinement_required=own['fine_required'],
            identical_initial_failed_gates=initial,new_full_conditions_at_decision=4,
            own_decision_snapshot_sha256=digest(OUT/'independent-refinement-decision.json')))
    # Re-read raw records for matched-dt old/new controls; the candidate is the
    # reference for scale only. These differences are NOT timestep errors.
    old,oldz=compute(OLD);new,newz=compute(A)
    same_dt={}
    acceptance=read(OLD/'protocol.json')['acceptance']
    for mode in ('off','live'):
        for dt in STEPS:
            n=name(mode,dt)
            if n not in newz:continue
            comparison=compare(oldz[n],newz[n],old['metrics'][n],new['metrics'][n],acceptance)
            same_dt[n]=dict(description='Historical parent minus candidate at identical dt; not convergence acceptance',
                values_and_reference_scales=comparison,
                recorded_initial_arrays_identical={field:bool(np.array_equal(oldz[n][field][0],newz[n][field][0]))
                    for field in ('neuron_voltage_mV','muscle_voltage_mV','muscle_calcium_mM','B_gate','pose','side_length')},
                speed_candidate_minus_parent_um_s=new['metrics'][n]['late_speed_um_s']-old['metrics'][n]['late_speed_um_s'],
                displacement_candidate_minus_parent_um=new['metrics'][n]['late_displacement_um']-old['metrics'][n]['late_displacement_um'])
    ratios={}
    for key,pair in new['comparisons'].items():
        ratios[key]={metric:dict(parent=old['comparisons'][key]['errors'][metric]['value'],candidate=v['value'],
            candidate_over_parent=(v['value']/old['comparisons'][key]['errors'][metric]['value']
                if old['comparisons'][key]['errors'][metric]['value']>1e-15 else None))
            for metric,v in pair['errors'].items()}
    effects={}
    for dt in STEPS:
        no,nl=name('off',dt),name('live',dt)
        if no in newz and nl in newz:
            effects[f'{dt:g}']=dict(B_unshifted_RMS_live_minus_off_mV=float(rms(newz[nl]['B'][400:]-newz[no]['B'][400:])),
                shape_RMS_live_minus_off_um=float(rms(newz[nl]['body_y_um'][400:]-newz[no]['body_y_um'][400:])),
                speed_live_minus_off_um_s=new['metrics'][nl]['late_speed_um_s']-new['metrics'][no]['late_speed_um_s'])
    save(OUT/'historical-comparison.json',dict(same_dt=same_dt,candidate_over_parent_step_errors=ratios,
        candidate_on_off_descriptive_differences=effects,independent_raw_data_calculation=True))
    print(json.dumps(result))


if __name__=='__main__':main()
