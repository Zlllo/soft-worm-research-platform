"""Compare the already-computed independent results with A's saved JSON only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from timing_review_static import ROOT, A, sha, write

def read(p):return json.loads(p.read_text())
def main():
    b=read(ROOT/'main-review-calculations.json');payload=(A/'summary.json').read_bytes();a=json.loads(payload)
    assert b['data_audit_passed'] and a['protocol_sha256']==b['protocol_sha256']
    paired=[];factorial=[];interaction=[];initial=[];improvement=[];native=[];differences=[]
    def compare(label,x,y,tol=1e-12):
        difference=float(abs(x-y));differences.append(difference)
        if difference>tol:raise AssertionError(dict(label=label,B=x,A=y,difference=difference,tolerance=tol))
        return difference
    names={'voltage_RMS_mV':('voltage_mV','RMS'),'voltage_mean_abs_mV':('voltage_mV','mean_difference'),
        'voltage_amplitude_relative':('voltage_mV','amplitude_relative'),
        'STATE_calcium_scaled_RMS':('concentration_mM','scaled_RMS'),'STATE_calcium_scaled_stats':('concentration_mM','scaled_stats_max'),
        'cai_scaled_RMS':('cai_mM','scaled_RMS'),'cai_scaled_stats':('cai_mM','scaled_stats_max'),
        'drive_scaled_RMS':('drive_cai','scaled_RMS'),'drive_scaled_stats':('drive_cai','scaled_stats_max')}
    for key,pair in b['pairs'].items():
        for target,(field,metric) in names.items():
            value=pair['metrics'][field][metric]
            if target=='voltage_mean_abs_mV':value=abs(value)
            paired.append(compare(key+'/'+target,value,a['paired_step_comparisons'][key]['errors'][target]['value'],1e-13))
        paired.append(compare(key+'/gate_state_RMS',max(pair['metrics'][g]['RMS'] for g in ['e_q','f_q','p_q','q_q','n_q']),a['paired_step_comparisons'][key]['errors']['gate_state_RMS']['value'],1e-13))
        assert pair['passed']==a['paired_step_comparisons'][key]['passed']
    mapped={'euler_direct_minus_cached':'euler_direct_minus_euler_cached','exp_direct_minus_cached':'exp_direct_minus_exp_cached',
            'cached_exp_minus_euler':'exp_cached_minus_euler_cached','direct_exp_minus_euler':'exp_direct_minus_euler_direct'}
    statmap={'RMS':'RMS','mean':'signed_mean','max_abs':'max_abs'}
    for dt,groups in b['factorial'].items():
        for label,target in mapped.items():
            existing=a['same_dt_factorial_contrasts'][target+'_'+dt]
            for field,stats in existing.items():
                for mine,theirs in statmap.items():
                    factorial.append(compare(dt+'/'+label+'/'+field+'/'+mine,groups[label]['late_4000_8000ms'][field][mine],stats[theirs],1e-18 if field.endswith('_mM') else 1e-12))
        for field,stats in a['same_dt_interactions'][dt].items():
            for mine,theirs in statmap.items():interaction.append(compare(dt+'/interaction/'+field+'/'+mine,groups['interaction']['late_4000_8000ms'][field][mine],stats[theirs],1e-18 if field.endswith('_mM') else 1e-12))
    # A uses closed early windows; independently reproduce that separate convention.
    early={'0_to_0.1_ms':(0,.1),'0.1_to_1_ms':(.1,1),'1_to_10_ms':(1,10),'10_to_100_ms':(10,100)}
    for alg in ['euler','exp']:
        for dt in [.025,.0125,.00625]:
            key=f'{alg}_direct_minus_{alg}_cached_dt{dt:g}';arrays=[]
            for branch,base in [('direct','main-runs'),('cached','reused-main')]:
                with np.load(A/base/f'{alg}_{branch}_dt{dt:g}'/'trace.npz') as z:
                    cols=z['columns'].tolist();data=z['data'][:4001].copy();arrays.append({name:data[:,j] for j,name in enumerate(cols)})
            d,c=arrays;t=d['time_ms']
            for window,(lo,hi) in early.items():
                m=(t>=lo)&(t<=hi)
                for field,s in a['initialization_windows'][key][window].items():
                    delta=d[field][m]-c[field][m]
                    for stat,value in [('RMS',float(np.sqrt(np.mean(delta*delta)))),('signed_mean',float(delta.mean())),('max_abs',float(abs(delta).max()))]:
                        initial.append(compare(key+'/'+window+'/'+field+'/'+stat,value,s[stat],1e-18 if field.endswith('_mM') else 1e-12))
            for branch,base in [('direct','main-runs'),('cached','reused-main')]:
                name=f'{alg}_{branch}_dt{dt:g}';q=a['natural_initial_states'][name];own=b['initialization']['rows'][name]
                compare(name+'/initialV',own['voltage_mV'],q['V_mV'])
                compare(name+'/initialC',own['concentration_mM'],q['STATE_calcium_mM'],1e-18)
                compare(name+'/initialIca',own['ica_mA_cm2'],q['recorded_total_ica_mA_cm2'],1e-18)
                for field,value in zip(['e_q','f_q','p_q','q_q','n_q'],q['STATE']):native.append(compare(name+'/initial/'+field,own[field],value,1e-14))
                for field,value in zip(['cache_e','cache_f','cache_p','cache_q','cache_n'],q['caches']):native.append(compare(name+'/initial/'+field,own[field],value,1e-14))
        mapping={'voltage_mV':'voltage_RMS_mV','concentration_mM':'STATE_calcium_scaled_RMS','cai_mM':'cai_scaled_RMS','drive_cai':'drive_scaled_RMS'}
        for field,target in mapping.items():
            mine=b['readout_improvement'][alg]['fields'][field];their=a['improvement_by_integrator'][alg]['fields'][target]
            improvement.append(compare(alg+'/'+field+'/ratio',mine['direct_over_cached'],their['direct_over_cached'],1e-13))
            assert mine['predeclared_rule_passed']==their['predeclared_improvement_requirement_passed']
    # Equivalent algebra may give different roundoff-level maxima; verify both are far below the same gate.
    passive=[]
    for name,q in b['current_reconstruction'].items():
        for field,c in q['channels'].items():
            x=a['direct_passive_current_phase_checks'][name]['per_channel'][field]
            assert c['max_abs_error_mA_cm2']<1e-12 and x['max_abs_error_mA_cm2']<1e-12 and x['all_samples_within_control_formula_tolerance']
            passive.append(abs(c['max_abs_error_mA_cm2']-x['max_abs_error_mA_cm2']))
    cross=dict(status='complete',passed=True,A_summary_sha256=hashlib.sha256(payload).hexdigest(),
        independent_calculations_sha256=sha(ROOT/'main-review-calculations.json'),A_analysis_functions_imported=False,
        paired_metrics=dict(compared=len(paired),max_abs_difference=max(paired)),
        same_dt_contrast_metrics=dict(compared=len(factorial),max_abs_difference=max(factorial)),
        signal_level_interaction_metrics=dict(compared=len(interaction),max_abs_difference=max(interaction)),
        A_closed_early_window_metrics=dict(compared=len(initial),max_abs_difference=max(initial)),
        improvement_ratios=dict(compared=len(improvement),max_abs_difference=max(improvement)),
        natural_initial_STATE_and_cache_values=dict(compared=len(native),max_abs_difference=max(native)),
        current_algebra_maxima_roundoff_difference=max(passive),all_current_formula_reconstructions_within_same_tolerance=True,
        early_window_note='Independent base calculation uses left-closed/right-open early windows; A uses both endpoints. Both conventions separately recomputed and retained, no model/threshold change.')
    (ROOT/'A-final-summary-snapshot.json').write_bytes(payload)
    write(ROOT/'A-summary-crosscheck.json',cross)
    out=dict(b);out['A_summary_crosscheck']=cross;out['review_status']='complete_pending_report_and_archive';out['pause_request']='Finish current saved-data report/archive, then pause; no new scientific condition authorized.'
    write(ROOT/'main-review.json',out)
    print(json.dumps(cross,ensure_ascii=False))

if __name__=='__main__':main()
