"""Independent offline audit of all 12 main arrays and factorial calculations."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]/'results/c1-current-timing-2026-11-10'
VARIANTS=['euler_cached','exp_cached','euler_direct','exp_direct'];STEPS=[.025,.0125,.00625]
FIELDS=['voltage_mV','e_q','f_q','p_q','q_q','n_q','concentration_mM','cai_mM','drive_cai']
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def near(a,b,atol=1e-15):np.testing.assert_allclose(a,b,rtol=1e-9,atol=atol)
def norm(x):return float(np.sqrt(np.sum(x*x)/x.size))
def load(v,dt,extra=False):
    folder=ROOT/('reused-main' if v.endswith('cached') else 'main-runs')/f'{v}_dt{dt:g}';r=read(folder/'record.json')
    assert r['complete'] and sha(folder/'trace.npz')==r['trace_sha256']
    with np.load(folder/'trace.npz') as z:
        a=z['data'];x={n:a[:,j] for j,n in enumerate(z['columns'].tolist())};e=None
        if extra:
            b=z['extras'];e={n:b[:,j] for j,n in enumerate(z['extra_columns'].tolist())}
    return x,r,e
def check_delta(delta,d):
    near(norm(delta),d['RMS']);near(float(np.mean(delta)),d['signed_mean']);near(float(max(abs(delta))),d['max_abs'])


def main():
    s=read(ROOT/'summary.json');p=read(ROOT/'protocol.json');lock=read(ROOT/'main-execution-lock.json')
    assert s['protocol_sha256']==sha(ROOT/'protocol.json')==lock['protocol_sha256']
    assert sha(ROOT/'execution-code/timing_main_run.py')==lock['runner_sha256']
    assert sha(ROOT/'control-summary.json')==lock['control_summary_sha256']
    for v,paths in lock['kernel_sha256'].items():
        for name,digest in paths.items():assert sha(ROOT/'proposed-variants'/v/name)==digest
    for key,d in read(ROOT/'reused-main-sources.json').items():
        for filename,hash_key in [('trace.npz','trace_sha256'),('record.json','record_sha256'),('spec.json','spec_sha256')]:assert sha(ROOT/'reused-main'/key/filename)==d[hash_key]
        assert 'INFERRED' in d['secondorder_evidence']
    gate_checks=0;current_samples=0;count=0
    with np.load(ROOT/'presynaptic-input.npz') as z:input_ref=z['voltage_mV'][::4].copy()
    for v in VARIANTS:
        data={}
        for dt in STEPS:
            a,r,e=load(v,dt,not v.endswith('cached'));data[dt]=a
            assert len(a['time_ms'])==320001 and np.array_equal(a['peer_at_record_mV'],input_ref)
            if e is not None:
                assert r['explicit_fcurrent_calls']==r['manual_rates_calls']==r['observer_hooks']==0
                assert r['secondorder']==0
                # Independent reconstruction from actual last pre-step snapshots,
                # with physical source constants; excludes initialized row 0.
                for label,gname,factor,E in [('ca','ca_boyle',e['pre_last_e']*e['pre_last_e']*e['pre_last_f']*e['pre_last_h_fcond'],40),
                    ('fast','k_fast',e['pre_last_p']**4*e['pre_last_q'],-60),('slow','k_slow',e['pre_last_n'],-60)]:
                    difference=e['post_I_'+label+'_mA_cm2']-r['gmax_S_cm2'][gname]*factor*(e['pre_last_voltage_mV']-E)
                    reported=s['direct_passive_current_phase_checks'][f'{v}_dt{dt:g}']['per_channel'][label]
                    near(max(abs(difference[1:])),reported['max_abs_error_mA_cm2'])
                    limit=np.maximum(1e-12,1e-9*abs(e['post_I_'+label+'_mA_cm2'][1:]))
                    assert np.all(abs(difference[1:])<=limit)
                    current_samples+=len(difference)-1
                assert max(abs(e['pre_last_time_ms'][1:]-(a['time_ms'][1:]-dt)))<1e-5
            count+=1
        for co,fi in [(.025,.0125),(.0125,.00625),(.025,.00625)]:
            a=data[co];b=data[fi];mask=a['time_ms']>=4000
            av=a['voltage_mV'][mask];bv=b['voltage_mV'][mask]
            span=lambda x:np.quantile(x,.95)-np.quantile(x,.05)
            values=dict(voltage_RMS_mV=norm(av-bv),voltage_mean_abs_mV=abs(av.mean()-bv.mean()),
                voltage_amplitude_relative=abs(span(av)-span(bv))/max(span(bv),1))
            limits=dict(voltage_RMS_mV=.5,voltage_mean_abs_mV=.2,voltage_amplitude_relative=.05,gate_state_RMS=.01)
            for field,label,floor in [('concentration_mM','STATE_calcium',1e-9),('cai_mM','cai',1e-9),('drive_cai','drive',.001)]:
                aa=a[field][mask];bb=b[field][mask];den=max(span(bb),floor)
                values[label+'_scaled_RMS']=norm(aa-bb)/den
                sa=np.array([aa.mean(),aa.std(),*np.quantile(aa,[.05,.95])]);sb=np.array([bb.mean(),bb.std(),*np.quantile(bb,[.05,.95])])
                values[label+'_scaled_stats']=float(max(abs(sa-sb))/den)
                limits[label+'_scaled_RMS']=limits[label+'_scaled_stats']=.05
            values['gate_state_RMS']=max(norm(a[k+'_q'][mask]-b[k+'_q'][mask]) for k in 'efpqn')
            reported=s['paired_step_comparisons'][f'{v}_{co:g}_vs_{fi:g}']
            for name,value in values.items():
                near(value,reported['errors'][name]['value']);near(limits[name],reported['errors'][name]['limit'])
                assert reported['gates'][name]==bool(value<=limits[name]+1e-12);gate_checks+=1
            assert reported['passed']==all(reported['gates'].values())
    contrast_fields=interaction_fields=window_fields=0
    for dt in STEPS:
        values={v:load(v,dt)[0] for v in VARIANTS};times=values['euler_cached']['time_ms'];late=times>=4000
        assert all(np.array_equal(values[v]['synapse_s'],values['euler_cached']['synapse_s']) for v in VARIANTS)
        for a,b in [('euler_direct','euler_cached'),('exp_direct','exp_cached'),('exp_cached','euler_cached'),('exp_direct','euler_direct')]:
            key=f'{a}_minus_{b}_dt{dt:g}'
            for field in FIELDS:
                check_delta(values[a][field][late]-values[b][field][late],s['same_dt_factorial_contrasts'][key][field]);contrast_fields+=1
            if a.endswith('direct') and b.endswith('cached'):
                for lo,hi in p['main']['initialization_report_windows_ms']:
                    mask=(times>=lo)&(times<=hi)
                    for field in FIELDS:
                        check_delta(values[a][field][mask]-values[b][field][mask],s['initialization_windows'][key][f'{lo:g}_to_{hi:g}_ms'][field]);window_fields+=1
        for field in FIELDS:
            delta=values['exp_direct'][field][late]-values['exp_cached'][field][late]-values['euler_direct'][field][late]+values['euler_cached'][field][late]
            # Independent subtraction grouping can differ by roundoff.
            check_delta(delta,s['same_dt_interactions'][f'dt{dt:g}'][field]);interaction_fields+=1
        for mode in ['euler','exp']:
            cached=values[mode+'_cached'];direct=values[mode+'_direct']
            d=s['native_readout_factor_effects'][f'{mode}_dt{dt:g}']
            near(max(abs(cached[k+'_q'][0]-direct[k+'_q'][0]) for k in 'efpqn'),d['STATE_initial_max_difference'])
            near(direct['ica_mA_cm2'][0]-cached['ica_mA_cm2'][0],d['native_total_ica_direct_minus_cached_mA_cm2'])
    for mode,d in s['improvement_by_integrator'].items():
        flags=[]
        for name,item in d['fields'].items():
            co=s['paired_step_comparisons'][f'{mode}_cached_0.0125_vs_0.00625']['errors'][name]
            di=s['paired_step_comparisons'][f'{mode}_direct_0.0125_vs_0.00625']['errors'][name]
            near(di['value']/co['value'],item['direct_over_cached'])
            flag=di['value']<=.5*co['value'] or co['value']<=.1*co['limit']
            assert item['predeclared_improvement_requirement_passed']==flag;flags.append(flag)
        assert d['all_predeclared_improvement_requirements_passed']==all(flags)
    assert len(list((ROOT/'main-runs').glob('*/record.json')))==6 and len(list((ROOT/'reused-main').glob('*/record.json')))==6
    result=dict(independent_main_audit_passed=True,all_main_arrays_checked=count,new_main=6,reused_main=6,
        paired_gate_values_checked=gate_checks,passive_current_samples_checked=current_samples,
        factorial_contrast_field_summaries_checked=contrast_fields,interaction_field_summaries_checked=interaction_fields,
        early_window_field_summaries_checked=window_fields,analyzer_imported=False,NEURON_imported=False,
        protocol_sha256=sha(ROOT/'protocol.json'),summary_sha256=sha(ROOT/'summary.json'),
        stage_budget_main_six_only=True,new_full302_body_training=0)
    (ROOT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
