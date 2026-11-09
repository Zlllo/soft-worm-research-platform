"""Independent NumPy/stdlib audit. Does not import the main analyzer or NEURON."""
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[3]/'results/c1-muscle-integration-isolation-2026-11-09'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def near(a,b):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-16)
def nvalue(text,n):return float(re.search(r'\b'+n+r'\s*=\s*([-+0-9.Ee]+)',text).group(1))
def load(case,v,dt):
    folder=ROOT/'runs'/f'{case}_{v}_dt{dt:g}';record=read(folder/'record.json')
    assert record['complete'] and sha(folder/'trace.npz')==record['trace_sha256']
    with np.load(folder/'trace.npz') as z:return dict(zip(z['columns'].tolist(),z['data'].T.copy())),record
def norm(x):return float(np.sqrt(np.dot(x,x)/len(x)))


def main():
    summary=read(ROOT/'summary.json');p=read(ROOT/'protocol.json')
    assert summary['protocol_sha256']==sha(ROOT/'protocol.json')==read(ROOT/'protocol-lock.json')['protocol_sha256']
    assert sha(ROOT/'harness/integration_prepare_at_lock.py')==p['prepare_script_sha256']
    for name,d in read(ROOT/'sources.json').items():assert sha(ROOT/name)==d['sha256']
    # Independently reconstruct permitted source diffs; protect complete files,
    # including INITIAL, rates, caches, current calculation and clipping order.
    replacements={"e_q' = rate_e_q":"e_q' = (e_inf-e_q)/e_tau",
        "f_q' = rate_f_q":"f_q' = (f_inf-f_q)/f_tau", "p_q' = rate_p_q":"p_q' = (p_inf-p_q)/p_tau",
        "q_q' = rate_q_q":"q_q' = (q_inf-q_q)/q_tau", "n_q' = rate_n_q":"n_q' = (n_inf-n_q)/n_tau"}
    files=0
    for variant in p['variants']:
        for path in (ROOT/'frozen').iterdir():
            if path.suffix not in ['.mod','.hoc']:continue
            expected=path.read_text()
            if variant in ['gates','both'] and path.stem in ['ca_boyle','k_fast','k_slow']:
                for a,b in replacements.items():expected=expected.replace(a,b)
            if variant in ['pool','both'] and path.name=='CaPool.mod':
                expected=expected.replace("concentration' = rate_concentration","concentration' = (iCa/surfaceArea)*rho-(concentration-restingConc)/decayConstant")
            assert (ROOT/'variants'/variant/path.name).read_text()==expected
            files+=1
        assert sha(ROOT/'variants'/variant/'IntegrationCaInput.mod')==p['helper_sha256']
    control_checked=0;pool_recurrence_errors={}
    for variant in ['original','both']:
        for dt in p['dt_ms']:
            x,d=load('gates',variant,dt);entry=summary['analytic_controls'][f'gates_{variant}_{dt:g}']
            errors=[]
            for mech,prefix in [('ca_boyle','e'),('ca_boyle','f'),('k_fast','p'),('k_fast','q'),('k_slow','n')]:
                src=(ROOT/'frozen'/f'{mech}.mod').read_text();tau=nvalue(src,prefix+'_timeCourse_tau')
                mid=nvalue(src,prefix+'_steadyState_midpoint');scale=nvalue(src,prefix+'_steadyState_scale')
                target=1/(1+math.exp(-(-10-mid)/scale));actual=x[prefix+'_q']
                ref=target+(actual[0]-target)*np.exp(-x['time_ms']/tau)
                error=float(np.max(np.abs(actual-ref)));near(error,entry['max_abs_state_error'][prefix+'_q']);errors.append(error)
                assert abs(actual[0]-target)>=.05
            assert entry['passed']==(max(errors)<=p['controls']['gates']['max_state_abs_error'] and max(abs(x['voltage_mV']+10))<=1e-6)
            control_checked+=1
            x,d=load('pool',variant,dt);entry=summary['analytic_controls'][f'pool_{variant}_{dt:g}']
            rho=d['pool_rho'];tau=d['pool_decay_ms'];forcing=-1e6*rho*x['ica_mA_cm2']
            near(x['ica_mA_cm2'],x['source_ica_mA_cm2'])
            near(x['pool_iCa_nA'][1:],-x['ica_mA_cm2'][1:]*1e6*d['area_um2']*1e-8)
            # Piecewise analytic solution as an independent interval recurrence;
            # uses the ACTUALLY measured density for each numerical interval.
            ref=np.empty(len(forcing));ref[0]=x['concentration_mM'][0]
            algorithm=np.empty_like(ref);exposed=np.empty_like(ref);algorithm[0]=ref[0];exposed[0]=x['cai_mM'][0]
            for i in range(1,len(ref)):
                steady=forcing[i]*tau
                ref[i]=steady+(ref[i-1]-steady)*math.exp(-dt/tau)
                if variant=='original':raw=algorithm[i-1]+dt*(forcing[i]-algorithm[i-1]/tau)
                else:raw=steady+(algorithm[i-1]-steady)*math.exp(-dt/tau)
                exposed[i]=raw;algorithm[i]=max(raw,0.)
            mask=x['time_ms']<=100
            err=max(abs(x['concentration_mM'][mask]-ref[mask]));near(err,entry['STATE_max_abs_error_mM'])
            assert entry['STATE_analytic_passed']==bool(err<=1e-15)
            near(x['concentration_mM'],algorithm);near(x['cai_mM'],exposed)
            pool_recurrence_errors[f'{variant}_{dt:g}']=dict(max_STATE_error_mM=float(max(abs(x['concentration_mM']-algorithm))),
                max_cai_error_mM=float(max(abs(x['cai_mM']-exposed))))
            control_checked+=1
    gate_values=0;common_input=None;synapses={};powers=np.array([2,1,4,1,1])
    for variant in p['variants']:
        sets={}
        for dt in p['dt_ms']:
            x,d=load('main',variant,dt);sets[dt]=x
            if common_input is None:common_input=x['peer_at_record_mV'].copy()
            else:assert np.array_equal(common_input,x['peer_at_record_mV'])
            if dt not in synapses:synapses[dt]=x['synapse_s'].copy()
            else:assert np.array_equal(synapses[dt],x['synapse_s'])
            m=summary['main_metrics'][f'{variant}_{dt:g}']
            near(max(abs(x['cai_mM']-x['concentration_mM'])),m['STATE_cai_sync_max_abs_mM'])
            cache=np.stack([x['cache_'+k] for k in 'efpqn'],axis=1)
            prev=np.stack([x['previous_'+k] for k in 'efpqn'],axis=1)
            cache_delta=np.max(abs(cache[1:]-prev[1:]**powers))
            assert abs(cache_delta-m['cache']['max_abs_to_recorded_previous_neural_state'])<=5*np.finfo(float).eps
        for co,fi in [(.025,.0125),(.0125,.00625),(.025,.00625)]:
            a=sets[co];b=sets[fi];mask=a['time_ms']>=4000;av=a['voltage_mV'][mask];bv=b['voltage_mV'][mask]
            span=lambda v:np.quantile(v,.95)-np.quantile(v,.05)
            values=dict(voltage_RMS_mV=norm(av-bv),voltage_mean_abs_mV=abs(np.mean(av)-np.mean(bv)),
                voltage_amplitude_relative=abs(span(av)-span(bv))/max(span(bv),1))
            limits=dict(voltage_RMS_mV=.5,voltage_mean_abs_mV=.2,voltage_amplitude_relative=.05,gate_state_RMS=.01)
            for field,label,floor in [('concentration_mM','STATE_calcium',1e-9),('cai_mM','cai',1e-9),('drive_cai','drive',.001)]:
                aa=a[field][mask];bb=b[field][mask];den=max(span(bb),floor)
                values[label+'_scaled_RMS']=norm(aa-bb)/den
                sa=np.array([aa.mean(),aa.std(),*np.quantile(aa,[.05,.95])]);sb=np.array([bb.mean(),bb.std(),*np.quantile(bb,[.05,.95])])
                values[label+'_scaled_stats']=float(max(abs(sa-sb))/den)
                limits[label+'_scaled_RMS']=.05;limits[label+'_scaled_stats']=.05
            values['gate_state_RMS']=max(norm(a[k+'_q'][mask]-b[k+'_q'][mask]) for k in 'efpqn')
            c=summary['paired_step_comparisons'][f'{variant}_{co:g}_vs_{fi:g}']
            for k,value in values.items():
                near(value,c['errors'][k]['value']);near(limits[k],c['errors'][k]['limit'])
                assert c['gates'][k]==bool(value<=limits[k]+1e-12)
                gate_values+=1
            assert c['passed']==all(c['gates'].values())
    assert len(list((ROOT/'runs').glob('main_*/record.json')))==12
    assert len(list((ROOT/'runs').glob('gates_*/record.json')))==6
    assert len(list((ROOT/'runs').glob('pool_*/record.json')))==6
    old=summary['paired_step_comparisons']['original_0.0125_vs_0.00625']['errors']
    new=summary['paired_step_comparisons']['both_0.0125_vs_0.00625']['errors']
    for k,d in summary['combined_over_original_improvement'].items():
        near(d['ratio'],new[k]['value']/old[k]['value'])
        assert d['strong_improvement']==bool(new[k]['value']<=.5*old[k]['value'] or old[k]['value']<=.1*old[k]['limit'])
    recommend=summary['checks']['code_update_forms_passed'] and summary['checks']['combined_analytic_STATE_controls_passed'] and summary['combined_main_last_pair_passed'] and all(d['strong_improvement'] for d in summary['combined_over_original_improvement'].values())
    assert summary['recommend_next_full_system_validation']==recommend
    result=dict(independent_audit_passed=True,protected_full_source_files_checked=files,analytic_controls_checked=control_checked,
        main_conditions_checked=12,paired_gate_values_checked=gate_values,actual_input_and_synapse_identity_checked=True,
        measured_current_pool_recurrence_and_clip_cai_order_checked=pool_recurrence_errors,
        main_analyzer_imported=False,NEURON_imported=False,summary_sha256=sha(ROOT/'summary.json'))
    write=ROOT/'independent-validation.json';write.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(independent_audit_passed=True,controls=control_checked,main=12,gate_values=gate_values)))


if __name__=='__main__':main()
