"""Independent raw 2x2 main-data audit, metrics and phase-safe current reconstruction."""
import ast
import json
import math
from pathlib import Path
import numpy as np
from timing_review_static import ROOT, REPO, A, sha, write, cppfunc, number

EXPECTED='8b9d11bda03f952c96d64d5b2b58f92496ecdcc008e5c471a51463664320f12a'
VARIANTS=['euler_cached','exp_cached','euler_direct','exp_direct']
STEPS=[.025,.0125,.00625]
GATES=['e_q','f_q','p_q','q_q','n_q']
FIELDS=['voltage_mV',*GATES,'concentration_mM','cai_mM','drive_cai','drive_STATE','ica_mA_cm2','synapse_i_nA']
def read(p):return json.loads(p.read_text())
def rms(x):return float(np.sqrt(np.mean(np.square(x))))
def amplitude(x):return float(np.quantile(x,.95)-np.quantile(x,.05))
def stats(x):return np.array([np.mean(x),np.std(x),np.quantile(x,.05),np.quantile(x,.95)])
def assign_literal(tree,name):return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
def source_audit():
    path=REPO/'analysis/neuromechanics/b_motor_validation/timing_main_run.py';text=path.read_text();tree=ast.parse(text)
    legacy=ast.parse((A/'legacy-code/integration_run.py').read_text())
    assert assign_literal(tree,'COLUMNS')==assign_literal(legacy,'COLUMNS')
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    forbidden=[ast.unparse(n.func) for n in calls if isinstance(n.func,ast.Attribute) and (n.func.attr=='fcurrent' or n.func.attr.startswith('rates') or n.func.attr in ['extra_scatter_gather','play','record','NetCon'])]
    assert not forbidden
    attrs=lambda name:sum(isinstance(n.func,ast.Attribute) and n.func.attr==name for n in calls)
    assert attrs('FInitializeHandler')==2 and attrs('finitialize')==1 and attrs('fadvance')==1
    for fn in [n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ['pre_snapshot','post_snapshot','gates','caches','record']]:
        for n in ast.walk(fn):
            assert not isinstance(n,(ast.AugAssign,ast.Delete)),fn.name
            if isinstance(n,ast.Assign):assert all(isinstance(target,ast.Name) for target in n.targets),fn.name
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):raise AssertionError((fn.name,ast.unparse(n.func)))
    return dict(runner_sha256=sha(path),old32_column_order_identical=True,forbidden_current_rates_observer_calls=[],
        native_initialization_handler_count=2,extra_handlers=0,snapshot_functions_passive_reads_only=True,
        note='Old main branch input assignments and record/advance order manually compared; removed non-main gate/pool cases, metadata and passive reads are the only runtime changes.')

def pair(a,b,acceptance,mask):
    values={};gates={}
    for key in FIELDS:
        x=a[key][mask];y=b[key][mask];values[key]=dict(RMS=rms(x-y),mean_difference=float(x.mean()-y.mean()),
            std_difference=float(x.std()-y.std()),q05_difference=float(np.quantile(x,.05)-np.quantile(y,.05)),
            q95_difference=float(np.quantile(x,.95)-np.quantile(y,.95)),max_abs=float(np.max(abs(x-y))))
    for field in ['concentration_mM','cai_mM','drive_cai','drive_STATE']:
        floor=acceptance['calcium_scale_floor_mM'] if 'mM' in field else acceptance['drive_scale_floor']
        scale=max(amplitude(b[field][mask]),floor)
        values[field].update(reference_q95_q05_or_floor=scale,scaled_RMS=values[field]['RMS']/scale,
            scaled_stats_max=float(np.max(abs(stats(a[field][mask])-stats(b[field][mask])))/scale))
    v=values['voltage_mV'];v['amplitude_relative']=abs(amplitude(a['voltage_mV'][mask])-amplitude(b['voltage_mV'][mask]))/max(amplitude(b['voltage_mV'][mask]),acceptance['voltage_amplitude_floor_mV'])
    gates['voltage_RMS_mV']=v['RMS']<=acceptance['voltage_RMS_mV']
    gates['voltage_mean_abs_mV']=abs(v['mean_difference'])<=acceptance['voltage_mean_abs_mV']
    gates['voltage_amplitude_relative']=v['amplitude_relative']<=acceptance['voltage_amplitude_relative']
    for field in ['concentration_mM','cai_mM','drive_cai']:
        tol=acceptance['calcium_scaled_stats_and_RMS'] if 'mM' in field else acceptance['drive_scaled_stats_and_RMS']
        gates[field+'_scaled_RMS']=values[field]['scaled_RMS']<=tol
        gates[field+'_scaled_stats']=values[field]['scaled_stats_max']<=tol
    gates['gate_state_RMS']=max(values[k]['RMS'] for k in GATES)<=acceptance['gate_state_RMS']
    return dict(metrics=values,gates=gates,passed=all(gates.values()),failed_gates=[k for k,v in gates.items() if not v])

def contrast(terms,weights,windows):
    out={}
    for window,mask in windows.items():
        fields={}
        for field in FIELDS:
            signal=sum(weight*d[field][mask] for weight,d in zip(weights,terms))
            fields[field]=dict(mean=float(signal.mean()),RMS=rms(signal),std=float(signal.std()),max_abs=float(abs(signal).max()))
        out[window]=fields
    return out

def main():
    result=dict(status='in_progress',protocol_sha256=EXPECTED,new_simulations=0,new_compilations=0,
        blocking_errors=[],review_tool_errors=[],raw_inputs={},current_reconstruction={},pairs={},factorial={},initialization={})
    def check(condition,name,detail=None):
        if not bool(condition):result['blocking_errors'].append(dict(check=name,detail=detail))
    try:
        p=read(A/'protocol.json');lock=read(A/'main-execution-lock.json');batch=read(A/'main-batch-status.json')
        check(sha(A/'protocol.json')==EXPECTED==lock['protocol_sha256'],'protocol identity')
        audit=source_audit();check(audit['runner_sha256']==lock['runner_sha256'],'runner hash')
        check(sha(REPO/'analysis/neuromechanics/b_motor_validation/timing_main_batch.py')==lock['batch_script_sha256'],'batch source hash')
        check(batch['declared_new_conditions']==6 and len(batch['status'])==6 and batch['retries']==0 and all(x['exit_code']==0 and x['complete'] for x in batch['status'].values()),'strict six new main conditions complete/no retries')
        expected_new={f'{v}_dt{dt:g}' for v in ['euler_direct','exp_direct'] for dt in STEPS}
        check({q.name for q in (A/'main-runs').iterdir() if q.is_dir()}==expected_new,'exact new raw run set')
        check(lock['main_conditions_authorized']==6 and lock['retries_allowed']==lock['observers']==lock['explicit_fcurrent_calls']==lock['full302_body_authorized']==0,'execution scope')
        for v,files in lock['kernel_sha256'].items():
            for rel,digest in files.items():check(sha(A/'proposed-variants'/v/rel)==digest,'kernel SHA',dict(variant=v,path=rel))
        reused=read(A/'reused-main-sources.json');check(sha(A/'reused-main-sources.json')==lock['reused_sources_sha256'],'reuse manifest hash')
        check(len(reused)==6,'six reused source conditions')
        with np.load(A/'presynaptic-input.npz') as z:source=z['voltage_mV'].copy()
        check(sha(A/'presynaptic-input.npz')==lock['input_sha256']==p['main']['input_sha256'],'fixed input file')
        t=np.arange(320001)*.025;late=t>=4000
        windows={'0_0.1ms':(t>=0)&(t<.1),'0.1_1ms':(t>=.1)&(t<1),'1_10ms':(t>=1)&(t<10),'10_100ms':(t>=10)&(t<100),'late_4000_8000ms':late}
        datasets={};synapse_hashes={};initial={};metrics={}
        for variant in VARIANTS:
            for dt in STEPS:
                name=f'{variant}_dt{dt:g}';folder=A/('main-runs' if variant.endswith('direct') else 'reused-main')/name
                record=read(folder/'record.json');is_new=variant.endswith('direct')
                check(record['complete'] and record['failure'] is None,'raw condition complete',dict(name=name,failure=record['failure']))
                check(sha(folder/'trace.npz')==record['trace_sha256'],'trace SHA',name)
                check(record['rows']==320001 and record['actual_dt_ms']==dt and abs(record['actual_neural_clock_ms']-8000)<1e-5,'shape/dt/native clock',name)
                check(record['geometry']==dict(L_um=20.,diam_um=5.,cm_uF_cm2=1.) and record['versions']['neuron']=='9.0.2','geometry and version',name)
                if is_new:
                    check(record['spec']['protocol_sha256']==EXPECTED and record['spec']['runner_sha256']==lock['runner_sha256'] and record['spec']['execution_lock_sha256']==sha(A/'main-execution-lock.json'),'new run source/execution provenance',name)
                    check(record['explicit_fcurrent_calls']==record['manual_rates_calls']==record['observer_hooks']==record['new_initialization_handlers']==0,'no added evaluation/observer/init calls',name)
                    check(record['executed_fadvance']==round(8000/dt) and record['secondorder']==record['cvode_active']==0,'executed numerical method',name)
                    check(record['gmax_S_cm2']==dict(ca_boyle=.0007,k_fast=.0001,k_slow=.00059999997),'unchanged main conductances',name)
                else:
                    for file,field in [('trace.npz','trace_sha256'),('record.json','record_sha256'),('spec.json','spec_sha256')]:check(sha(folder/file)==reused[name][field],'reused file exact provenance',dict(name=name,file=file))
                with np.load(folder/'trace.npz') as z:
                    a=z['data'];columns=z['columns'].tolist()
                    check(columns==assign_literal(ast.parse((A/'legacy-code/integration_run.py').read_text()),'COLUMNS'),'old32 labels exactly preserved',name)
                    check(a.shape==(320001,32) and np.isfinite(a).all(),'all old32 raw fields finite',name)
                    c=lambda key:a[:,columns.index(key)]
                    check(np.array_equal(c('time_ms'),t),'common nominal sample clock',name)
                    check(np.array_equal(c('peer_at_record_mV'),source[::4]),'all320001 endpoint inputs exact',name)
                    ix=np.maximum(np.arange(320001)*4-round(dt/.00625),0)
                    check(np.array_equal(c('last_applied_peer_mV'),source[ix]),'all last-interval inputs exact',name)
                    check(np.all(c('commanded_ica_mA_cm2')==0) and np.all(c('source_ica_mA_cm2')==0),'no external Ca helper drive',name)
                    synapse_hashes[name]=sha_array(c('synapse_s'))
                    for ca,drive in [('cai_mM','drive_cai'),('concentration_mM','drive_STATE')]:
                        cc=np.maximum(c(ca),0);check(np.max(abs(c(drive)-.3*cc/(cc+2.5e-7)))<1e-14,'drive reconstruction',dict(name=name,source=ca))
                    datasets[name]={field:c(field).copy() for field in FIELDS}
                    initial[name]={field:float(c(field)[0]) for field in columns}
                    metrics[name]=dict(voltage_min_max_mV=[float(c('voltage_mV')[late].min()),float(c('voltage_mV')[late].max())],
                        late_V_mean_mV=float(c('voltage_mV')[late].mean()),late_calcium_mean_mM=float(c('cai_mM')[late].mean()),
                        late_drive_mean=float(c('drive_cai')[late].mean()),late_drive_SD=float(c('drive_cai')[late].std()),
                        STATE_calcium_min_mM=float(c('concentration_mM').min()),cai_min_mM=float(c('cai_mM').min()),
                        max_cai_STATE_difference_mM=float(abs(c('cai_mM')-c('concentration_mM')).max()))
                    if is_new:
                        extra=z['extras'];extra_columns=z['extra_columns'].tolist();ec=lambda key:extra[:,extra_columns.index(key)]
                        check(extra.shape==(320001,27) and np.isfinite(extra).all(),'all passive extra fields finite',name)
                        check(np.max(abs(ec('pre_last_time_ms')[1:]-(t[1:]-dt)))<1e-5,'native before-last-substep clock',name)
                        for gate in ['e','f','p','q','n']:check(np.array_equal(ec('pre_last_'+gate),c('previous_'+gate)),'old previous-gate column equals new passive pre-snapshot',dict(name=name,gate=gate))
                        # No endpoint V/state/h is used in this actual-current reconstruction.
                        ca_src=(A/'proposed-variants'/variant/'arm64/ca_boyle.cpp').read_text();fast_src=(A/'proposed-variants'/variant/'arm64/k_fast.cpp').read_text()
                        Eca=number(''.join(cppfunc(ca_src,'initmodel','void').split()),'eca');Ek=number(''.join(cppfunc(fast_src,'initmodel','void').split()),'ek')
                        E=ec('pre_last_e');F=ec('pre_last_f');P=ec('pre_last_p');Q=ec('pre_last_q');N=ec('pre_last_n');H=ec('pre_last_h_fcond');V=ec('pre_last_voltage_mV')
                        open_gate={'ca':E**2*F*H,'fast':P**4*Q,'slow':N}
                        gmax={'ca':.0007,'fast':.0001,'slow':.00059999997};reversal={'ca':Eca,'fast':Ek,'slow':Ek};errs={};post_hyp={}
                        for field in ['ca','fast','slow']:
                            predicted=gmax[field]*open_gate[field]*(V-reversal[field]);actual=ec('post_I_'+field+'_mA_cm2')
                            error=abs(actual-predicted);tolerance=np.maximum(p['control_tolerances']['formula_current_abs_mA_cm2'],p['control_tolerances']['formula_current_relative']*abs(predicted))
                            check(np.all(error<=tolerance),'actual direct current uses saved pre-step V/STATE/h',dict(name=name,channel=field,max_error=float(error.max())))
                            check(np.max(abs(ec('post_gion_'+field+'_S_cm2')-gmax[field]*open_gate[field]))<1e-12,'saved current-stage gion reconstructed',dict(name=name,channel=field))
                            check(np.max(abs(ec('post_fopen_'+field)-open_gate[field]))<1e-12,'saved current-stage fopen reconstructed',dict(name=name,channel=field))
                            errs[field]=dict(max_abs_error_mA_cm2=float(error.max()),RMS_error_mA_cm2=rms(error),current_RMS_mA_cm2=rms(actual),native_initial_current_mA_cm2=float(actual[0]))
                            if field=='ca':post_factor=c('e_q')**2*c('f_q')*c('cache_h')
                            elif field=='fast':post_factor=c('p_q')**4*c('q_q')
                            else:post_factor=c('n_q')
                            post_hyp[field]=rms(actual[1:]-gmax[field]*post_factor[1:]*(c('voltage_mV')[1:]-reversal[field]))
                        check(np.array_equal(ec('post_I_ca_mA_cm2'),c('ica_mA_cm2')),'per-channel Ca and ionic Ca raw views agree',name)
                        result['current_reconstruction'][name]=dict(passed=all(v['max_abs_error_mA_cm2']<=1e-12 for v in errs.values()),channels=errs,
                            unaligned_endpoint_formula_RMS_mA_cm2=post_hyp,meaning='Endpoint formula mismatch is expected staggering diagnostic, not an independent extra-delay estimate.',
                            effective_private_reversal_mV=dict(ca=Eca,k=Ek),instantaneous_ion_eca_range_mV=[float(ec('post_ion_eca_mV').min()),float(ec('post_ion_eca_mV').max())])
                    result['raw_inputs'][name]=dict(trace_sha256=record['trace_sha256'],record_sha256=sha(folder/'record.json'),spec_sha256=sha(folder/'spec.json'),new_run=is_new)
        for dt in STEPS:check(len({synapse_hashes[f'{v}_dt{dt:g}'] for v in VARIANTS})==1,'same-dt all four synapse s bitwise identical',dt)
        base=initial['euler_cached_dt0.025'];initial_fields=['voltage_mV','concentration_mM','cai_mM',*GATES,'cache_e','cache_f','cache_p','cache_q','cache_n','cache_h','h_q','h_inf','synapse_s']
        for name,values in initial.items():check(all(values[k]==base[k] for k in initial_fields),'native initial STATE/cache identity across all12',name)
        result['initialization']=dict(all12_initial_STATE_cache_values_identical=True,rows=initial,
            Ca_current_field='Existing old32 ica_mA_cm2: native source-preserving current, not a query.',
            individual_K_current_limit='Only new direct trajectories have actual per-channel K currents; no cached old K currents fabricated from endpoints.',
            interval_convention='Initial windows left-inclusive/right-exclusive to avoid overlapping boundaries',
            attribution_limit='Startup and continuing readout mechanisms are not decomposed by a time-window comparison.')
        result['metrics']=metrics;result['same_dt_synapse_s_sha256']=synapse_hashes
        for variant in VARIANTS:
            for coarse,fine in [(.025,.0125),(.0125,.00625),(.025,.00625)]:
                key=f'{variant}_{coarse:g}_vs_{fine:g}'
                result['pairs'][key]=pair(datasets[f'{variant}_dt{coarse:g}'],datasets[f'{variant}_dt{fine:g}'],p['acceptance'],late)
        for dt in STEPS:
            ec,xc,ed,xd=[datasets[f'{v}_dt{dt:g}'] for v in VARIANTS]
            result['factorial'][f'dt{dt:g}']={
                'euler_direct_minus_cached':contrast([ed,ec],[1,-1],windows),
                'exp_direct_minus_cached':contrast([xd,xc],[1,-1],windows),
                'cached_exp_minus_euler':contrast([xc,ec],[1,-1],windows),
                'direct_exp_minus_euler':contrast([xd,ed],[1,-1],windows),
                'interaction':contrast([xd,xc,ed,ec],[1,-1,-1,1],windows)}
        result['readout_improvement']={}
        for alg in ['euler','exp']:
            cached=result['pairs'][f'{alg}_cached_0.0125_vs_0.00625'];direct=result['pairs'][f'{alg}_direct_0.0125_vs_0.00625'];fields={}
            for field,metric,limit in [('voltage_mV','RMS',.5),('concentration_mM','scaled_RMS',.05),('cai_mM','scaled_RMS',.05),('drive_cai','scaled_RMS',.05)]:
                old=cached['metrics'][field][metric];new=direct['metrics'][field][metric];small=old<=limit/10
                fields[field]=dict(cached_error=old,direct_error=new,direct_over_cached=new/old if old else None,
                    observed_reduction=new<old,at_least_twofold=new<=.5*old,tiny_old_error_exception=small,predeclared_rule_passed=new<=.5*old or small)
            result['readout_improvement'][alg]=dict(fields=fields,direct_last_pair_passed=direct['passed'],
                all_improvement_rules_passed=all(d['predeclared_rule_passed'] for d in fields.values()))
        result['source_audit']=audit
        result['new_run_count']=6;result['reused_run_count']=6;result['paired_comparisons']=12
        result['scope_limits']=['Main initial current and continuing current-readout effects are not uniquely decomposed.',
            'Fine-step old cached data do not contain missing pre-current V/cache/h; no old actual K current or fine-call history reconstructed.',
            'A same-time interaction is formed on signals before RMS/statistics; it is not a difference of RMS values.',
            'Same-dt effects and dt-pair errors include the declared left-hold input convention; they do not establish a rigorous continuous-input solution error.',
            'h(cai), CaPool/NMJ timing, full302/body convergence, propulsion and biological effects remain unverified.']
        result['provenance']=dict(main_execution_lock_sha256=sha(A/'main-execution-lock.json'),main_batch_status_sha256=sha(A/'main-batch-status.json'),
            generated_code_validation_sha256=sha(A/'generated-code-validation-v2.json'),independent_script_sha256=sha(Path(__file__)),A_analysis_functions_used=False)
    except Exception as e:
        import traceback
        result['review_tool_errors'].append(dict(error=str(e),traceback=traceback.format_exc()))
    result['status']='incomplete' if result['review_tool_errors'] else 'passed_data_audit' if not result['blocking_errors'] else 'failed_data_audit'
    result['data_audit_passed']=None if result['review_tool_errors'] else not result['blocking_errors']
    result['A_summary_crosscheck']=dict(status='pending_A_summary_after_independent_calculation')
    write(ROOT/'main-review-calculations.json',result);write(ROOT/'main-review.json',result)
    print(json.dumps(dict(status=result['status'],errors=result['blocking_errors'],tool_errors=result['review_tool_errors'],pairs=len(result['pairs']),factorial_dt=len(result['factorial']),reconstruction=len(result['current_reconstruction'])),ensure_ascii=False))
    if result['review_tool_errors'] or result['blocking_errors']:raise SystemExit(1)

def sha_array(x):
    import hashlib
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

if __name__=='__main__':main()
