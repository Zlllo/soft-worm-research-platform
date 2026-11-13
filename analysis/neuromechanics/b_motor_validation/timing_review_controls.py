"""Independent compiled-function and raw-control audit; never load a simulator."""
import json
import math
from pathlib import Path
import re
import numpy as np
from timing_review_static import ROOT, REPO, A, sha, write, block, number, cppfunc

EXPECTED='8b9d11bda03f952c96d64d5b2b58f92496ecdcc008e5c471a51463664320f12a'
VARIANTS=['euler_cached','exp_cached','euler_direct','exp_direct']
GATES=['e','f','p','q','n']
def read(p):return json.loads(p.read_text())
def compact(s):return re.sub(r'\s+','',s)
def longest(mask):
    best=run=0
    for x in mask:
        run=run+1 if x else 0;best=max(best,run)
    return best
def mixed(pred,tol):return np.maximum(tol['formula_current_abs_mA_cm2'],tol['formula_current_relative']*np.abs(pred))

def main():
    result=dict(status='in_progress',protocol_sha256=EXPECTED,independent_of_A_analysis=True,
        new_simulations=0,new_compilations=0,blocking_errors=[],review_tool_errors=[],nonblocking_findings=[],generated={},zero_step={},short_K_fast={},raw_sources={})
    def check(ok,label,context=None):
        if not bool(ok):result['blocking_errors'].append(dict(check=label,context=context))
    try:
        p=read(A/'protocol.json');tol=p['control_tolerances'];lock=read(A/'control-execution-lock.json')
        check(sha(A/'protocol.json')==EXPECTED,'protocol lock')
        check(sha(REPO/'analysis/neuromechanics/b_motor_validation/timing_control_run.py')==lock['runner_sha256'],'runner fingerprint')
        check(sha(A/'generated-code-validation-v2.json')==lock['generated_code_validation_sha256'],'compiled evidence fingerprint')
        # Inspect actual files and locked binaries, not A's claimed pass booleans.
        kernels={v:(A/'frozen'/('original' if v.startswith('euler') else 'both') if v.endswith('cached') else A/'proposed-variants'/v) for v in VARIANTS}
        for v,folder in kernels.items():
            for rel,digest in lock['kernel_sha256'][v].items():check(sha(folder/rel)==digest,'locked kernel fingerprint',dict(variant=v,file=rel))
        replacement={
            'ca_boyle':('fopen0=e_fcond*f_fcond*h_fcond;','fopen0=(pow(e_q,e_instances))*(pow(f_q,f_instances))*h_fcond;'),
            'k_fast':('fopen0=p_fcond*q_fcond;','fopen0=(pow(p_q,p_instances))*(pow(q_q,q_instances));'),
            'k_slow':('fopen0=n_fcond;','fopen0=pow(n_q,n_instances);')}
        reverse={}
        for v in ['euler_direct','exp_direct']:
            parent='euler_cached' if v.startswith('euler') else 'exp_cached';details={}
            for mech in ['ca_boyle','k_fast','k_slow','CaPool','neuron_to_neuron_exc_syn','neuron_to_neuron_inh_syn','neuron_to_neuron_elec_syn','Leak']:
                old=(kernels[parent]/'arm64'/f'{mech}.cpp').read_text();new=(kernels[v]/'arm64'/f'{mech}.cpp').read_text()
                checked=[]
                for fn,kind in [('states','int'),('rates','int'),('initmodel','void'),('nrn_init','void'),('nrn_state','void'),('nrn_cur','void'),('nrn_jacob','void')]:
                    pattern=r'(?:static )?'+kind+r'\s+'+fn+r'\s*\([^;\n]*\)\s*\{'
                    if not re.search(pattern,old):continue
                    check(compact(block(old,pattern))==compact(block(new,pattern)),'non-current physical function unchanged',dict(variant=v,mechanism=mech,function=fn));checked.append(fn)
                pattern=r'static double\s+_nrn_current\s*\([^;\n]*\)\s*\{'
                if re.search(pattern,old):
                    before=compact(block(old,pattern));after=compact(block(new,pattern));expected=before
                    if mech in replacement:
                        a,b=replacement[mech];check(before.count(a)==1,'single current expression replacement',dict(variant=v,mechanism=mech));expected=before.replace(a,b)
                        check('rates(' not in after,'no rates refresh in current',dict(variant=v,mechanism=mech))
                        curwrapper=compact(cppfunc(new,'nrn_cur','void'))
                        check(curwrapper.index('_g_local=_nrn_current')<curwrapper.index('_rhs=_nrn_current'),'numeric voltage probe order',dict(variant=v,mechanism=mech))
                    check(after==expected,'current body only declared change',dict(variant=v,mechanism=mech))
                details[mech]=dict(noncurrent_functions_compared=checked,generated_sha256=sha(kernels[v]/'arm64'/f'{mech}.cpp'))
            result['generated'][v]=details
        for v,folder in kernels.items():
            c=(folder/'arm64/ca_boyle.cpp').read_text();k=(folder/'arm64/k_fast.cpp').read_text()
            check(re.search(r'#define eca\s+_ml->template fpfield',c) is not None,'Ca reversal is a private mechanism field',v)
            check('_ion_eca' not in cppfunc(c,'_nrn_current','double'),'Ca current does not read ion reversal',v)
            reverse[v]=dict(ca=number(compact(cppfunc(c,'initmodel','void')),'eca'),k=number(compact(cppfunc(k,'initmodel','void')),'ek'))
            check(reverse[v]==dict(ca=40.,k=-60.),'frozen effective channel reversals',v)
        result['effective_mechanism_reversals_mV']=reverse
        all_data={};records={};metadata_mismatch=[]
        expected_names=[f'zero_{v}_v{voltage}' for v in VARIANTS for voltage in [-20,-10]]+[f'short_{v}_v-10' for v in VARIANTS]
        check({q.name for q in (A/'controls').iterdir() if q.is_dir()}==set(expected_names),'exact twelve raw control directories')
        for name in expected_names:
            folder=A/'controls'/name;r=read(folder/'record.json');records[name]=r
            check(r['complete'] and r['failure'] is None,'raw control completed',dict(name=name,failure=r['failure']))
            check(sha(folder/'trace.npz')==r['trace_sha256'],'trace SHA',name)
            check(r['spec']['protocol_sha256']==EXPECTED and r['spec']['execution_lock_sha256']==sha(A/'control-execution-lock.json'),'run provenance',name)
            check(r['spec']['runner_sha256']==lock['runner_sha256'],'run runner SHA',name)
            check(r['physical_observer_hooks']==0,'no observers',name)
            meta=r['metadata'];check(meta['secondorder']==0 and meta['cvode_active']==0 and meta['actual_dt_ms']==.025,'runtime method and dt',name)
            with np.load(folder/'trace.npz') as z:
                a=z['data'].copy();cols=z['columns'].tolist();phase=z['phase'].copy();step=z['step'].copy()
            check(np.isfinite(a).all() and len(a)==r['rows'],'finite rows',name)
            x={key:a[:,j] for j,key in enumerate(cols)};all_data[name]=(x,phase,step)
            result['raw_sources'][name]=dict(trace_sha256=r['trace_sha256'],record_sha256=sha(folder/'record.json'),spec_sha256=sha(folder/'spec.json'))
            v=r['spec']['variant']
            if meta['reversal_mV']['ca']!=reverse[v]['ca']:
                metadata_mismatch.append(dict(name=name,recorded_ion_eca_mV=meta['reversal_mV']['ca'],effective_channel_eca_mV=reverse[v]['ca']))
            check(np.max(abs(x['h_fcond']-(1+(x['h_q']-1)*meta['h_alpha'])))<1e-12,'h_fcond algebra preserved',name)
            # INITIAL private eca, not seg.eca, governs the actual density current.
            mode='state' if v.endswith('direct') else 'cache'
            def formula(index,which):
                gate={g:(x[g][index]**meta['gate_parameters'][g]['power'] if which=='state' else x['cache_'+g][index]) for g in GATES}
                V=x['voltage_mV'][index];h=x['h_fcond'][index]
                return np.array([meta['gmax_S_cm2']['ca_boyle']*gate['e']*gate['f']*h*(V-reverse[v]['ca']),
                    meta['gmax_S_cm2']['k_fast']*gate['p']*gate['q']*(V-reverse[v]['k']),
                    meta['gmax_S_cm2']['k_slow']*gate['n']*(V-reverse[v]['k'])])
            currents=lambda i:np.array([x['I_ca_mA_cm2'][i],x['I_fast_mA_cm2'][i],x['I_slow_mA_cm2'][i]])
            native_error=np.abs(currents(0)-formula(0,mode))
            check(np.all(native_error<=mixed(formula(0,mode),tol)),'native finitialize current formula',name)
            if r['spec']['kind']=='zero':
                check(list(phase)==['native_finitialize_return','algebra_before_explicit_fcurrent','algebra_after_explicit_fcurrent'],'zero phases',name)
                check(len(a)==3 and r['executed_fadvance']==0 and r['explicit_fcurrent_calls']==1,'zero execution count',name)
                invariant_fields=['time_ms','voltage_mV','concentration_mM','cai_mM','h_q','h_fcond']+GATES+['cache_'+g for g in GATES]
                changes={key:float(abs(x[key][2]-x[key][1])) for key in invariant_fields}
                for key,error in changes.items():
                    limit=0. if key=='time_ms' else tol['zero_step_concentration_mM'] if key in ['concentration_mM','cai_mM'] else tol['zero_step_voltage_mV'] if key=='voltage_mV' else tol['zero_step_STATE_and_cache_abs']
                    check(error<=limit,'zero-step invariant',dict(name=name,field=key,error=error,limit=limit))
                check(np.array_equal(np.array([x[g][1] for g in GATES]),[.8,.2,.7,.1,.6]),'intentional zero-step STATE values',name)
                check(np.array_equal(np.array([x['cache_'+g][1] for g in GATES]),[.11,.61,.08,.77,.19]),'intentional distinct cache powers',name)
                chosen=formula(2,mode);opposite=formula(2,'cache' if mode=='state' else 'state');actual=currents(2)
                err=abs(actual-chosen);alternative=abs(actual-opposite);separation=abs(chosen-opposite)
                check(np.all(err<=mixed(chosen,tol)),'zero-step per-channel formula',dict(name=name,error=err.tolist()))
                check(np.all(separation>5*mixed(chosen,tol)),'zero-step formula hypotheses distinguishable',name)
                check(abs(x['sum_ica_mA_cm2'][2]-actual[0])<1e-12,'ionic Ca buffer equals sole Ca current',name)
                check(abs(x['sum_ik_mA_cm2'][2]-actual[1:].sum())<1e-12,'ionic K buffer equals two individual currents',name)
                result['zero_step'][name]=dict(invariants_max_change=max(changes.values()),per_field_changes=changes,
                    current_after_mA_cm2=actual.tolist(),matching_formula_error_mA_cm2=err.tolist(),
                    opposite_formula_error_mA_cm2=alternative.tolist(),implied_Ca_reversal_mV=float(x['voltage_mV'][2]-actual[0]/x['gion_ca_S_cm2'][2]),
                    fcurrent_current_change_mA_cm2=(actual-currents(1)).tolist(),native_current_mA_cm2=currents(0).tolist(),
                    native_formula_error_mA_cm2=native_error.tolist(),chosen_formula=mode)
            else:
                check(r['executed_fadvance']==200 and r['explicit_fcurrent_calls']==0 and len(a)==402,'short execution counts',name)
                pre=np.flatnonzero(phase=='before_fadvance');post=np.flatnonzero(phase=='after_fadvance')
                check(np.array_equal(step[pre],np.arange(1,201)) and np.array_equal(step[post],np.arange(1,201)),'all short steps saved',name)
                check(np.max(abs(x['voltage_mV'][np.r_[pre,post]]+10))<=tol['voltage_hold_mV'],'actual short voltage hold',name)
                check(np.max(abs(x['time_ms'][post]-np.arange(1,201)*.025))<1e-10,'short time clock',name)
                check(meta['gmax_S_cm2']==dict(ca_boyle=0.,k_fast=.0001,k_slow=0.),'K-fast only density',name)
                gp=meta['gate_parameters']['p'];gq=meta['gate_parameters']['q'];V=-10.;dt=.025
                p_inf=gp['rate']/(1+math.exp(-(V-gp['midpoint_mV'])/gp['scale_mV']))
                q_inf=gq['rate']/(1+math.exp(-(V-gq['midpoint_mV'])/gq['scale_mV']))
                j=np.arange(1,201);decay=math.log1p(-dt/gp['tau_ms']) if v.startswith('euler') else -dt/gp['tau_ms']
                closed=lambda n:p_inf+(.7-p_inf)*np.exp(n*decay)
                p_error=float(abs(x['p'][post]-closed(j)).max());q_error=float(abs(x['q'][post]-q_inf).max())
                check(p_error<=tol['short_gate_discrete_recurrence_max_abs'],'p discrete closed form',dict(name=name,error=p_error))
                check(q_error<=tol['short_constant_q_max_abs'],'q remains q_inf',dict(name=name,error=q_error))
                check(abs(x['p'][1]-.7)<1e-15 and abs(x['cache_p'][1]-.7**4)<1e-15 and abs(x['q'][1]-q_inf)<1e-15 and abs(x['cache_q'][1]-q_inf)<1e-15,'one initial aligned cache',name)
                G=meta['gmax_S_cm2']['k_fast'];drive=x['voltage_mV'][pre]-reverse[v]['k'];actual=x['I_fast_mA_cm2'][post]
                from_pre_state=G*x['p'][pre]**gp['power']*x['q'][pre]**gq['power']*drive
                from_pre_cache=G*x['cache_p'][pre]*x['cache_q'][pre]*drive
                selected=from_pre_state if mode=='state' else from_pre_cache
                formula_error=float(abs(actual-selected).max());check(np.all(abs(actual-selected)<=mixed(selected,tol)),'short actual pre-step formula',dict(name=name,error=formula_error))
                normal=G*closed(j-1)**gp['power']*q_inf**gq['power']*drive
                extra=G*closed(np.maximum(j-2,0))**gp['power']*q_inf**gq['power']*drive
                endpoint=G*closed(j)**gp['power']*q_inf**gq['power']*drive
                prediction=normal if mode=='state' else extra;other=extra if mode=='state' else normal
                fit=abs(actual-prediction)<=mixed(prediction,tol)
                informative=(j>=2)&fit&(abs(normal-extra)>5*mixed(prediction,tol))&(abs(actual-other)>5*mixed(prediction,tol))
                run=longest(informative);check(run>=5,'persistent state-generation discrimination',dict(name=name,longest=run))
                check(abs(actual[0]-normal[0])<=mixed(normal[:1],tol)[0],'first current equal after aligned caches',name)
                cache_error=float(abs(x['cache_p'][post]-x['p'][pre]**gp['power']).max())
                check(cache_error<1e-12,'post cache equals pre-state power',name)
                check(np.max(abs(x['concentration_mM']))==0 and np.max(abs(x['cai_mM']))==0,'CaPool stays unforced at zero in K-only control',name)
                result['short_K_fast'][name]=dict(p_closed_form_max_error=p_error,q_constant_max_error=q_error,
                    pre_step_formula_max_error_mA_cm2=formula_error,normal_generation_formula_max_error_mA_cm2=float(abs(actual-normal).max()),
                    extra_generation_formula_max_error_mA_cm2=float(abs(actual-extra).max()),endpoint_formula_max_error_mA_cm2=float(abs(actual-endpoint).max()),
                    chosen_generation='p_(j-1)' if mode=='state' else 'p_(j-2), first step p0',
                    informative_steps=int(informative.sum()),longest_consecutive_informative_steps=run,
                    post_cache_previous_state_power_max_error=cache_error,first_current_mA_cm2=float(actual[0]),
                    all_200_steps=dict(step=j.tolist(),actual_current_mA_cm2=actual.tolist(),normal_current_mA_cm2=normal.tolist(),extra_current_mA_cm2=extra.tolist(),informative=informative.tolist()))
        result['nonblocking_findings'].append(dict(id='C1',kind='metadata_field_misidentification',blocking=False,
            field='record.metadata.reversal_mV.ca',observed=metadata_mismatch,
            explanation='This records ionic seg.eca (Nernst buffer at zero cai), not the private ca_boyle eca used in generated current. Actual model private field is initialized to40mV and not read from _ion_eca. Both test voltages independently imply40mV from I/gion. Raw arrays need no rerun or alteration; analyses must use the effective mechanism reversal and preserve the metadata discrepancy.',
            analysis_gate='Any formula report using this field as channel ECa must be corrected; this does not invalidate the raw controls.'))
        result['initialization_comparison']={}
        for alg in ['euler','exp']:
            key0=f'zero_{alg}_cached_v-10';key1=f'zero_{alg}_direct_v-10'
            xa,_,_=all_data[key0];xb,_,_=all_data[key1]
            states_equal=all(xa[g][0]==xb[g][0] for g in GATES)
            check(states_equal,'native STATE initialization same across readout',alg)
            result['initialization_comparison'][alg]=dict(STATE_values_equal=states_equal,
                cached_native_current_mA_cm2=result['zero_step'][key0]['native_current_mA_cm2'],
                direct_native_current_mA_cm2=result['zero_step'][key1]['native_current_mA_cm2'],
                meaning='Unchanged STATE initialization does not mean unchanged initial current; keep this as part of the main readout total effect.')
        result['limits']=['Fixed-voltage K-only control identifies one additional gate-state generation in cached K-fast readout; not a full membrane/biological latency measurement.',
            'Zero-step proves actual arithmetic dependency and fcurrent assigned-buffer changes; the process never advances afterward.',
            'h_fcond remains untouched; CaPool, h(cai), NMJ timing and6 new main outcomes are not validated by these controls.',
            'No claim of main response improvement or full-system convergence; these data are only the pre-main gates.']
        result['provenance']=dict(control_execution_lock_sha256=sha(A/'control-execution-lock.json'),runner_sha256=lock['runner_sha256'],
            generated_validation_sha256=sha(A/'generated-code-validation-v2.json'),independent_script_sha256=sha(Path(__file__)))
    except Exception as exc:
        import traceback
        result['review_tool_errors'].append(dict(check='review could not complete; not a scientific-control failure',exception=str(exc),traceback=traceback.format_exc()))
    result['status']='incomplete' if result['review_tool_errors'] else 'passed' if not result['blocking_errors'] else 'failed'
    result['control_acceptance_passed']=None if result['review_tool_errors'] else not result['blocking_errors']
    result['metadata_correction_required_in_analysis']=bool(result['nonblocking_findings'])
    result['original_failures_not_removed']=True
    result['review_tool_history']=dict(saved_attempt='control-review-attempts/negative-scalar-parser-before-fix.json',
        resolution='Generated C++ writes the negative literal as - 60.0. Canonicalizing whitespace fixes the independent parser; no scientific run was repeated or changed.')
    if result['control_acceptance_passed']:
        result['summary_metrics']=dict(zero_invariant_max_change=max(x['invariants_max_change'] for x in result['zero_step'].values()),
            zero_current_formula_max_error_mA_cm2=max(max(x['matching_formula_error_mA_cm2']) for x in result['zero_step'].values()),
            inferred_private_ECa_max_error_mV=max(abs(x['implied_Ca_reversal_mV']-40) for x in result['zero_step'].values()),
            short_p_closed_form_max_error=max(x['p_closed_form_max_error'] for x in result['short_K_fast'].values()),
            short_current_pre_step_formula_max_error_mA_cm2=max(x['pre_step_formula_max_error_mA_cm2'] for x in result['short_K_fast'].values()),
            minimum_consecutive_informative_steps=min(x['longest_consecutive_informative_steps'] for x in result['short_K_fast'].values()))
    write(ROOT/'control-review.json',result)
    if result['control_acceptance_passed']:
        m=result['summary_metrics']
        lines=['# v2 控制独立验收', '',
            '实际生成函数及12个原始控制验收通过；未发现阻断控制结论的错误。复核未调用 A 的分析函数，未导入 NEURON、编译或运行仿真。', '',
            f"协议 SHA：`{EXPECTED}`。生成 C++ 只在三处电流表达式改为五个真 STATE 的幂；状态更新、rates、初始化、CaPool、NMJ 和 h_fcond 保持原样。实际 nrn_cur 的数值电压探测为 v+0.001 后 v，因此最终逐机制电流暴露值对应最后的 v 求值。", '',
            f"8个零步控制的时间、V、五个 STATE、五缓存、h 和浓度不变量最大变化为 {m['zero_invariant_max_change']:.6g}。实际逐通道公式最大误差为 {m['zero_current_formula_max_error_mA_cm2']:.6g} mA/cm²，所有缓存/直接读出假设均可区分。一次显式 fcurrent 改变了电流与离子缓冲暴露值；这些实例没有再推进。", '',
            f"4个快K控制的 p 离散闭式最大误差为 {m['short_p_closed_form_max_error']:.6g}；按真实步前状态/缓存和步前 V 重算电流的最大误差为 {m['short_current_pre_step_formula_max_error_mA_cm2']:.6g} mA/cm²。第一步因初始缓存对齐而相同。其后每条控制均连续 {m['minimum_consecutive_informative_steps']} 步判别：直接读出对应 p_(j−1)，缓存读出对应 p_(j−2)。直接读出与刚推进的 p_j 不同，属于正常的先求电流、后推状态；这里确认的是缓存额外引用上一代门控状态，而不是整个系统统一的物理延迟。", '',
            '| 控制 | 选中公式 | p闭式最大误差 | 连续可判别步数 |',
            '|---|---|---:|---:|']
        for name,x in result['short_K_fast'].items():lines.append(f"| {name} | {x['chosen_generation']} | {x['p_closed_form_max_error']:.6g} | {x['longest_consecutive_informative_steps']} |")
        lines += ['', '## 一处非阻断元数据问题', '',
            f"原 record.metadata.reversal_mV.ca 为离子层 seg.eca（零 cai 时10⁶ mV），并非 ca_boyle 使用的局部反转电位。实际生成代码将私有 eca 存在独立 fpfield 并初始化为40 mV，当前函数不读取 _ion_eca。8个零步结果用 V−Ica/gion 独立反推40 mV，最大差 {m['inferred_private_ECa_max_error_mV']:.6g} mV。Ca公式因此使用经源码与原始电流共同验证的私有40 mV，不用错误对应的离子字段。原 metadata 与原始迹线保持不变，不需要重跑；后续分析与报告必须明确字段含义。", '',
            '自然初始化时，缓存版各目标离子电流为零，而直接版初始Ca电流约−2.58489×10⁻⁷ mA/cm²、慢K约1.47781×10⁻⁴ mA/cm²；五个 STATE 初值相同。这是读出处理的真实初始化效应，不能从主结果中隐去。h_fcond 的变换在保存数据中成立，h/cai刷新没有混入控制。', '',
            '本验收仅支持规定的门控电流依赖机制。6条新主条件、误差改善、Ca/h/NMJ其他时序、完整302/身体收敛均未据此验收；后续主条件由主线另行接续。', '',
            '独立脚本：`analysis/neuromechanics/b_motor_validation/timing_review_controls.py`。机器可读证据、全部200步预测与实际值、原始SHA在同目录 `control-review.json`。第一次独立解析器失败已保存在 `control-review-attempts/`，属于复核工具解析空格的问题，已修正；没有删除或重跑任何科学控制。']
        (ROOT/'control-review.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],blocking_errors=result['blocking_errors'],zero_controls=len(result['zero_step']),short_controls=len(result['short_K_fast']),nonblocking_findings=len(result['nonblocking_findings']),new_simulations=0),ensure_ascii=False))
    if result['blocking_errors'] or result['review_tool_errors']:raise SystemExit(1)

if __name__=='__main__':main()
