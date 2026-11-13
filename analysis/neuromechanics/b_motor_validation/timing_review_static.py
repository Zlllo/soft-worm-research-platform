"""Official-version call-order and algebra audit; no simulator import or execution."""
import hashlib
import json
import math
from pathlib import Path
import re

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'results/c1-current-timing-review-2026-11-10'
OLD=REPO/'results/c1-muscle-integration-isolation-2026-11-09'
A=REPO/'results/c1-current-timing-2026-11-10'
REPORT=REPO/'docs/reports/c1-current-timing-review-2026-11-10.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def block(text,pattern):
    m=re.search(pattern,text);assert m,pattern
    start=text.index('{',m.start());depth=0
    for i in range(start,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if not depth:return text[start+1:i]
    raise ValueError('unclosed block')
def number(text,name):return float(re.search(r'\b'+name+r'\s*=\s*([-+\d.Ee]+)',text).group(1))
def cppfunc(text,name,kind='int'):
    return block(text,r'(?:static )?'+kind+r'\s+'+name+r'\s*\([^;\n]*\)\s*\{')
def ordered(text,tokens):
    found=[text.index(x) for x in tokens];assert found==sorted(found),tokens

def main():
    ROOT.mkdir(exist_ok=True)
    f=(ROOT/'official-9.0.2/fadvance.cpp').read_text();tree=(ROOT/'official-9.0.2/treeset.cpp').read_text()
    mid=cppfunc(f,'nrn_fixed_step_thread','void');tail=cppfunc(f,'nrn_fixed_step_lastpart','void')
    ordered(mid,['nt._t += .5 * nt._dt','setup_tree_matrix(cache_token, nt)','nrn_solve(nth)','second_order_cur(nth)','nrn_update_voltage(cache_token, *nth)','nrn_fixed_step_lastpart(cache_token, nt)'])
    ordered(tail,['nth->_t += .5 * nth->_dt','nonvint(cache_token, nt)','fixed_record_continuous(cache_token, nt)'])
    assert 'memb_func[tml->index].state(sorted_token' in cppfunc(f,'nonvint','void')
    assert 'current(cache_token, _nt, tml->ml, tml->index)' in cppfunc(tree,'nrn_rhs','void')
    ordered(cppfunc(tree,'setup_tree_matrix','void'),['nrn_rhs(cache_token, nt)','nrn_lhs(cache_token, nt)'])
    fc=cppfunc(f,'fcurrent','void');assert 'setup_tree_matrix' in fc and 'nonvint(' not in fc
    sources={str(p.relative_to(REPO)):sha(p) for p in (ROOT/'official-9.0.2').iterdir() if p.is_file()}
    sources[str((REPO/'docs/c1-current-timing-2026-11-10.md').relative_to(REPO))]=sha(REPO/'docs/c1-current-timing-2026-11-10.md')
    local={}
    for variant in ['original','both']:
        local[variant]={}
        for mech,states in [('ca_boyle',['e_q','f_q']),('k_fast',['p_q','q_q']),('k_slow',['n_q'])]:
            p=OLD/'variants'/variant/'arm64'/f'{mech}.cpp';text=p.read_text();sources[str(p.relative_to(REPO))]=sha(p)
            state=cppfunc(text,'states');rates=cppfunc(text,'rates');cur=cppfunc(text,'_nrn_current','double')
            assert state.index('rates (')<min(state.index(s+' =') for s in states)
            assert 'rates (' not in cur
            for q in states:
                assert q[0]+'_fcond' in rates and q[0]+'_fcond' in cur
            local[variant][mech]=dict(state_updates_after_rates=True,current_uses_stored_powers=True,current_calls_rates=False,
                current_cpp_line=next(i+1 for i,l in enumerate(text.splitlines()) if 'static double _nrn_current' in l))
    ca=(OLD/'frozen/ca_boyle.mod').read_text();stateblock=block(ca,r'\bSTATE\s*\{')
    assert 'h_q' not in stateblock and 'h_q = h_inf' in ca and 'h_fcond = 1 +(( h_q -1) *  h_alpha )' in ca
    alpha=number(ca,'h_alpha');hk=number(ca,'h_k');half=number(ca,'h_ca_half')
    hvalues=[]
    for calcium in [0.,half,2*half]:
        hq=1/(1+math.exp((half-calcium)/hk));hvalues.append(dict(cai_mM=calcium,h_q=hq,h_fcond=1+(hq-1)*alpha))
    syn=OLD/'variants/original/arm64/neuron_to_neuron_exc_syn.cpp';st=syn.read_text();sc=cppfunc(st,'_nrn_current','double')
    assert '_current += i' in sc and 'rates (' not in sc
    assert 'i = - 1.0 * weight * conductance * s * ( erev - v )' in cppfunc(st,'rates')
    syn_note=dict(current_is_also_cached_in_rates=True,numeric_voltage_probe_does_not_recompute_i=True,
        saturation_branch_can_assign_s_in_current='s = inf' in sc,
        scope='preserve NMJ implementation; direct five gate powers does not repair all current dependencies',sha256=sha(syn))
    # A proposed K-only clamp control, evaluated by closed-form algebra only.
    k=(OLD/'frozen/k_fast.mod').read_text();v=-10.;ek=-60.;g=.0001
    inf=lambda q:1/(1+math.exp(-(v-number(k,q+'_steadyState_midpoint'))/number(k,q+'_steadyState_scale')))
    tau=number(k,'p_timeCourse_tau');pi=inf('p');qi=inf('q');p0=.7;expect=[]
    for dt in [.025,.0125,.00625]:
        for algorithm in ['Euler','exponential']:
            def gate(n):return pi+(p0-pi)*((1-dt/tau)**n if algorithm=='Euler' else math.exp(-n*dt/tau))
            current=lambda n:g*gate(n)**4*qi*(v-ek)
            expect.append(dict(dt_ms=dt,algorithm=algorithm,p0=p0,p_inf=pi,q_constant=qi,tau_ms=tau,
                initial_cache_condition='p_fcond=p0^4; q_fcond=q_inf, control only; no explicit fcurrent in trajectory',
                endpoint_predictions=[dict(step=n,time_ms=n*dt,p_after=gate(n),I_direct_mA_cm2=current(n-1),
                    I_cached_mA_cm2=current(max(n-2,0)),I_from_new_endpoint_state_mA_cm2=current(n)) for n in [1,2,3,4]]))
    contract=dict(factor_A='original Euler gate+CaPool versus combined exponential gate+CaPool (bundled integration factor)',
        factor_B='cached five STATE powers versus those same five current STATE powers, h_fcond preserved',
        reused_main=6,new_main_max=6,full302_or_body_runs=0,
        main_forbidden=['extra fcurrent after finitialize or in record/loop','rates() refresh in current path','h_q replacing h_fcond',
                        'refresh h(cai)','CaPool sync/clipping changes','manual cache synchronization in main','new initialization handlers'],
        main_required=['exact old source/input/geometry/NMJ/initialization/left-input convention hashes for reuse',
            'explicit secondorder=0 or assertion/record of unchanged default; no change to CVODE disabled mode',
            'STATE/cache/current/voltage snapshot phase labels and actual currents before any recomputation',
            'new direct traces should retain before-last-neural-substep V, five STATE values, cache powers and h_fcond, plus unchanged after-step per-channel currents; post-step h/V alone cannot reconstruct the actual earlier matrix current',
            'native t=0 current differences kept; startup is part of readout total effect',
            'same-dt identical prescribed input and synaptic s; unchanged non-density current cache',
            'report within-algorithm direct-minus-cache and the two-factor interaction, not only best version'],
        minimal_controls=[
            dict(name='zero-step algebra/current evaluation',design='use independent disposable instances, set distinct STATE and cache powers, freeze measured h_fcond; compare density current formula at two voltages; h.fcurrent only here; verify time/STATE/concentration unchanged and save all assigned-variable changes'),
            dict(name='short fixed-voltage K-fast control',design='4 variants at one declared dt, e.g.0.025ms for1..5ms; Ca/other currents off, -10mV clamp, p0=.7, q=q_inf; set cache powers consistent once in this control only; verify p closed form and one-step normal staggering versus extra cached-update offset; no explicit fcurrent between steps'),
            dict(name='h-boundary algebra',design='verify h_fcond=1+(h_q-1)*alpha on saved values and generated current retains h_fcond; h is not a dynamic STATE and is not refreshed')],
        limitations=['2x2 is sufficient for the bundled integration/readout total effects and interaction only',
            'main late-window effects do not uniquely decompose startup-current differences from ongoing readout differences',
            'short aligned-cache control isolates continuing dependency but cannot quantify its share in an 8s main trajectory',
            'normal engine splitting remains; h(cai), CaPool and NMJ timing are not repaired by five direct gate powers',
            'old .0125/.00625ms main traces are decimated to .025ms and do not preserve every before-current cache/V/h snapshot; use them for matched outcome metrics, not full fine-step current-call reconstruction',
            'paired dt differences are not a certified continuous-input error bound'])
    result=dict(status='static_review_complete_waiting_for_A_protocol_and_outputs',official_sources=sources,
        verified=dict(NEURON_version='9.0.2 from installed metadata/header, without importing NEURON',
            engine_order='half time -> currents/matrix -> voltage update -> remaining half time -> state callbacks -> recording',
            default_secondorder=0,secondorder_two_time_accuracy_labels_must_not_be_applied_as_proof_for_zero=True,
            normal_staggering='current stored after fadvance can precede the newly advanced endpoint states; I!=G(endpoint STATE)*(endpoint V-E) alone is not an error',
            additional_cache_dependency='C1 rates writes powers before updating STATE; next current callback still reads those powers instead of the available latest STATE. This is one additional gate-state generation relative to direct current readout, not a claim that the entire measured current has one common physical delay.',
            fcurrent='calls current/matrix setup without state callbacks; can overwrite assigned current/ionic buffers, cannot generally refresh these C1 powers or h held inside rates; custom current code may also have state side effects'),
        local_generated=local,h_boundary=dict(dynamic_STATE_names=['e_q','f_q','p_q','q_q','n_q'],h_q_is_STATE=False,
            h_fcond_min=1-alpha,h_fcond_max=1,examples=hvalues),NMJ_scope_note=syn_note,
        protocol_recommendations=contract,controls_are_algebra_only=True,new_simulations=0,A_protocol_available=(A/'protocol.json').exists(),issues=[])
    write(ROOT/'static-review.json',result);write(ROOT/'analytic-control-expectations.json',expect)
    previous=json.loads((ROOT/'review.json').read_text()) if (ROOT/'review.json').exists() else {}
    if previous.get('protocol_review',{}).get('status') not in ['complete','reviewed_with_findings']:
        stage=dict(result,status='static_stage_complete',
            protocol_review=dict(status='pending_A_files',static_design_assessment='sufficient_for_bundled_total_effects_and_interaction_with_stated_limits',
                main_new_condition_budget=6,full302_or_body_budget=0,small_control_budget_max=12,
                control_plan='up to8 zero-step voltage/current evaluations plus4 short fixed-voltage K-fast controls at one dt',
                no_polling_or_background_work=True,next_trigger='mainline will recall after A protocol is saved'))
        write(ROOT/'review.json',stage)
    lines=['# C1 电流/门控调用时序：独立静态复核', '',
        '2026-11-10。已核对 NEURON 9.0.2 官方文档、对应标签源文件、本地版本头与内置 hh.mod，以及旧 C1 实际生成 C++。本分支没有导入 NEURON、编译或运行科学仿真。', '',
        '## 正常分裂与额外缓存依赖', '',
        '9.0.2 的固定步长先推进半个时钟并建立电流/电导矩阵，再更新电压，最后完成另半个时钟并调用机制状态更新、记录。电流与刚更新的端点 STATE 不同拍是正常现象。默认 secondorder=0 是一阶隐式电压法；不能直接把 secondorder=2 的半步二阶精度解释套在当前数据上。[程序控制说明](https://www.neuronsimulator.org/en/9.0.2/progref/simctrl/programmatic.html)、[9.0.2 fadvance.cpp](https://github.com/neuronsimulator/nrn/blob/9.0.2/src/nrnoc/fadvance.cpp)、[9.0.2 treeset.cpp](https://github.com/neuronsimulator/nrn/blob/9.0.2/src/nrnoc/treeset.cpp)。', '',
        '本 C1 的额外依赖来自生成代码：rates() 先计算五个门控幂，随后 states() 才更新相应 STATE；下一次求电流仍读取这批幂，且 _nrn_current() 不调用 rates()。与直接读取“该次电流调用时已可用的最新 STATE”相比，缓存引用额外落后一个门控状态世代。这是结合调用顺序和生成代码作出的推断，不能概括为整个电流或钙反馈都多延迟同一个 dt。官方 hh.mod 在电流表达式直接使用门控 STATE 的幂，提供了对照结构。[内置 HH 源码](https://github.com/neuronsimulator/nrn/blob/9.0.2/src/nrnoc/hh.mod)。', '',
        '## 2×2 能回答什么', '',
        '两因子应命名为“原 Euler/合并指数更新”与“五个 STATE 幂的缓存/直接读出”。前者同时改变门控和 CaPool 的积分表示；设计足以检验这一组合、读出方式及交互，不能把效应单独归给某一个门控或浓度池。复用 original/both 的六条旧主条件、仅新增两种直接读出各三步长，预算为六条新孤立肌肉主条件、零完整网络/身体运行。', '',
        '直接读出会自然改变初始化后求出的电流，即使 STATE 初值相同。主条件应保留原初始化，不通过额外调用或手工同步缓存掩盖差异；启动差异须保存。4–8秒窗口仍可能保留初始相位差，因此主结果只能解释为整个读出处理的总效应；短控制能证明持续调用依赖，但不能量化其占8秒结果的比例。', '',
        '新直接读出轨迹宜保留保存端点前最后一个神经子步的电压、五个 STATE、门控缓存和 h_fcond，并保存推进后未重算的逐通道电流。不能用推进后的 V/h 反算早先矩阵阶段的实际电流。旧细步长主数据按0.025ms保存，缺少部分细步调用前快照；足以复用作匹配输出比较，但不能据此完整还原细步电流调用时序。', '',
        '## h 与其余通路的边界', '',
        'h_q 不在 STATE 中；它由 cai 代数生成。h_fcond=1+(h_q−1)×h_alpha，范围为 '+str(1-alpha)+' 到1，不能用 h_q 替代。仅改 e²、f、p⁴、q、n 的读出，保留 h_fcond 缓存、不刷新 h(cai) 或 rates()，是本轮必要的隔离边界。因此即使该改动有效，也没有修复全部 Ca 电流时间关系。', '',
        '旧 NMJ 的 i 也在 rates() 中缓存；其 _nrn_current() 不重算 i，且含饱和时可赋值 s 的分支。本轮保持 NMJ 原样，只将其列为保留依赖；不得把五个密度门控直接读出称为全部电流时序修复。', '',
        '## fcurrent 与少量控制', '',
        '官方 fcurrent() 路径重新建立电流/矩阵，不调用这里的状态回调。它会覆写电流等观测缓冲；对于本 C1，它也不保证刷新位于 rates() 内的幂和 h。自定义电流代码可能含状态赋值，所以不能把 fcurrent 当作无副作用观察。[官方说明](https://www.neuronsimulator.org/en/9.0.2/progref/simctrl/programmatic.html#fcurrent)。主条件及 record() 禁止增加显式 fcurrent/rates；仅允许一次性独立控制，保存调用前后的 STATE、浓度、缓存、电流与时间后丢弃实例。', '',
        '建议少量控制：', '',
        '1. 零步代数控制：在独立实例内将 STATE 与缓存设为不同已知值，固定实测 h_fcond，用两个电压核对当前生成电流及电导；显式 fcurrent 仅在此使用。',
        '2. 短固定电压快K控制：四版本、一个预先固定步长；关闭其他电流，−10mV钳制，p0=0.7、q=q_inf，仅在此将缓存初始置为一致。用闭式门控解预测每一步实际电流。正常直接读出在端点 n 的保存电流对应本步开始的 p(n−1)，缓存版则从第二步起对应 p(n−2)。控制中不插入显式 fcurrent。',
        '3. h 边界代数核对：用保存值验证 h_fcond 的变换和其在实际生成电流中的保留。', '',
        '闭式预测见 `analytic-control-expectations.json`，是独立代数计算，不是仿真。实际协议、补丁、密集采样和结果尚待 A 保存后续审阅。所有输出仅位于本复核的新目录与 timing_review 文件中。']
    REPORT.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],A_protocol=result['A_protocol_available'],new_simulations=0)))

if __name__=='__main__':main()
