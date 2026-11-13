"""Prepare a locked current-timing protocol and source-only patches.

This program has no compilation or scientific-run entry point. It only copies
pinned inputs, verifies existing records and writes a proposal for main review.
"""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'results/c1-current-timing-2026-11-10'
OLD=PROJECT/'results/c1-muscle-integration-isolation-2026-11-09'
REPORT=PROJECT/'docs/reports/c1-current-timing-2026-11-10.md'
REPLACEMENTS={
    'ca_boyle.mod':('fopen0 = e_fcond * f_fcond * h_fcond',
                    'fopen0 = (e_q ^ e_instances) * (f_q ^ f_instances) * h_fcond'),
    'k_fast.mod':('fopen0 = p_fcond * q_fcond',
                  'fopen0 = (p_q ^ p_instances) * (q_q ^ q_instances)'),
    'k_slow.mod':('fopen0 = n_fcond', 'fopen0 = n_q ^ n_instances')}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def prepare():
    if ROOT.exists():
        assert not (ROOT/'protocol.json').exists(), 'Never overwrite a locked protocol'
        assert {p.name for p in ROOT.iterdir()} <= {'task-control.json'}
    ROOT.mkdir(exist_ok=True)
    old_protocol=read(OLD/'protocol.json');builds=read(OLD/'build-evidence.json')
    assert sha(OLD/'protocol.json')==read(OLD/'protocol-lock.json')['protocol_sha256']
    sources={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        assert sha(source)==sha(target)
        sources[str(target.relative_to(ROOT))]=dict(source=str(source.relative_to(PROJECT)),sha256=sha(source))
    for variant in ['original','both']:
        for name,digest in builds[variant]['sha256'].items():
            source=OLD/'variants'/variant/name;assert sha(source)==digest,(variant,name)
            copy(source,ROOT/'frozen'/variant/name)
    for name in ['protocol.json','protocol-lock.json','build-evidence.json','summary.json','independent-validation.json','archive-validation.json']:
        copy(OLD/name,ROOT/'legacy'/name)
    copy(OLD/'presynaptic-input.npz',ROOT/'presynaptic-input.npz')
    assert sha(ROOT/'presynaptic-input.npz')==old_protocol['input_sha256']
    for name in ['integration_run.py','integration_analyze.py','integration_verify.py']:
        copy(PROJECT/'analysis/neuromechanics/b_motor_validation'/name,ROOT/'legacy-code'/name)
    for name in ['docs/c1-current-timing-2026-11-10.md','docs/reports/research-stage-summary-2026-11-09.md',
        'docs/reports/c1-muscle-integration-isolation-2026-11-09.md']:
        copy(PROJECT/name,ROOT/'context'/Path(name).name)
    # No model is loaded here. Hash/clock/input checks inspect old raw data only.
    with np.load(ROOT/'presynaptic-input.npz') as z:
        dense=z['voltage_mV'].copy();time=z['time_ms'].copy()
    assert np.array_equal(time,np.arange(1280001)*.00625)
    reuse={};hashes={};expected_names=None
    for legacy,new in [('original','euler_cached'),('both','exp_cached')]:
        for dt in [.025,.0125,.00625]:
            folder=OLD/'runs'/f'main_{legacy}_dt{dt:g}';record=read(folder/'record.json');spec=read(folder/'spec.json')
            assert record['complete'] and record['spec']==spec
            assert sha(folder/'trace.npz')==record['trace_sha256']
            assert spec['runner_sha256']==sha(ROOT/'legacy-code/integration_run.py')
            assert spec['protocol_sha256']==sha(ROOT/'legacy/protocol.json')
            assert spec['input_sha256']==old_protocol['input_sha256'] and spec['dt_ms']==dt
            assert record['duration_ms']==8000 and record['sample_ms']==.025 and record['rows']==320001
            assert record['actual_dt_ms']==dt and abs(record['actual_neural_clock_ms']-8000)<1e-5
            assert record['versions']['neuron']=='9.0.2' and record['versions']['numpy']==np.__version__
            with np.load(folder/'trace.npz') as z:
                names=z['columns'].tolist();data=z['data'];get=lambda n:data[:,names.index(n)]
                assert data.shape==(320001,32) and np.isfinite(data).all()
                assert np.array_equal(get('time_ms'),np.arange(320001)*.025)
                assert np.array_equal(get('peer_at_record_mV'),dense[::4])
                previous=np.maximum(np.arange(320001)*4-round(dt/.00625),0)
                assert np.array_equal(get('last_applied_peer_mV'),dense[previous])
                if expected_names is None:expected_names=names
                assert names==expected_names
                syn_sha=hashlib.sha256(get('synapse_s').tobytes()).hexdigest()
            hashes[(legacy,dt)]=syn_sha
            key=f'{new}_dt{dt:g}'
            reuse[key]=dict(source=str(folder.relative_to(PROJECT)),trace_sha256=record['trace_sha256'],
                spec_sha256=sha(folder/'spec.json'),record_sha256=sha(folder/'record.json'),
                verified=['complete','pinned source and compiled kernel','spec and runner SHA','NEURON/Numpy versions',
                    'actual dt and endpoint clock','320001 rows / .025 ms','complete endpoint and left-held input arrays'],
                eligibility='eligible subject to protocol review and runtime-default limitation',
                missing_runtime_field='Historical record does not save secondorder. Source never sets it; current same-version fresh stdrun default is 0. This is inference from source/default, not a measured historical field.')
    for dt in [.025,.0125,.00625]:assert hashes[('original',dt)]==hashes[('both',dt)]
    write(ROOT/'reuse-audit.json',reuse)
    for legacy,new in [('original','euler_direct'),('both','exp_direct')]:
        target=ROOT/'proposed-variants'/new;target.mkdir(parents=True)
        diff=[];changed=[]
        for source in sorted((ROOT/'frozen'/legacy).iterdir()):
            if source.suffix not in ['.mod','.hoc']:continue
            text=source.read_text();updated=text
            if source.name in REPLACEMENTS:
                before,after=REPLACEMENTS[source.name];assert text.count(before)==1
                updated=text.replace(before,after);changed.append(source.name)
                diff.extend(difflib.unified_diff(text.splitlines(True),updated.splitlines(True),
                    fromfile=f'{legacy}/{source.name}',tofile=f'{new}/{source.name}'))
                assert updated.replace(after,before)==text
            (target/source.name).write_text(updated)
        assert changed==['ca_boyle.mod','k_fast.mod','k_slow.mod']
        (ROOT/'patches').mkdir(exist_ok=True);(ROOT/'patches'/f'{new}.diff').write_text(''.join(diff))
    # Read installed first-party API documentation without constructing cells,
    # loading mechanisms, initializing state or advancing a simulation.
    import neuron
    docs=dict(neuron_version=neuron.__version__,fadvance=neuron.h.fadvance.__doc__,
        fcurrent=neuron.h.fcurrent.__doc__,finitialize=neuron.h.finitialize.__doc__)
    write(ROOT/'installed-api-docs.json',docs)
    # The old worker also loads stdrun; inspect its present default only.
    neuron.h.load_file('stdrun.hoc')
    defaults=dict(neuron_version=neuron.__version__,secondorder=float(neuron.h.secondorder),
        cvode_active=float(neuron.h.cvode_active()),python=sys.version,
        note='Current fresh interpreter defaults only; no historical measurement inferred as fact',
        cells_constructed=0,finitialize_calls=0,fadvance_calls=0)
    assert defaults['secondorder']==0 and defaults['cvode_active']==0
    write(ROOT/'current-runtime-defaults.json',defaults)
    write(ROOT/'sources.json',sources)
    control_fields=['Python phase and step index','h.t before/after fadvance','current callback count and callback t/v',
        'five states visible inside the actual _nrn_current callback','five cached powers visible at that callback',
        'effective fopen/gion and individual ica/ik at that callback','h_fcond/h_q/cai snapshots without refreshing',
        'five states and caches before/after fadvance','voltage and concentration STATE/cai',
        'natural finitialize return snapshot','post-control state override before any current refresh',
        'terminal snapshots before/after one fcurrent in observer arm only']
    protocol=dict(date='2026-11-10',created_utc=datetime.now(timezone.utc).isoformat(),status='locked_pending_main_review',
        scientific_execution_authorized=False,compile_or_new_scientific_run_performed=False,
        design=dict(factors=['state integrator: original Euler / 11-09 combined exponential',
            'current readout: frozen cached powers / five STATE powers directly in BREAKPOINT'],
            cells={'euler_cached':'reuse original','exp_cached':'reuse both','euler_direct':'new original plus three current-expression replacements',
                'exp_direct':'new both plus the same three replacements'}),
        protected=dict(parameters=True,geometry=True,STATE_initialization_blocks=True,initial_STATE_values=True,
            NMJ=True,presynaptic_file=True,rates_and_observer_caches=True,CaPool_sync_and_clipping=True,
            h_scope='h_q is ASSIGNED, not STATE; retain h_fcond exactly in Ca current; do not substitute h_q or refresh h(cai)/rates/cai',
            initial_current='May change within this factor despite unchanged INITIAL/STATE values. Explicitly measure native finitialize current before fadvance. Do not claim complete initialization equivalence.'),
        main=dict(new_conditions=6,reused_conditions=6,duration_ms=8000,dt_ms=[.025,.0125,.00625],record_ms=.025,window_ms=[4000,8000],
            inherited_protocol_sha256=sha(ROOT/'legacy/protocol.json'),input_sha256=sha(ROOT/'presynaptic-input.npz'),
            input_convention=old_protocol['main']['input_convention'],NMJ_nS=.9,NMJ_Vth_mV=-30,
            start_condition='Run only after explicit protocol approval, successful source/generated-code checks and observer noninterference controls',
            loop='Copy old worker state/advance loop exactly; only root, variant selection, provenance and passive reads may change. No h.fcurrent, rates calls, observer hooks or event callbacks in main conditions.',
            recordings='Keep all old 32 columns unchanged. Passive extra reads may add individual Ca/K current, gion/fopen, natural t=0 h.t and secondorder; never invoke a calculation to obtain them.',
            initialization_report_windows_ms=[[0,.1],[.1,1],[1,10],[10,100]],
            initialization_windows_are='Readouts of the same approved main trajectories, not extra conditions, time shifts, exclusions or initial transient correction',
            allowed_extra_main_conditions_if_reuse_rejected=0,
            input_hold_confound='Same function/file and common endpoints, but hold interval differs with dt. Same-dt factorial contrasts isolate readout/integrator more directly; dt-pair differences are not strict continuous-solution errors.'),
        generated_code_gate=dict(required='Inspect executable function bodies, not embedded MOD strings',
            state_functions='Must match parent version up to whitespace for all four mechanisms',
            current_functions='Only Ca e_q^e_instances*f_q^f_instances, Kfast p_q^p_instances*q_q^q_instances, Kslow n_q^n_instances replace cached powers; Ca retains h_fcond',
            no_hidden_refresh='No rates(), h_inf/h_q recomputation, concentration synchronization or INITIAL refresh inserted into current calculation',
            legacy_helpers='Unmodified; IntegrationCaInput compiled if preserved, never inserted in main'),
        controls=dict(conditions=16,kernels=['euler_cached','exp_cached','euler_direct','exp_direct'],dt_ms=[.025,.00625],
            arms=['uninstrumented reference','control-only passive current observer'],duration_ms=5,
            fresh_process_each=True,voltage_mV=-10,SEClamp_rs_MOhm=1e-9,
            geometry_and_nonzero_Ca_K_densities='Unchanged C1 muscle; SEClamp holds voltage while actual ionic currents are measured',
            natural_init='Initially clamp at -45 mV; call original initialization and snapshot native returned STATE, caches, individual currents before any override. Compare across readout factor separately.',
            nonsteady_start='Then set clamp and membrane to -10 mV and set e/f/p/q/n=.8/.2/.7/.1/.6 only in these controls; no fcurrent or rates refresh before first fadvance. Clearly label the stale pre-advance current field as not re-evaluated.',
            observation=control_fields,
            observer_implementation='Control-only ASSIGNED/RANGE snapshots and call counters placed around BREAKPOINT current arithmetic; generated C++ must show them inside actual _nrn_current, before/after the original arithmetic. No writes to physical state, cached powers, rates, current or concentrations. Verify same arithmetic and same states(). Not implemented or proven inert at protocol time.',
            nonintrusion_pair='Compare the complete physical reference vs observed control before terminal intervention. Observer differs only in metadata. If beyond tolerances, stop; do not run main or silently refresh.',
            terminal_fcurrent='In the observer arm only, after the last fadvance at 5 ms, take pre snapshot, call h.fcurrent once, take post snapshot, never advance further. Not part of natural current sequence or main. Metadata-only before/after reads are separate from this active query.',
            callback_limit='Last callback snapshot plus actual current evaluation count; do not pretend last-only observations are a full call trace. Record generated v+.001/v order; do not label last snapshot as integrated time-average.',
            analytic_gate_reference='At measured constant V, compare five gate trajectories with fixed-coefficient exponential, and also the exact discrete Euler recurrence for original integrators. Both are diagnostics; do not expect Euler to pass exponential roundoff tolerance.',
            current_hypotheses=['actual callback caches and callback V/h_fcond','actual callback STATE powers and callback V/h_fcond',
                'returned post-fadvance STATE powers (endpoint mismatch hypothesis)'],
            current_formula_units='gmax in S/cm2 times gate factors times driving voltage mV gives mA/cm2; evaluate each Ca, fast K, slow K current separately, not only summed ik',
            claim_rule='Endpoint cache mismatch alone never establishes extra current delay. Require callback-level current/formula match, distinguish callback state from post-advance state and initialization, then interpret alongside independent NEURON9.0.2 call-order review. No full-system or biological latency claim.'),
        control_tolerances=dict(voltage_hold_mV=1e-6,initial_distance_from_inf_min=.05,
            observed_vs_plain_voltage_mV=1e-9,observed_vs_plain_gate_abs=1e-12,
            observed_vs_plain_concentration_mM=1e-15,observed_vs_plain_current_mA_cm2=1e-12,
            formula_current_abs_mA_cm2=1e-12,formula_current_relative=1e-9,
            gate_recurrence_max_abs=1e-9,power_match_abs=1e-12,
            hypothesis_separation='If two formula hypotheses are within 5 times the mixed current tolerance, classify as unresolved at that sample',
            sustained='Report first two updates separately. Persistent distinction requires at least 5 successive informative current-callback samples after those updates; otherwise no persistent-delay conclusion.'),
        acceptance=old_protocol['acceptance'],
        factorial_analysis=dict(same_dt_effects='Direct minus cached at each integrator; exponential minus Euler at each current readout; difference-of-differences as interaction',
            report_fields=['V','five gates','STATE calcium','cai','drive','individual currents when comparable'],
            step_pairs_ms=[[.025,.0125],[.0125,.00625],[.025,.00625]],
            scaling='Copy old per-field statistics, RMS floors and no-shift rules exactly. Reused old current columns are endpoint values, not callback observations.',
            no_scope_creep='No fitted phase shift, additional timestep, parameter search, input-hold intervention or initialization correction',
            improvement='For each integrator separately apply old >=2x reduction rule with its predefined tiny-error exemption to cached-vs-direct last-pair V/STATE-Ca/cai/drive errors. Report actual ratios even if exempt; no improvement claimed from an exemption.',
            conclusion='Components may support or fail a current-readout mechanism. No automatic full302/body authorization or completion-progress change.'),
        budgets=dict(new_main=6,main_duration_s_each=8,reused_main=6,small_controls=16,small_control_duration_ms_each=5,
            new_pool_stress_conditions=0,new_full302=0,new_body=0,new_training=0,new_parameter_search=0,
            terminal_fcurrent_calls=8,main_fcurrent_calls=0,
            compile='After approval only: two new uninstrumented direct versions plus four control-only observer versions; original cached binaries eligible for reuse after hashes'),
        reuse=dict(audit_sha256=sha(ROOT/'reuse-audit.json'),missing_secondorder_field='Disclosed; reviewer may accept same-version source/default evidence or reject reuse. Rejection does not authorize more runs.',
            archive_payload='After approval copy/link verified six old trace/spec/record sets into the new evidence bundle, keeping old files untouched'),
        pending_review=['Approve this frozen protocol and the 6 main + 16 short control budget',
            'Accept or reject historical reuse evidence, including the explicitly missing recorded secondorder field',
            'Independent branch call-order review must inform interpretation; it does not authorize changing this lock',
            'Observer source and actual generated placement must pass protected-arithmetic and noninterference gates before main'],
        source_manifest_sha256=sha(ROOT/'sources.json'),prepare_script_sha256=sha(Path(__file__)))
    write(ROOT/'protocol.json',protocol)
    write(ROOT/'protocol-lock.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),
        status='awaiting_explicit_main_review',new_scientific_runs=0,new_compilations=0))
    # Hash new proposal sources separately; they are not frozen historical inputs.
    proposed={str(path.relative_to(ROOT)):sha(path) for path in sorted((ROOT/'proposed-variants').rglob('*')) if path.is_file()}
    write(ROOT/'proposal-source-hashes.json',proposed)
    render_report(protocol)
    print(json.dumps(dict(status='protocol_locked_pending_review',protocol_sha256=sha(ROOT/'protocol.json'),
        reused_trace_audits=len(reuse),new_main_budget=6,short_control_budget=16,new_scientific_runs=0,new_compilations=0)))


def render_report(p):
    text=f'''# C1电流时序隔离：锁定协议，等待主线审核

日期：2026-11-10。本文件是运行前协议，不是新实验结果。科学条件、编译和完整网络/身体运行均为0；不会自动执行后续阶段。

协议：`results/c1-current-timing-2026-11-10/protocol.json`；SHA-256：`{sha(ROOT/'protocol.json')}`。`protocol-lock.json`标记等待明确审核。统一交接与旧阶段总结已保存来源快照。

## 因素与最小补丁

| 积分方式 | 缓存电流 | STATE幂直接电流 |
|---|---|---|
| 原Euler | 复用11-09 original三条主迹线 | 新增euler_direct三条 |
| 合并指数更新 | 复用11-09 both三条主迹线 | 新增exp_direct三条 |

两种新版本各只替换3条BREAKPOINT表达式：

- Ca：`e_fcond*f_fcond*h_fcond` → `(e_q^e_instances)*(f_q^f_instances)*h_fcond`。
- 快K：`p_fcond*q_fcond` → `(p_q^p_instances)*(q_q^q_instances)`。
- 慢K：`n_fcond` → `n_q^n_instances`。

提议源码位于本结果目录的`proposed-variants/`，补丁在`patches/*.diff`。所有缓存观测量及rates保持原样；不改几何、参数、NMJ、STATE初始化、CaPool同步和负浓度截断。

**h_q不是STATE，也不等于h_fcond。** Ca仍乘原h_fcond，不刷新cai/h/rates，因此本方案没有修复h(cai)或完整膜电流时序。直接读出可能在finitialize末尾就改变初始电流：INITIAL/STATE初值不改，并不表示初始化后的所有电流等价。自然t=0电流及0–0.1、0.1–1、1–10、10–100 ms窗口均单列，不从主条件中删除初始化影响。

## 主条件预算及复用

只新增两个直接读出版本×0.025/0.0125/0.00625 ms，共6个8秒孤立肌肉主条件。沿用旧输入文件、左端保持、0.025 ms记录和4–8秒验收；旧32列保持原义。主条件不插入h.fcurrent、rates调用或观察钩子。完整302、身体、训练、调参新增预算均为0。

六条旧主迹线已只读核对：完整性、trace/spec/record与运行器SHA、kernel SHA、版本、实际dt/时钟、320001条记录、全部共同端点与实际前一区间输入均一致；同dt的original/both突触s逐元素一致。具体哈希在`reuse-audit.json`，旧文件未改。当前尚未复制大迹线，后续获批后收入新归档。

复用仍有一项明确记录缺口：旧record没有secondorder字段。旧运行器未设置它，当前相同NEURON9.0.2的fresh stdrun默认值实测为0；这支持复用，但不是历史运行时字段的直接证明。由主线审核是否接受；不接受时不自动重跑旧条件，新增6次预算不扩大。

共同端点输入相同不意味着不同dt的区间保持完全相同。保持宽度仍随dt改变；同dt两因素比较更直接，步长配对差不能当作对连续真解的严格误差。

## 如何避免把端点缓存误当作电流延迟

旧生成代码显示_current先使用缓存，states另行推进门控；安装版fadvance说明还区分求电流/导纳、解膜电位、再推进状态的阶段。仅比较h.fadvance返回后的q与缓存，无法确定实际求电流时用了哪一份q。

获批后先检查新生成函数：states与父版不变，_nrn_current只发生上述幂替换，没有隐藏rates/离子刷新。再做4个版本×两个极端dt×有/无控制观察器，共16个5 ms小控制：

1. 原自然初始化并记录返回时的q、缓存及各离子电流，单列直接读出的初始电流差。随后仅在控制中把电压钳至−10 mV、5门控设为0.8/0.2/0.7/0.1/0.6；初值偏离稳态至少0.05。控制保留原非零Ca/K密度，核对实际钳制电压≤1e−6 mV。
2. 无观察器参考与被动观察器分别在fresh进程运行。观察器只在控制中增加ASSIGNED/RANGE快照/计数；必须从生成C++确认它在实际_current算式两侧捕获回调时刻、v、当时q/缓存/h_fcond、fopen/gion和单独Ca/快K/慢K电流。记录前后推进状态，并分别标记v+.001和v调用。当前尚未编写/编译观察器，不声称已证明无扰动。
3. 逐电流核对“同次回调缓存幂”“同次回调STATE幂”“推进返回后STATE幂”三假设。单位为S/cm²×mV=mA/cm²；绝对容差1e−12 mA/cm²与相对1e−9取较大者。两个假设相距不足5倍容差则标为不可区分。前两次更新单列；持续区别至少需后续5个连续、有区分力的回调样本。
4. 观察器有/无两组的物理轨迹须分别在1e−9 mV、1e−12门控、1e−15 mM浓度、1e−12 mA/cm²电流内一致，否则停止主条件。观察器只保留最近回调快照和调用计数时，不伪称已记录完整调用序列；须结合真实生成顺序及独立NEURON实现复核解释。
5. 每个观察器控制在5 ms末尾另做一次h.fcurrent前后快照，不再推进；共8次主动求值，全部隔离于主条件和自然调用序列。控制重置后的旧current字段没有重新求值，不能与新STATE混为同一时刻。

门控另外与固定电压解析指数解及Euler离散递推各自比较，递推误差门槛1e−9；不要求Euler达到解析指数的舍入误差。观察器若改变系统或生成位置不符，只保留失败记录，不通过刷新rates等方式补救。

## 判据与尚待审核项

主响应沿用11-09全部幅度、统计、逐时RMS、尺度下限与不平移规则；三档所有两两配对均报告。分别在Euler和指数积分下比较直接/缓存读出，再报告交互项。改善仍用旧的至少减半规则和原误差极小时的预定豁免，报告实际比值，豁免不叫改善。

即使回调证实缓存和同次可见STATE有区别，也先限定为该电流读出机制的结果；须结合正常分裂积分的时间定义，不能自动称为生物延迟、全部误差来源或完整系统修复。

待主线决定：是否批准6个主条件＋16个短控制；是否接受旧数据secondorder的来源/默认值证据。独立分支的调用顺序复核用于解释，不授权改动锁定协议。观察器具体实现及实际生成位置、无扰动性须在主条件前检查通过。任何需要改预算、刷新h/cai或修正初始化的发现均另交协议，不自动扩展。
'''
    REPORT.write_text(text)


if __name__=='__main__':prepare()
