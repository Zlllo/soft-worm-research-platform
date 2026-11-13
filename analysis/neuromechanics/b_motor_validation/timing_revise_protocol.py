"""Apply the main thread's pre-execution budget/control steering as version 2.

No compilation, NEURON import, model initialization or scientific execution.
The superseded, entirely unexecuted v1 is preserved with its exact lock/source.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'results/c1-current-timing-2026-11-10'
REPORT=PROJECT/'docs/reports/c1-current-timing-2026-11-10.md'
REVIEW=PROJECT/'results/c1-current-timing-review-2026-11-10'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    p=read(ROOT/'protocol.json');lock=read(ROOT/'protocol-lock.json')
    assert sha(ROOT/'protocol.json')==lock['protocol_sha256'] and p.get('protocol_version',1)==1
    assert lock['new_scientific_runs']==0 and lock['new_compilations']==0
    history=ROOT/'protocol-history/v1-unexecuted';history.mkdir(parents=True,exist_ok=False)
    for path in [ROOT/'protocol.json',ROOT/'protocol-lock.json',REPORT,Path(__file__).with_name('timing_prepare.py')]:
        shutil.copy2(path,history/path.name)
    assert sha(history/'timing_prepare.py')==p['prepare_script_sha256']
    review_sources={}
    for name in ['static-review.json','analytic-control-expectations.json']:
        source=REVIEW/name;target=ROOT/'review-context'/name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(source,target)
        review_sources[str(target.relative_to(ROOT))]=dict(source=str(source.relative_to(PROJECT)),sha256=sha(target))
    write(ROOT/'review-source-manifest.json',review_sources)
    p.update(protocol_version=2,supersedes_protocol_sha256=lock['protocol_sha256'],
        revised_utc=datetime.now(timezone.utc).isoformat(),
        revision_reason='Main thread directed 8 zero-step current evaluations + 4 short K-fast controls, no observer trajectory; incorporate independent branch 9.0.2 source/algebra review. No scientific conditions or compilations occurred under v1.')
    p['main'].update(
        passive_extra_fields=['Before LAST neural substep preceding each .025ms record: h.t, voltage, five STATE values, five cached powers, h_fcond',
            'After fadvance without recomputation: individual Ca/fast-K/slow-K exposed current, gion/fopen if available',
            'Natural finitialize returned STATE, caches, per-channel currents, voltage, CaPool STATE/cai and secondorder'],
        initialization_limit='Startup-current effect is part of the readout total effect. Early/late windows do not uniquely separate startup effects from ongoing readout effects.',
        legacy_decimation_limit='Old .0125/.00625ms traces omit before-current voltage/cache/h snapshots for fine substeps. Reuse them for matched response metrics, not full fine-step current reconstruction.',
        non_density_current='Preserve NMJ source exactly, including i assigned inside rates, lack of numeric-voltage-probe recomputation and its original saturation branch. No assertion that all current timing is repaired.')
    p['generated_code_gate'].update(
        no_observer_in_main=True,
        NMJ='Exact SHA equality to frozen parent .mod and generated current/state arithmetic; do not add rates or recompute NMJ i at voltage probes',
        accessible_current_readouts='Use existing per-channel exposures (e.g. negative i__ca_boyle/i__k_fast/i__k_slow), distinguish from summed seg.ik; reads only, no computation callbacks')
    p['controls']=dict(total_conditions=12,observer_instrumented_kernels=0,modified_observer_current_hooks=0,
        zero_step=dict(conditions=8,kernels=['euler_cached','exp_cached','euler_direct','exp_direct'],voltage_mV=[-20,-10],
            duration_ms=0,finitialize_once_each=True,fadvance_calls=0,explicit_fcurrent_calls=1,fresh_process_each=True,
            cell='Frozen C1 muscle with original nonzero channel densities; no NMJ inserted',
            preparation='Record natural finitialize return first. In this disposable zero-step control only, set membrane voltage, e/f/p/q/n=.8/.2/.7/.1/.6, and set five cache powers deliberately distinct: e=.11, f=.61, p=.08, q=.77, n=.19. Do not call rates.',
            h='Read and freeze actual natural-initialization h_fcond; do not change h_q, h_fcond, cai or concentration to create the algebra test',
            action='Save complete before snapshot; one h.fcurrent; save complete after snapshot; dispose process, never advance',
            readouts=['h.t','voltage','five STATE','five caches','h_q/h_fcond','per-channel currents','summed ionic current buffers','STATE concentration/cai'],
            invariants='Time, membrane voltage, five STATE, cached powers, h values and concentration STATE/cai must not change. Assigned currents and ionic buffers may change and are the measured output; save all changes.',
            current_hypotheses=['cached powers at requested measured voltage and measured h_fcond','STATE powers at the same voltage and h_fcond'],
            cpp_phase='Confirm generated nrn_cur calls v+.001 then v; last exposed per-channel current after fcurrent corresponds to the v evaluation. This is an active zero-step query, not a natural trajectory sample.'),
        short_K_fast=dict(conditions=4,kernels=['euler_cached','exp_cached','euler_direct','exp_direct'],dt_ms=.025,
            duration_ms=5,neural_steps=200,record_every_neural_step=True,SEClamp_rs_MOhm=1e-9,voltage_mV=-10,
            NMJ_inserted=False,
            active_density='Only original K-fast gmax nonzero; Leak/Ca/slow-K zero only in this named isolation control. CaPool/sync implementation unchanged.',
            init='Record natural finitialize return; set control V=-10, p0=.7, q=q_inf(-10). Align p_fcond=p0^4 and q_fcond=q_inf exactly ONCE in the control before first fadvance. No later cache/rates refresh.',
            no_explicit_fcurrent=True,no_observer_hooks=True,
            record_before=['h.t','actual membrane V','p/q STATE','p_fcond/q_fcond','STATE concentration/cai'],
            record_after=['h.t','actual membrane V','p/q STATE','p_fcond/q_fcond','existing per-mechanism fast-K current exposure without recomputation','STATE concentration/cai'],
            algebra='p_j=p_inf+(p0-p_inf)*(1-dt/tau)^j for Euler; p_j=p_inf+(p0-p_inf)*exp(-j*dt/tau) for exponential; q remains q_inf.',
            direct_current_prediction='Stored current after step j uses p_(j-1)^4: normal current-before-state ordering. Compare at recorded pre-step voltage.',
            cached_current_prediction='After initial alignment: step1 uses p0^4; j>=2 uses p_(j-2)^4. Test this extra STATE-generation dependence separately from the normal endpoint mismatch.',
            endpoint_hypothesis='A stored current not equal to current computed from newly advanced p_j is normal splitting evidence, not by itself a defect.',
            scope='Aligned control separates continuing readout dependence from an initial cache mismatch, but cannot quantify its share in 8s main responses.'),
        offline_h_algebra=dict(new_runs=0,expression='h_fcond=1+(h_q-1)*h_alpha',
            boundary='h_q is ASSIGNED, not STATE. Verify saved values/generated readout and retain measured h_fcond in every Ca formula; no cai/h refresh.'),
        interpretation='Independent review establishes current/matrix -> voltage -> state -> recording for NEURON9.0.2, secondorder=0. Refer to additional gate-state generation only if zero-step and short-control predictions agree; never impose secondorder=2 accuracy time labels onto secondorder=0.')
    p['control_tolerances']=dict(voltage_hold_mV=1e-6,zero_step_STATE_and_cache_abs=1e-12,
        zero_step_voltage_mV=1e-12,zero_step_concentration_mM=1e-15,zero_step_time_ms=0,
        formula_current_abs_mA_cm2=1e-12,formula_current_relative=1e-9,
        short_gate_discrete_recurrence_max_abs=1e-9,short_constant_q_max_abs=1e-9,
        current_units='S/cm2 * mV = mA/cm2; use actual per-channel gmax and driving voltage; never substitute summed ik for fast-K current',
        hypothesis_separation='At a sample, alternatives within 5 times max(abs tolerance, rel tolerance*current magnitude) are indeterminate.',
        sustained='First step is expected equal after aligned caches. Require five consecutive informative steps from step2 onward to distinguish cached-extra-generation vs normal splitting; report all 200 steps.',
        no_unplanned_repair='If an invariant or current-exposure/phase check fails, preserve failure and stop main execution. No observer, rates, initialization or ion-sync repair may be improvised inside this protocol.')
    p['budgets'].update(small_controls=12,zero_step_controls=8,zero_step_duration_ms_each=0,
        short_K_fast_controls=4,small_control_duration_ms_each=5,short_control_dt_ms=.025,
        explicit_fcurrent_calls=8,main_fcurrent_calls=0,short_trajectory_fcurrent_calls=0,
        observer_conditions=0,compile='After approval only, two new direct-current kernels. Reuse parent cached kernels after hashes. No instrumented kernels.')
    p['budgets'].pop('terminal_fcurrent_calls',None)
    p['review_evidence']=dict(manifest_sha256=sha(ROOT/'review-source-manifest.json'),
        static_review_sha256=sha(ROOT/'review-context/static-review.json'),
        analytic_expectations_sha256=sha(ROOT/'review-context/analytic-control-expectations.json'),
        accepted_source_facts=['fixed-step current/matrix then voltage then STATE then record',
            'default secondorder=0','fcurrent has assigned/ionic-buffer side effects but does not generally refresh these rates caches',
            'normal returned-current/state mismatch is distinct from extra cached STATE-generation dependence',
            'NMJ current dependency remains out of scope'])
    p['pending_review']=['Explicit approval of this v2 lock and 6 new main + 8 zero-step + 4 short control budget',
        'Accept reuse audit including historical missing recorded secondorder; independent pinned-source/default evidence supports 0 but is not a historical runtime log',
        'After approval actual generated functions and control phase/exposure/invariant checks must pass before six main conditions',
        'No full302/body/training or extra conditions implicitly authorized by a passing component result']
    p['revision_script_sha256']=sha(Path(__file__))
    write(ROOT/'protocol.json',p)
    write(ROOT/'protocol-lock.json',dict(protocol_version=2,protocol_sha256=sha(ROOT/'protocol.json'),
        supersedes_protocol_sha256=lock['protocol_sha256'],status='awaiting_explicit_main_review',
        new_scientific_runs=0,new_compilations=0))
    render(p)
    write(ROOT/'protocol-preflight.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),
        exact_direct_source_files_checked=check_source(),scientific_runs=0,compilations=0,
        original_v1_preserved=True,review_inputs_preserved=True,report_sha256=sha(REPORT)))
    print(json.dumps(dict(protocol_version=2,protocol_sha256=sha(ROOT/'protocol.json'),new_main=6,small_controls=12,
        zero_step=8,short_ms_5=4,new_runs=0,new_compilations=0)))


def check_source():
    changes={'ca_boyle.mod':('fopen0 = e_fcond * f_fcond * h_fcond','fopen0 = (e_q ^ e_instances) * (f_q ^ f_instances) * h_fcond'),
        'k_fast.mod':('fopen0 = p_fcond * q_fcond','fopen0 = (p_q ^ p_instances) * (q_q ^ q_instances)'),
        'k_slow.mod':('fopen0 = n_fcond','fopen0 = n_q ^ n_instances')}
    count=0
    for parent,child in [('original','euler_direct'),('both','exp_direct')]:
        for path in (ROOT/'frozen'/parent).iterdir():
            if path.suffix not in ['.hoc','.mod']:continue
            expected=path.read_text()
            if path.name in changes:expected=expected.replace(*changes[path.name])
            assert expected==(ROOT/'proposed-variants'/child/path.name).read_text();count+=1
    return count


def render(p):
    REPORT.write_text(f'''# C1电流时序隔离：v2锁定协议，等待主线批准

2026-11-10。**本轮仅完成协议、最小源码补丁和旧数据复用核对；新增科学条件0，新增编译0。** 未启动完整302、身体或训练。v1未执行并保存在`protocol-history/v1-unexecuted/`；v2在运行前依据主线预算与B审查修订。

主协议：`results/c1-current-timing-2026-11-10/protocol.json`；v2 SHA-256：`{sha(ROOT/'protocol.json')}`。`protocol-lock.json`仍为等待明确批准。

## 2×2与最小补丁

| 积分因素 | 原缓存电流 | 五STATE直接电流 |
|---|---|---|
| 原Euler | 复用original三条旧迹线 | 新euler_direct三条 |
| 11-09合并指数更新 | 复用both三条旧迹线 | 新exp_direct三条 |

只改每个密度机制中的一条fopen0表达式：Ca为`(e_q^e_instances)*(f_q^f_instances)*h_fcond`，快K为`(p_q^p_instances)*(q_q^q_instances)`，慢K为`n_q^n_instances`。源码在`proposed-variants/`，可直接审核的最小差异在`patches/*.diff`。未编译，不将预期生成形式说成已验证的新结果。

所有参数、几何、NMJ、STATE初始化、rate/缓存观察量、CaPool同步与截断均不改。**h_q不是STATE，也不等于h_fcond。** Ca始终乘原h_fcond，不刷新cai/h/rates；h关系只做离线代数。NMJ原有i缓存、数值电压探针不重算i及原饱和分支均保留，因此本轮不修复全部膜/突触电流时序。

STATE初值与INITIAL代码保持，并不保证初始化电流等价：finitialize末尾自然求电流可能已受直接读出影响。须保存自然t=0状态、缓存和逐通道电流；主迹线的0–0.1、0.1–1、1–10、10–100 ms窗口及4–8秒分别报告。初始效应属于读出因素总效应，窗口分析不能唯一拆出其对后半程的贡献。

## 主条件：最多6次新增，6次历史复用

仅两个直接读出版本×0.025/0.0125/0.00625 ms，各8秒。沿用旧输入文件、左端保持、0.025 ms采样、4–8秒验收和全部误差门槛/尺度下限；三档所有配对均报告，不平移曲线。主条件禁止插入h.fcurrent、rates刷新、观察钩子、新初始化handler或手动cache同步。

六条旧迹线已只读核对trace/spec/record/runner/kernel SHA、完成状态、版本、实际dt与时钟、320001记录及全部共同端点/实际前一区间输入。同dt两旧版突触s逐元素一致。`reuse-audit.json`保留来源；旧文件没改，尚未复制大迹线。

旧record未存secondorder。旧worker不设置该值，当前相同版本fresh stdrun默认0，B也依据9.0.2固定版本源码确认默认0；这是来源/默认值证据，不伪称已有历史运行时字段。由主线接受其复用充分性；若拒绝，不自动重跑旧基线或扩大6次预算。

新主迹线除原32列外，须在每个记录点前最后一次神经子步推进前，保留真实V、5个STATE、5个缓存幂、h_fcond及h.t；推进后只读取现有逐通道电流/状态，不调用计算。旧细步迹线缺少同等的pre-current V/cache/h字段，仅用于匹配响应指标，不能补造完整电流调用证据。

同dt的直接/缓存比较更能隔离读出因素；跨dt时输入保持区间仍变化。不能把其步长差称为严格连续输入真解误差。分别报告两积分算法下direct−cached、两种读出下exp−Euler与交互项，不只选最好版本。改善沿用旧至少减半规则及预定小误差豁免，豁免不叫实测改善。

## 小控制：8个零步＋4个5 ms，合计12个

**零步代数/实际电流：** 四版×−20/−10 mV，共8个fresh实例，无fadvance。先保存自然finitialize返回值；只在可丢弃控制中设置5个STATE为0.8/0.2/0.7/0.1/0.6，缓存幂故意设为0.11/0.61/0.08/0.77/0.19。保留实测h_fcond，不调用rates；保存前快照，调用一次h.fcurrent，保存后快照并结束。必须核对时间、V、STATE、cache、h及浓度不变；assigned电流和离子缓冲可以改变，全部变化须保存。用实际gmax、V与h_fcond分别核对三种密度电流，不把求和ik冒充K-fast。单位S/cm²×mV=mA/cm²，采用max(1e−12绝对,1e−9相对)电流容差。

**短时K-fast：** 四版×预锁dt=0.025 ms，共4个fresh实例，各5 ms/200步，逐神经步保存。关闭控制中的Leak/Ca/slow-K电导，仅保留原K-fast，−10 mV电压钳，p0=0.7、q=q_inf。只在开头一次将p/q缓存对齐相应STATE；之后不再刷新，不额外fcurrent，不插观察器。推进前保存真实V/p/q/cache，推进后保存真实V/p/q/cache及未重算的现有K-fast电流。

Euler按`p_j=p_inf+(p0−p_inf)(1−dt/tau)^j`，指数版按解析指数检查。正常分裂求解中，direct存下的第j步电流对应p_(j−1)，无需等于新返回端点p_j。缓存对齐后的第一步两种读出应一致；随后cached假设对应p_(j−2)。需至少5个连续、有区分力的后续步区分两者，而不是拿端点差当证据。固定电压误差≤1e−6 mV，递推/常q误差≤1e−9；两个电流假设小于5倍混合容差时记为不可区分。

实际生成代码须确认states未改、_nrn_current只替换5个幂、h_fcond保留、逐通道现有暴露量对应的计算位置及v+.001/v最后调用顺序。B已提供9.0.2引擎顺序：电流/矩阵→电压→STATE→记录；secondorder=0不能直接套用secondorder=2的时间精度标签。短控制用于持续读出依赖的隔离，不量化它在8秒响应中的贡献比例。

明确调用预算：零步控制显式h.fcurrent共8次；短轨迹和主条件的额外h.fcurrent均0。没有ASSIGNED观察器版本或原位观测代码，因此无须扩展16个有/无观察器条件。失败时保存并停在门槛，不即时刷新rates、修初始化或改离子同步。

## 审核状态与边界

尚待主线明确批准本v2及6主＋12小控制预算、接受复用证据。获批后先核对实际生成代码并完成上述小控制，再运行6个主条件。控制失败、需要额外预算或改变输入/初始化/h/cai的发现另提协议，不自动扩展。完整302/身体/训练新增预算始终0；组件通过不改变既有完整系统未收敛结论或3/5完成度。

B的静态审查与独立解析预测已复制到`review-context/`并保存SHA，原件不改。当前目录的`protocol-preflight.json`只证明来源/最小差异及锁定状态，不证明尚未执行的科学控制已通过。
''')


if __name__=='__main__':main()
