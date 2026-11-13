"""Control-stage report/archive only; never authorize or run main conditions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'results/c1-current-timing-2026-11-10'
REPORT=PROJECT/'docs/reports/c1-current-timing-2026-11-10.md'
ASSETS=REPORT.parent/'assets/c1-current-timing-2026-11-10'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def f(n):return f'{n:.6g}'


def main():
    s=read(ROOT/'control-summary.json');audit=read(ROOT/'control-validation.json')
    assert audit['independent_control_audit_passed'] and audit['control_summary_sha256']==sha(ROOT/'control-summary.json')
    assert s['new_main_runs']==0
    lines=['# C1电流时序隔离：控制阶段结果', '', '2026-11-10；权威方案v2（8零步＋4短时，无观察器）。', '',
        '**12个控制均通过预定判据。缓存电流和五个STATE幂直接读出确实使用不同的门控信息；固定电压K-fast控制将额外缓存代次与正常的电流先于状态更新区分开。六条8秒主条件尚未启动，等待主线读控制结果后单独许可。**', '',
        f"协议SHA-256：`{s['protocol_sha256']}`。主线与B的v2审查许可已复制在`approvals/`；`control-execution-lock.json`锁定运行器、内核哈希和12个条件，主运行授权数为0。", '',
        '## 最小编译与实际生成代码', '',
        '只新编译euler_direct、exp_direct两版。原Euler/11-09合并指数更新分别保留，且每版只把Ca e/f、快K p/q、慢K n的幂从缓存改为当前STATE表达式。实际生成states()与父版去空白后一致；_nrn_current除fopen0表达式外的算术不变，没有rates或观察钩子；NMJ和CaPool源码、生成电流/状态函数均与父版一致。', '',
        'Ca仍乘原h_fcond，未使用h_q替代它。h_q为代数量，h_fcond=1+(h_q−1)×h_alpha；仅离线核对该关系，没有刷新h/cai。NMJ原有电流缓存及电压探针不重算i的问题保持不改。本阶段不是所有膜/突触电流时序的完整修复。', '',
        'v1的16个观察器控制从未执行；按最终决定恢复已封存的v2原字节。两份协议和旧构建器的哈希门槛退出记录保留。最终没有观察器内核、原位观测钩子或额外控制，也没有真实科学条件失败后重跑。', '',
        '## 8个零步代数/实际电流控制', '',
        '四个版本分别在−20、−10 mV进行，每个fresh进程保留自然finitialize快照，再按协议设置彼此不同的STATE与缓存幂；冻结实测h_fcond。调用一次h.fcurrent后结束，fadvance数为0。时间、电压、STATE、缓存、h及浓度均应不变；assigned电流与离子缓冲允许改变并完整保存。每个通道分别比较缓存幂与STATE幂公式，不能以求和ik代替K-fast。', '',
        '| 版本/电压 mV | 不变量 | 选定电流公式最大误差 mA/cm² | 三通道均区分替代公式 |', '|---|---|---:|---|']
    for name,d in s['zero_step_results'].items():
        err=max(v['selected_formula_abs_error_mA_cm2'] for v in d['currents'].values())
        lines.append(f"| {name} | {'通过' if all(d['invariant_gates'].values()) else '未通过'} | {f(err)} | {'是' if all(v['alternative_distinguishable'] for v in d['currents'].values()) else '否'} |")
    lines+=['', '电流混合容差为max(1e−12 mA/cm², 1e−9×实测电流幅度)，两个假设相距须大于5倍该容差才称可区分。缓存版吻合缓存幂，直接版吻合STATE幂；不是仅依据返回端点cache与q不同。', '',
        '## 4个短时固定电压K-fast控制', '',
        '预锁dt=0.025 ms，每版5 ms/200步，合计20 ms积分。仅保留原K-fast电导，其余Leak/Ca/慢K电导在此控制中关闭；q初值设为q_inf，p0=0.7。只在开始把p/q缓存与STATE对齐一次，随后不调用rates、额外h.fcurrent或插观察器。保存每一步推进前V/STATE/cache，以及推进后的现有逐通道电流和STATE。', '',
        '| 版本 | p递推最大差 | 电压钳误差 mV | 电流代次公式最大差 mA/cm² | 首步两读出相同 | 后续连续可区分步数 |', '|---|---:|---:|---:|---|---:|']
    for name,d in s['short_K_results'].items():
        lines.append(f"| {name} | {f(d['p_reference_max_error'])} | {f(d['voltage_max_error_mV'])} | {f(d['current_discrete_generation_max_error_mA_cm2'])} | {'是' if d['first_step_aligned_formulas_equal'] else '否'} | {d['longest_consecutive_informative_matching_steps_after_first']} |")
    lines+=['', '递推与恒q容差1e−9，电压钳容差1e−6 mV。Euler按离散递推检查，指数版按解析指数检查，未要求Euler达到指数解的舍入误差。', '',
        '在此固定电压控制中，第j步返回的direct电流对应推进前p_(j−1)，不等于刚更新的p_j是正常分裂顺序；cache对齐后第一步两读出一致，从第二步起cached电流对应p_(j−2)。四版后续199步均满足预定区分要求。B对NEURON9.0.2实现的复核给出电流/矩阵→膜电位→STATE→记录顺序，与实际生成代码及本数据一致。', '',
        '举例（Euler）：第一步两读出约0.000434775 mA/cm²；第二步cached仍为0.000434775，direct约0.000427517，而新返回STATE推算的电流约0.000420430。正常端点错位与额外缓存代次因此可分别识别。这里不是直接记录了每次_current回调的完整日志；证据来自独立零步求值、真实生成调用顺序及逐步前后量/已有电流的一致性。', '',
        '**可得的结论限定为此孤立控制的额外门控状态代次依赖。** 不能据一个dt把所有电流归成共同物理延迟，也不能由此确定它在8秒自由肌肉响应或完整神经—身体误差中占多少。主条件仍将包含原初始化和输入保持效应。', '',
        '## 自然初始化电流与反转电位字段说明', '',
        '自然finitialize返回时，两个积分算法各自的缓存/直接版STATE初值最大差为0，但电流已有差异：直接−缓存的Ca电流约−2.58489e−7 mA/cm²，快K约3.15503e−12，慢K约1.47781e−4。INITIAL代码与STATE不变不等于初始电流行为完全等价；未来主条件必须保留并报告这个总效应，不将早期影响悄悄消去。', '',
        '原record中的`metadata.reversal_mV.ca`实际读取seg.eca=1e6，是零胞内钙时的离子层Nernst读数；它不是ca_boyle私有通道的40 mV反转电位。冻结MOD的INITIAL及生成fpfield存储采用内部eca=40、ek=−60，USEION没有READ eca。8个零步的实测V−I/gion均独立反推Ca=40 mV、K=−60 mV，公式分析使用这些通道值。原record和运行器保持原样，补充归属说明保存在`control-readout-clarification.json`；没有改参数或重跑。后续运行器应把离子读数与通道常数分别命名。', '',
        '## 验证与停止位置', '',
        f"独立离线程序不导入主分析或NEURON，复算12条件、800个神经步和{audit['separate_current_formula_samples_checked']}个逐通道公式样本；控制摘要SHA与执行锁均核对。科学条件只执行了既定12个；显式h.fcurrent共8次，全部位于零步控制。没有observer、8秒新主条件、完整302或身体运行。", '',
        '当前停在六条8秒主条件之前；此结果不自行授权主条件。主线还需结合B实际数据复核再许可。旧六条主迹线的复用来源已核对，但secondorder历史字段仍只按同版本源码/默认值推断接受，不能补记为历史实测。', '',
        '- `control-summary.json`：判据、逐条件结果、首4步示例、自然初始化差异及限制。',
        '- `control-validation.json`：独立复算与摘要哈希；`generated-code-validation-v2.json`：生成函数与最小变化证据。',
        '- `controls/*/`：12份spec、原始NPZ和record；原始失败/协议修订/构建门槛记录保留。',
        '- 本阶段控制证据归档：`docs/reports/assets/c1-current-timing-2026-11-10/controls-reproduction.zip`；不包含六条旧8秒大迹线，离线控制复算不依赖它们。', '',
        '解压后在项目根目录（仅需NumPy）：', '', '```sh',
        'OPENBLAS_NUM_THREADS=1 python analysis/neuromechanics/b_motor_validation/timing_control_analyze.py',
        'OPENBLAS_NUM_THREADS=1 python analysis/neuromechanics/b_motor_validation/timing_control_verify.py',
        '```', '']
    REPORT.write_text('\n'.join(lines))
    (ROOT/'control-report.md').write_text(REPORT.read_text())
    state=dict(stage='controls_complete_waiting_for_main_permission',protocol_sha256=s['protocol_sha256'],
        code_builds=2,control_conditions=12,zero_step=8,short_integrated_ms=20,
        control_gates_passed=s['all_control_gates_passed'],independent_validation_passed=True,
        new_main_runs=0,new_full302_body=0,main_runs_authorized_by_this_stage=0,all_control_processes_finished=True)
    (ROOT/'execution-state.json').write_text(json.dumps(state,indent=2)+'\n')
    ASSETS.mkdir(parents=True,exist_ok=True);payload={}
    for path in ROOT.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path!=ROOT/'control-archive-validation.json':payload[str(path.relative_to(PROJECT))]=path
    for name in ['timing_build_direct.py','timing_control_run.py','timing_control_batch.py','timing_control_analyze.py','timing_control_verify.py','timing_control_report.py']:
        path=Path(__file__).resolve().with_name(name);payload[str(path.relative_to(PROJECT))]=path
    payload[str(REPORT.relative_to(PROJECT))]=REPORT
    manifest={name:sha(path) for name,path in sorted(payload.items())};archive=ASSETS/'controls-reproduction.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(payload.items()):z.write(path,name,compress_type=zipfile.ZIP_STORED if path.suffix=='.npz' else zipfile.ZIP_DEFLATED)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='worm-current-controls-') as td:
        with zipfile.ZipFile(archive) as z:assert z.testzip() is None;z.extractall(td)
        for name,digest in manifest.items():assert sha(Path(td)/name)==digest
        for name in ['timing_control_analyze.py','timing_control_verify.py']:
            subprocess.run([sys.executable,str(Path(td)/'analysis/neuromechanics/b_motor_validation'/name)],cwd=td,check=True,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1'},stdout=subprocess.DEVNULL)
        assert read(Path(td)/'results/c1-current-timing-2026-11-10/control-summary.json')==s
    digest=sha(archive);archive.with_suffix('.zip.sha256').write_text(digest+'  controls-reproduction.zip\n')
    record=dict(archive_sha256=digest,archive_bytes=archive.stat().st_size,payload_files=len(payload),CRC_passed=True,
        extracted_SHA_passed=True,isolated_extracted_reanalysis_and_independent_validation_passed=True,
        control_summary_sha256=sha(ROOT/'control-summary.json'),report_sha256=sha(REPORT),new_main_runs=0)
    (ROOT/'control-archive-validation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':main()
