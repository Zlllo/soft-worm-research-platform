"""Final fixed-budget 2x2 report and standalone, raw-data evidence archive."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'results/c1-current-timing-2026-11-10'
REPORT=PROJECT/'docs/reports/c1-current-timing-2026-11-10.md'
ASSETS=REPORT.parent/'assets/c1-current-timing-2026-11-10'
VARIANTS=['euler_cached','exp_cached','euler_direct','exp_direct'];STEPS=[.025,.0125,.00625]
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def f(x):return f'{x:.6g}'


def main():
    s=read(ROOT/'summary.json');v=read(ROOT/'validation.json');controls=read(ROOT/'control-summary.json')
    assert v['independent_main_audit_passed'] and v['summary_sha256']==sha(ROOT/'summary.json')
    assert s['new_main_conditions']==s['reused_main_conditions']==6
    lines=['# C1电流时序2×2隔离：指数积分与直接读出组合改善步长差', '',
        '2026-11-10。本阶段完成12个小控制、6个新增8秒孤立肌肉主条件，并复用6个已核验的历史主条件。未启动完整302、身体或训练。', '',
        '**指数积分＋五STATE直接电流读出通过预定改善条件：最细两档的电位RMS差由0.168627降至0.0156984 mV，约为原来的9.31%；钙/驱动标准化RMS差约为原来的8.6%–8.7%。Euler下仅有小幅改善，没有通过减半要求。** 四版本最细配对均通过原组件容差；不能把本组件结果记作完整系统已收敛。', '',
        '## 预定因素、预算与来源', '',
        f"执行协议v2 SHA-256：`{s['protocol_sha256']}`。协议在科学运行前锁定，控制和主条件各有独立的主线许可、执行锁及运行器SHA。v1的16观察器方案与v2修订过程均封存；最终仅执行v2的8零步＋4短控制，没有observer方案或科学失败后的重跑。", '',
        '| 积分因素 | 原缓存电流（复用） | 五STATE直接电流（新增） |', '|---|---|---|',
        '| Euler | euler_cached：11-09 original三步长 | euler_direct：三步长 |',
        '| 合并指数更新 | exp_cached：11-09 both三步长 | exp_direct：三步长 |', '',
        '每个新增版本只替换三条fopen0表达式：Ca使用e/f的STATE幂并保留原h_fcond，快K使用p/q的STATE幂，慢K使用n的STATE幂。实际生成_current只发生这些替换，states与父版一致；没有调用rates、刷新h/cai、修改CaPool同步/截断或NMJ缓存。h_q不是STATE，也不等于h_fcond。', '',
        '旧6条trace/spec/record字节及来源哈希原样复制到reused-main，原件未修改。历史secondorder未记录，按同版源码与fresh默认0推断，经主线/B接受；它仍不是历史实测字段。新增6条均实际记录secondorder=0与CVODE关闭。', '',
        '主条件保持旧C1几何、原初始化、0.9 nS NMJ、−30 mV释放阈值与已冻结的规定输入文件。输入在250–350 ms平滑进入−32±6 mV、1.9 Hz波形，属于组件刺激，不是神经自主运动。神经步长0.025/0.0125/0.00625 ms、记录0.025 ms，后4秒统计。所有共同端点输入逐元素相同，同dt四版的突触s逐元素相同；实际左端保持值也已逐条核验。', '',
        '## 控制证明了哪一项时序依赖', '',
        '8个零步控制在STATE与缓存故意不同时，实测电流分别吻合各版本应采用的公式；主动fcurrent没有改变时间、V、STATE、缓存、h及浓度，assigned电流和离子缓冲变化完整保存。另4个5 ms K-fast控制在开始仅对齐一次缓存，之后不调用fcurrent/rates。第一步两读出一致，从第二步起均有199个连续可辨样本。', '',
        '在该固定电压控制中，第j步返回的直接电流对应p_(j−1)，是电流先于状态推进的正常分裂；缓存版对应更旧的p_(j−2)。这将额外缓存状态代次与普通端点错位区分开。控制依赖真实生成代码、NEURON9.0.2调用顺序、独立零步求值及逐步数据，不是把端点cache与q不同直接判为延迟。它不表示所有电流具有一个共同物理延迟。详见control-report.md、control-summary.json及独立审计。', '',
        '## 全部步长配对结果', '',
        '沿用11-09的字节固定旧指标函数：电位RMS≤0.5 mV、均值差≤0.2 mV、幅度相对差≤5%（幅度下限1 mV）；STATE钙、cai、驱动的统计及逐时RMS标准化后≤5%，钙尺度下限1e−9 mM、驱动0.001；5门控最大RMS≤0.01。不平移曲线，所有两两配对都报告。', '',
        '| 版本 | dt ms | 电位RMS mV | STATE钙标准化RMS | 驱动标准化RMS | 门控最大RMS | 全部判据 |',
        '|---|---|---:|---:|---:|---:|---|']
    for variant in VARIANTS:
        for co,fi in [(.025,.0125),(.0125,.00625),(.025,.00625)]:
            q=s['paired_step_comparisons'][f'{variant}_{co:g}_vs_{fi:g}'];e=q['errors']
            lines.append(f"| {variant} | {co:g}→{fi:g} | "+' | '.join(f(e[k]['value']) for k in ['voltage_RMS_mV','STATE_calcium_scaled_RMS','drive_scaled_RMS','gate_state_RMS'])+f" | {'通过' if q['passed'] else '未通过：'+', '.join(q['failed_gates'])} |")
    lines+=['', '标准化RMS为相对动态范围的比例；例如0.000391对应0.0391%，并非0.391或39.1%。本组所有记录点STATE/cai一致且非负，二者配对指标相同；完整统计误差均保留在summary.json。旧exp_cached的0.025→0.00625比较仍有电位/门控失败，没有被删除。', '',
        '预定改善要求是直接读出相对同积分缓存版的最细配对误差至少减半；原误差已≤对应容差1/10可豁免。', '',
        '| 积分 | 指标 | 直接/缓存误差比 | 改善要求 |', '|---|---|---:|---|']
    for mode,d in s['improvement_by_integrator'].items():
        for name,q in d['fields'].items():
            label='豁免（原误差已小）；实际减小' if q['cached_already_tiny'] else '通过减半' if q['predeclared_improvement_requirement_passed'] else '未达减半'
            lines.append(f"| {mode} | {name} | {f(q['direct_over_cached'])} | {label} |")
    lines+=['', '指数组合的三个钙/驱动指标虽适用事先的小误差豁免，本次也确实缩小约11.5倍；不是靠豁免宣布改善。Euler电位比为0.9343，没有达到0.5要求，负结果保留。', '',
        '## 同dt的因素效应与交互', '',
        '以下均为4–8秒逐时差的电位RMS（mV），不只是幅度差。交互按`(exp_direct−exp_cached)−(euler_direct−euler_cached)`计算；不能把误差比等同于交互强度。', '',
        '| dt ms | Euler：直接−缓存 | 指数：直接−缓存 | 缓存：指数−Euler | 直接：指数−Euler | 差分交互RMS |',
        '|---:|---:|---:|---:|---:|---:|']
    for dt in STEPS:
        pairs=[('euler_direct','euler_cached'),('exp_direct','exp_cached'),('exp_cached','euler_cached'),('exp_direct','euler_direct')]
        values=[s['same_dt_factorial_contrasts'][f'{a}_minus_{b}_dt{dt:g}']['voltage_mV']['RMS'] for a,b in pairs]
        values.append(s['same_dt_interactions'][f'dt{dt:g}']['voltage_mV']['RMS'])
        lines.append(f'| {dt:g} | '+' | '.join(f(x) for x in values)+' |')
    lines+=['', '固定dt下读出因素的效应在两种积分算法中较接近，但它与各算法原有步长差叠加后的结果明显不同。差分交互随细化减小；三个步长不构成严格收敛阶或连续真解证明。一个与结果相容的解释是原Euler弛豫偏差与额外缓存代次曾部分抵消；本轮2×2没有把各项误差独立分解，不能把这个解释写成已证实的唯一机制。所有9个字段的RMS、均值差、最大差及交互保留在JSON。', '',
        '## 初始化与实际电流读出', '',
        '六组同dt直接/缓存比较的原始STATE初值最大差都为0，但自然初始化Ca电流已相差−2.584886e−7 mA/cm²。新增直接版还保存了各K电流；旧缓存主迹线未保存逐通道K初始化量，不能补造为历史观测。控制中的自然初始K电流差另有直接证据。', '',
        '按协议同时报告0–0.1、0.1–1、1–10、10–100 ms窗口；下表给出最细步长的电位差最大值（mV）。这些窗口是总读出因素作用的一部分，不能唯一识别初始化对后4秒效应贡献的比例。', '',
        '| 积分，dt=0.00625 ms | 0–0.1 ms | 0.1–1 ms | 1–10 ms | 10–100 ms |', '|---|---:|---:|---:|---:|']
    for mode in ['euler','exp']:
        key=f'{mode}_direct_minus_{mode}_cached_dt0.00625'
        values=[s['initialization_windows'][key][window]['voltage_mV']['max_abs'] for window in ['0_to_0.1_ms','0.1_to_1_ms','1_to_10_ms','10_to_100_ms']]
        lines.append(f'| {mode} | '+' | '.join(f(x) for x in values)+' |')
    worst=max(c['max_abs_error_mA_cm2'] for d in s['direct_passive_current_phase_checks'].values() for c in d['per_channel'].values())
    lines+=['', f'新增6条均保留了每个记录点前最后子步的实际V/5STATE/缓存/h等快照，推进后只读取现有电流。576万个逐通道样本与相应推进前公式相符，最大绝对差{f(worst)} mA/cm²。没有额外fcurrent/rates/观察器；回调时刻本身没有被日志化，旧细步迹线缺少相同pre-current V/cache/h字段，不能据其端点数据重建每次电流调用。', '',
        '控制原record的reversal_mV.ca读取的是seg.eca离子层Nernst值（可为1e6），不是ca_boyle内部40 mV。原记录未改；源码和零步V−I/gion均确认内部Ca=40 mV、K=−60 mV。新主元数据将内部常数来源与即时离子读数分开命名，避免将二者混用。', '',
        '## 可用结论与下一阶段边界', '',
        '- 本阶段支持将“合并指数积分＋五STATE直接电流”作为后续数值验证候选；它在本规定输入的孤立肌肉中降低了步长差，并通过原容差。Euler直接读出没有满足预定减半要求。',
        '- 结果包含自然初始化电流改变与持续读出改变的总效应；短控制隔离了持续代次依赖，不能给出其在8秒结果中的贡献比例。',
        '- 不同步长的左端输入保持宽度仍不同，配对差不等同于严格连续输入误差；未增加统一保持时钟、额外dt或参数条件。',
        '- h(cai)、CaPool负浓度同步与NMJ电流缓存均未修复；共享密度机制还会影响284个C1神经元，本次没有测试该影响。普通主迹线STATE/cai非负一致，不能把旧压力控制的问题描述成这些主条件已失稳。',
        '- 完整系统此前六次未收敛结果保持封存。本阶段没有完整302、身体或训练运行，也不自动改变3/5验收进度。后续完整系统预算由主线决定，本阶段不继续启动。', '',
        '## 验证与复现材料', '',
        f"主离线审计不导入分析器或NEURON：核对12条迹线、{v['paired_gate_values_checked']}个配对判据、{v['factorial_contrast_field_summaries_checked']}个因素字段汇总、{v['interaction_field_summaries_checked']}个交互字段、{v['early_window_field_summaries_checked']}个早期窗口字段，以及576万个电流样本。12控制另有独立审计。所有失败判据、协议历史、初始化和观测字段归属说明均保存。", '',
        '- `summary.json`/`validation.json`：本阶段2×2结果与独立复算。',
        '- `control-summary.json`/`control-validation.json`与`control-report.md`：控制阶段完整证据；较早的controls-reproduction.zip原样保留。',
        '- `main-runs/`：6个新增spec、record、原始data/extras；`reused-main/`：6个历史原始副本及来源索引，历史原件未改。',
        '- 协议、许可、运行器快照、生成代码、内核SHA、输入及补丁均在本结果目录；分析/运行代码仅在`timing_*.py`（不含B的timing_review*）。',
        '- 最终证据包：`docs/reports/assets/c1-current-timing-2026-11-10/reproduction.zip`及SHA，包含完整原始数据；约1 GB。原始数据分析只需NumPy，本机NEURON二进制不保证跨平台运行。', '',
        '在解压后的项目根目录离线重算（不启动仿真）：', '', '```sh',
        'OPENBLAS_NUM_THREADS=1 python analysis/neuromechanics/b_motor_validation/timing_control_analyze.py',
        'OPENBLAS_NUM_THREADS=1 python analysis/neuromechanics/b_motor_validation/timing_control_verify.py',
        'OPENBLAS_NUM_THREADS=1 python analysis/neuromechanics/b_motor_validation/timing_main_analyze.py',
        'OPENBLAS_NUM_THREADS=1 python analysis/neuromechanics/b_motor_validation/timing_main_verify.py',
        '```', '']
    REPORT.write_text('\n'.join(lines));(ROOT/'final-report.md').write_text(REPORT.read_text())
    if (ROOT/'execution-state.json').exists() and not (ROOT/'control-execution-state.json').exists():shutil.copy2(ROOT/'execution-state.json',ROOT/'control-execution-state.json')
    state=dict(stage='fixed_budget_stage_complete',protocol_sha256=s['protocol_sha256'],new_main_runs=6,reused_main_runs=6,
        small_controls=12,full302_body_training=0,retries=0,all_scientific_processes_finished=True,
        independent_control_and_main_audits_passed=True,more_scientific_runs_authorized=False)
    (ROOT/'execution-state.json').write_text(json.dumps(state,indent=2)+'\n')
    ASSETS.mkdir(parents=True,exist_ok=True);payload={}
    for path in ROOT.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path not in [ROOT/'archive-validation.json']:
            payload[str(path.relative_to(PROJECT))]=path
    for path in Path(__file__).resolve().parent.glob('timing_*.py'):
        if not path.name.startswith('timing_review'):payload[str(path.relative_to(PROJECT))]=path
    payload[str(REPORT.relative_to(PROJECT))]=REPORT
    manifest={name:sha(path) for name,path in sorted(payload.items())};archive=ASSETS/'reproduction.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(payload.items()):z.write(path,name,compress_type=zipfile.ZIP_STORED if path.suffix=='.npz' else zipfile.ZIP_DEFLATED)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    print('archive written; verifying isolated recomputation',flush=True)
    with tempfile.TemporaryDirectory(prefix='worm-timing-final-') as td:
        with zipfile.ZipFile(archive) as z:assert z.testzip() is None;z.extractall(td)
        for name,digest in manifest.items():assert sha(Path(td)/name)==digest
        for script in ['timing_control_analyze.py','timing_control_verify.py','timing_main_analyze.py','timing_main_verify.py']:
            subprocess.run([sys.executable,str(Path(td)/'analysis/neuromechanics/b_motor_validation'/script)],cwd=td,check=True,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1'},stdout=subprocess.DEVNULL)
        assert read(Path(td)/'results/c1-current-timing-2026-11-10/summary.json')==s
    digest=sha(archive);archive.with_suffix('.zip.sha256').write_text(digest+'  reproduction.zip\n')
    record=dict(archive_sha256=digest,archive_bytes=archive.stat().st_size,payload_files=len(payload),CRC_passed=True,
        all_extracted_SHA_passed=True,isolated_control_and_main_analysis_and_audits_passed=True,
        summary_sha256=sha(ROOT/'summary.json'),report_sha256=sha(REPORT),new_main=6,reused_main=6,controls=12,new_full302_body_training=0)
    (ROOT/'archive-validation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':main()
