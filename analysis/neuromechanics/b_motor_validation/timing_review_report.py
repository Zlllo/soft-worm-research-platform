"""Render the final independent review, keeping pause and numerical limits explicit."""
import json
from pathlib import Path
from timing_review_static import ROOT, REPO, A, REPORT, sha, write

def read(p):return json.loads(p.read_text())
def f(x):return f'{x:.7g}'
def main():
    s=read(ROOT/'main-review.json');c=read(ROOT/'control-review.json');cross=s['A_summary_crosscheck']
    assert s['data_audit_passed'] and c['control_acceptance_passed'] and cross['status']=='complete' and cross['passed']
    original=REPO/'docs/reports/c1-current-timing-2026-11-10.md';snapshot=ROOT/'A-final-report-snapshot.md'
    if not snapshot.exists():snapshot.write_bytes(original.read_bytes())
    text=snapshot.read_text()
    # Machine checks supplement, rather than replace, the explicit textual review performed by B.
    required=['0.0156984','0.9343','不能把本组件结果记作完整系统已收敛','初始化','h_fcond','p_(j−2)','0.025→0.00625']
    assert all(x in text for x in required)
    s['A_final_text_review']=dict(status='complete',snapshot='A-final-report-snapshot.md',sha256=sha(snapshot),
        scientific_text_reviewed=True,findings=[],assessment='Numbers, all-pair negative result, initializer/current limits, source/ion reversal distinction, normal splitting, non-autonomous input and no full-system acceptance are stated correctly. Candidate language is not new execution authority; current user request is finish existing artifacts then pause.')
    s['review_status']='complete';s['stage_control']=dict(user_requested_pause=True,pause_effective_after_archive_verification=True,new_scientific_runs_authorized=0,next_stage_started=False)
    write(ROOT/'main-review.json',s)
    m=c['summary_metrics'];max_current=max(q['max_abs_error_mA_cm2'] for d in s['current_reconstruction'].values() for q in d['channels'].values())
    lines=['# C1电流时序2×2隔离：最终独立复核', '',
        '日期：2026-11-10。**现有数据的执行与计算审查通过；指数积分＋五STATE直接读出在本孤立肌肉输入下显著减少步长差，Euler直接读出未满足预定减半要求。** 本分支仅做源码、代数及保存数据复算，没有导入 NEURON、编译或启动科学仿真。按用户“完成目前后暂停”的要求，本轮在报告及可重算归档验收后暂停，不启动下一阶段。', '',
        '## 执行、输入与复用', '',
        'v2 协议 SHA：`'+s['protocol_sha256']+'`。只执行6条新8秒孤立肌肉主条件，复用6条已核对的原始主基线；另有8零步与4短快K控制。新主条件目录、记录、执行锁、实际神经步数及批次状态一致，重试为0；完整302、身体和训练新增运行均为0。v1/v2协议及既有审查文件原样保留。', '',
        '新增运行器源码与锁定 SHA 一致：原32列顺序、输入设置和推进/记录顺序保留，只增加被动快照；没有额外 fcurrent、rates、observer 或初始化处理。所有12条轨迹的320001个共同端点输入及实际上一子步输入与冻结文件逐元素一致。同dt四组合的突触s按原字节哈希一致。几何、NMJ与参数未改，输入是规定的组件刺激，不是自主神经运动。', '',
        '旧 secondorder 缺少运行时字段，仍按已披露的同版本源码/新进程默认值证据接受复用；新6条均实测 secondorder=0、CVODE关闭。不能将该推断补写成旧数据的直接测量。', '',
        '## 控制与实际电流', '',
        f"实际生成函数只改三处电流表达式中的五个 STATE 幂，父版状态更新、初始化、rates、CaPool、NMJ及 h_fcond 均保留。8零步控制不变量最大变化为0，逐通道公式最大误差 {f(m['zero_current_formula_max_error_mA_cm2'])} mA/cm²。4个短K控制每条均有199个连续可区分步：正常直接读出对应 p_(j−1)，缓存读出对应 p_(j−2)。这是额外门控状态代次依赖，不是整个电流或生物系统的统一延迟。", '',
        f"新主数据保存了每个端点前最后子步的实际 V、5STATE、缓存、h及浓度。6条新迹线的逐Ca/快K/慢K电流可由这些推进前快照重建，最大误差 {f(max_current)} mA/cm²。重建采用该次电流可用的状态与V，没有用推进后的端点量代替。旧细步基线缺少同等 pre-current V/cache/h，因此未补造旧逐通道K电流或完整调用时钟。", '',
        '原控制 metadata.reversal_mV.ca 记录的是离子层 seg.eca，不是 Ca 通道使用的私有40mV。生成代码与8零步 V−I/gion 独立验证了40mV；原记录保持不变，分析明确区分。新主记录已分开通道内部常数与即时离子层读数。h_q不是STATE，电流仍乘 h_fcond；本次没有刷新 h(cai)、CaPool同步或NMJ缓存。', '',
        'NEURON原生分裂的解释依据已核对的[9.0.2固定步长实现](https://github.com/neuronsimulator/nrn/blob/9.0.2/src/nrnoc/fadvance.cpp)与[官方程序控制说明](https://www.neuronsimulator.org/en/9.0.2/progref/simctrl/programmatic.html)。不能将secondorder=2的二阶半步精度定义直接套到本次secondorder=0。', '',
        '## 全部步长配对', '',
        '沿用原0.025、0.0125、0.00625ms、4–8秒窗口、逐时比较、统计尺度与不平移规则。下列钙/驱动为相对细步参考q95−q05动态范围（含原尺度下限）的比例，不是绝对浓度或力误差。', '',
        '| 组合 | 配对 ms | V RMS mV | STATE钙缩放RMS | cai缩放RMS | 驱动缩放RMS | 全部判据 |',
        '|---|---|---:|---:|---:|---:|---|']
    for variant in ['euler_cached','exp_cached','euler_direct','exp_direct']:
        for coarse,fine in [(.025,.0125),(.0125,.00625),(.025,.00625)]:
            q=s['pairs'][f'{variant}_{coarse:g}_vs_{fine:g}'];d=q['metrics']
            lines.append(f"| {variant} | {coarse:g}→{fine:g} | {f(d['voltage_mV']['RMS'])} | {f(d['concentration_mM']['scaled_RMS'])} | {f(d['cai_mM']['scaled_RMS'])} | {f(d['drive_cai']['scaled_RMS'])} | {'通过' if q['passed'] else '未通过：'+', '.join(q['failed_gates'])} |")
    lines += ['', '所有组合最细配对通过原组件容差。旧 exp_cached 的0.025→0.00625ms仍未通过电位RMS及门控RMS，负结果完整保留。', '',
        '| 积分方式 | V误差直接/缓存 | 钙误差直接/缓存 | 驱动误差直接/缓存 | 预定改善要求 |',
        '|---|---:|---:|---:|---|']
    for alg in ['euler','exp']:
        d=s['readout_improvement'][alg];x=d['fields']
        lines.append(f"| {alg} | {f(x['voltage_mV']['direct_over_cached'])} | {f(x['concentration_mM']['direct_over_cached'])} | {f(x['drive_cai']['direct_over_cached'])} | {'通过' if d['all_improvement_rules_passed'] else '未通过电位至少减半'} |")
    exp=s['readout_improvement']['exp']['fields'];eu=s['readout_improvement']['euler']['fields']
    lines += ['',f"指数积分下，电位差从 {f(exp['voltage_mV']['cached_error'])} 降至 {f(exp['voltage_mV']['direct_error'])} mV，缩小约 {f(1/exp['voltage_mV']['direct_over_cached'])} 倍；钙与驱动约缩小 {f(1/exp['concentration_mM']['direct_over_cached'])} 和 {f(1/exp['drive_cai']['direct_over_cached'])} 倍。这几项实际都超过减半，不依赖小原误差豁免。Euler电位仅降低约 {(1-eu['voltage_mV']['direct_over_cached'])*100:.4g}%，未通过要求；钙/驱动的小误差豁免不被描述为显著改善。", '',
        '## 同dt差与交互', '',
        '逐时先形成四条因素对照和差分交互，再计算统计；交互为 (exp_direct−exp_cached)−(euler_direct−euler_cached)，没有拿RMS之差充当交互。下表为4–8秒的电位RMS（mV）。', '',
        '| dt ms | Euler直接−缓存 | 指数直接−缓存 | 缓存指数−Euler | 直接指数−Euler | 交互 |',
        '|---:|---:|---:|---:|---:|---:|']
    for dt,g in s['factorial'].items():
        fields=['euler_direct_minus_cached','exp_direct_minus_cached','cached_exp_minus_euler','direct_exp_minus_euler','interaction']
        lines.append('| '+dt[2:]+' | '+' | '.join(f(g[key]['late_4000_8000ms']['voltage_mV']['RMS']) for key in fields)+' |')
    lines += ['', '相同dt下读出信号效应在两种算法中相近，但其与各自步长误差组合后的结果不同。这里的交互和误差缩小不等于已经分解所有误差来源，也不是严格收敛阶或连续真解证明。不同dt的左端保持宽度仍不同。', '',
        '## 初始化与复算一致性', '',
        '12条主条件自然初始化的V、5STATE、缓存、h和浓度逐元素相同；Ca电流从缓存版0变为直接版约−2.584886×10⁻⁷mA/cm²。直接版保存的初始快K/慢K电流也不同于控制中的缓存值；旧主数据没有逐通道K，未当作历史观测补写。启动窗口和晚期结果均是总读出因素效应，不能进一步分配初始化与持续依赖各自的贡献比例。', '',
        '独立初期窗口采用左闭右开，A采用两端都含。已额外按A约定复算648项窗口统计并完全吻合；两种约定及结果分开保存，不改模型或判据。主条件全部STATE/cai非负，且两者最大差为 '+f(max(x['max_cai_STATE_difference_mM'] for x in s['metrics'].values()))+'mM；旧外流压力段同步问题不被描述为本次主条件失稳。', '',
        f"独立结果与A汇总 SHA `{cross['A_summary_sha256']}` 核对：120个配对指标、324个同dt统计、648个闭区间初始窗口统计、120个原生初始STATE/缓存值均一致；81个交互统计最大差 {f(cross['signal_level_interaction_metrics']['max_abs_difference'])}，属浮点求和顺序尺度。独立计算完成后才读取A汇总，没有导入其分析函数。A最终中文报告的数值、负结果、初始化及当前/端点边界已人工审阅，文本快照和SHA保存。", '',
        '## 交付、归档与暂停', '',
        '本结果只支持本规定输入下的组件数值机制与组合候选。共享机制对284个C1神经元的影响、完整302/身体收敛、原h/CaPool/NMJ其余时序、神经自主推进及趋温均未验证。此前完整系统未通过的结果和3/5验收口径不变。', '',
        '按用户要求，仅完成现有数据复算、报告和归档后暂停。没有新条件、额外dt、参数搜索、完整系统运行或下一阶段任务。归档验收和暂停生效记录分别见 `archive-validation.json` 与 `stage-pause.json`。', '',
        '- 主复核：`results/c1-current-timing-review-2026-11-10/main-review.json`。',
        '- 独立原始计算：`main-review-calculations.json`；A实际比对：`A-summary-crosscheck.json`；控制验收：`control-review.json`。',
        '- A最终文字和汇总快照、所有原始SHA及旧审查/工具失败记录均保留。',
        '- 自包含复算包：`docs/reports/assets/c1-current-timing-review-2026-11-10/reproduction.zip`；包括主/控制原始数据、执行源和内核、输入及独立审查脚本。验证仅运行只读计算，不加载NEURON。']
    REPORT.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(REPORT),A_text_sha256=sha(snapshot),review_status='complete',pause_after_archive=True)))

if __name__=='__main__':main()
