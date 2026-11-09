"""Build the Chinese branch-B report from recomputed evidence, never from logs."""
import json
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'results/b-motor-feedback-causality-2026-11-09'
REPORT=REPO/'docs/reports/b-motor-feedback-causality-2026-11-09.md'
def fmt(x):return '—' if x is None else f'{x:.5g}'

def main():
    s=json.loads((ROOT/'summary.json').read_text()); numerical=json.loads((ROOT/'numerical-crosscheck.json').read_text())
    lines=['# 身体扰动与反馈重放：分支 B 因果验证', '',
      '日期：2026-11-09。模型为未做生理拟合的 B 类 Ca/K 工程原型。', '',
      f"**{sum(s['checks'].values())}/{len(s['checks'])} 项实现与因果检查通过。** 同一身体干预下，冻结无扰动基线电流会阻断神经响应；实时形变反馈会改变感觉电流与 B 活动。同一干预下重放其自身实时电流，能够逐元素复现全部神经和身体状态。", '',
      '## 固定协议与运行预算', '',
      '保持完整 302 神经元、95 肌肉、4665 条化学连接、1192 对电连接；18 个 B 细胞使用原工程原型，其余维持 C1。B→肌肉 NMJ 为 0.9 nS，AVB—B 总电导为每个 B 0.1 nS；AVBL/R 在 0.5 秒从 −60 mV 钳制至 −20 mV。神经步长 0.025 ms、身体交换 1 ms、总时长 8 秒；原形变反馈增益 100 pA/strain、截断 ±1 pA。', '',
      '**外部物理干预**为第 2000–2299 个交换步（2.0≤t<2.3 s）前 8 段背侧额外驱动 0.2。它通过原身体肌肉滤波器产生形变，仅用于因果验证，产生的运动不能算作神经自主运动。所有扰动条件的额外输入逐元素相同，实际命令未触及 ±1 截断。', '',
      '共完成 8 次新完整运行：seed13 的实时扰动、基线电流重放扰动、反馈关闭扰动、扰动实时电流自身重放；seed29 的无扰动实时基线、实时扰动及两种重放。seed13 的无扰动实时/关闭基线从旧归档复用，SHA、参数、时钟及相关源码均校验通过。第二种子不追加反馈关闭条件，保留运行预算。', '',
      '协议在新运行前保存为 `protocol.json`；密集相位参考的说明另存 `analysis-addendum.json`。没有训练、参数扫描或根据结果调整模型。', '',
      '## 因果顺序、匹配与重放检验', '',
      '每个交换步 k 从已接受的身体与神经状态计算感觉电流，或读取重放数组的第 k 行，再设置神经电流和身体驱动，随后推进身体与神经各 1 ms。重放读取来源的完整 SHA、种子、参数、B 顺序和 8001 点交换时间轴；没有用未来目标状态作控制。身体内部积分时钟始终没有超前于交换时钟。', '',
      '实时扰动、基线重放扰动和自身电流重放在 0–2.000 秒与匹配实时基线的神经、肌肉和身体状态逐元素一致。反馈关闭条件从一开始的历史不同，因此与自身关闭基线匹配检查；没有要求它与实时反馈条件在扰动前一致。', '',
      '两个种子的“扰动实时电流→同一扰动重放”均逐元素复现完整 8 秒：302 神经电位、95 肌肉电位/钙、B 门控、驱动、身体姿态、速度、两侧长度，以及密集交换记录均一致。重放实现验证通过。', '',
      '| seed | 感觉电流首次偏离基线 s | B 电位首次响应 s | 身体首次偏离无扰动基线的保存时刻 s |',
      '|---|---:|---:|---:|']
    for seed in ['13','29']:
        o=s['causal_onsets'][seed]
        lines.append(f"| {seed} | {fmt(o['applied_feedback_s'])} | {fmt(o['B_voltage_s'])} | {fmt(o['perturbed_body_vs_unperturbed_first_saved_s'])} |")
    lines += ['', '身体与基线的共享姿态/长度保存间隔为 10 ms，表中身体时刻是首次保存到差异的时刻；感觉电流及 B 电位时刻由 1 ms 记录确定。B 差异至少晚于电流差异一个交换步。身体长度的反馈来源也在每个交换步从原始长度独立重算，误差低于 10⁻¹² pA。', '',
      '下表给出 2–2.3 秒干预期相对无扰动实时基线的身体与感觉电流变化。身体应变按各段初始长度归一化，RMS 跨所有身体两侧位置与保存时刻计算。', '',
      '| seed | 身体两侧应变差 RMS | 感觉电流差 RMS pA | 最大电流差 pA |',
      '|---|---:|---:|---:|']
    for seed in [13,29]:
        d=s['comparisons'][f's{seed}_live_pert__minus__s{seed}_baseline']['during']
        lines.append(f"| {seed} | {fmt(d['side_strain_difference_rms'])} | {fmt(d['feedback_rms_pA'])} | {fmt(d['feedback_max_pA'])} |")
    lines += ['',
      '冻结基线电流时，即使身体发生外部形变，整个 8 秒神经电位和肌肉钙仍与无扰动基线逐元素相同；seed13 反馈关闭条件也有同样阻断结果。代码中身体进入神经网络的通路仅为这 18 个感觉电流。因此本组模型内的“身体干预→感觉电流→B 活动”因果通路成立。', '',
      '## 电流、B 活动与峰时分别报告', '',
      '下表比较同一扰动下实时反馈与冻结基线电流重放；RMS 均跨对应窗口和全部 18 个 B 细胞。', '',
      '| seed | 窗口 s | 感觉电流差 RMS pA | B 电位差 RMS mV | 有符号平均速度差 μm/s |',
      '|---|---|---:|---:|---:|']
    for seed in [13,29]:
        d=s['comparisons'][f's{seed}_live_pert__minus__s{seed}_replay_base_pert']
        for key,label in [('during','2–2.3'),('recovery','2.3–4'),('late','4–8')]:
            v=d[key];lines.append(f"| {seed} | {label} | {fmt(v['feedback_rms_pA'])} | {fmt(v['B_voltage_rms_mV'])} | {fmt(v['speed_difference_um_s'])} |")
    lines += ['', '峰值检测固定 prominence=3 mV、最小间距 0.25 s。密集参考来自“扰动＋重放无扰动电流”的 B 电位：其全部共享 10 ms 神经/肌肉时刻与旧基线完全一致，且固定电流后没有其他身体到神经通路，因此可作 1 ms 神经参考。原始基线 10 ms 比较仍单独保留在汇总中。', '',
      '| seed | 峰数相同且位移小于半周期的可配对细胞 /18 | 首个峰时偏移中位数 ms | 4–8 秒峰时偏移中位数 ms | 最大绝对峰时偏移 ms |',
      '|---|---:|---:|---:|---:|']
    for seed in [13,29]:
        p=s['comparisons'][f's{seed}_live_pert__minus__s{seed}_replay_base_pert']['dense_peak_timing']
        lines.append(f"| {seed} | {p['valid_matched_cells']} | {fmt(p['median_first_peak_shift_ms'])} | {fmt(p['median_late_peak_shift_ms'])} | {fmt(p['max_abs_peak_shift_ms'])} |")
    lines += ['', '上述中位数只涉及通过配对的细胞，不是 18 个 B 的整体相位。seed13 逐细胞结果如下；峰数为 2 秒之后到结束的检测峰数。', '',
              '| B 细胞 | 实时峰数 / 重放峰数 | 下一个峰事件延迟差 ms | 单一峰序列配对 |',
              '|---|---:|---:|---|']
    for p in s['comparisons']['s13_live_pert__minus__s13_replay_base_pert']['dense_peak_timing']['per_cell']:
        valid=p['ordinal_match_valid'] and not p.get('ambiguous_half_cycle',False)
        reason='可配对' if valid else ('峰数不同' if not p['ordinal_match_valid'] else '位移超过半周期')
        lines.append(f"| {p['name']} | {len(p['peak_times_a_s'])} / {len(p['peak_times_b_s'])} | {fmt(p['next_peak_latency_difference_ms'])} | {reason} |")
    lines += ['', '正号表示实时反馈峰值较晚。时间格点为 1 ms；两个峰时之差在保守量化口径下有约 2 ms 的采样不确定尺度，另有下文数值步长影响。每个细胞的完整峰时、峰数、幅度及未通过配对的原因均保存在 `summary.json`。峰数不同或位移超过半周期时不强行给单一相位偏移。', '',
      '这项单次干预检查的是峰事件时序和活动变化；电位差不自动等于纯相位变化。幅度或峰数同时改变时，不能将全部电位差归为相位重置。没有扫描多个干预相位，未测得完整相位响应曲线，也没有证明周期牵引或推进波组织。', '',
      '## 运动结果与负结果', '',
      '以下统计为 4–8 秒。净位移为身体中心两端位置之差的模；速度为沿瞬时头尾轴投影的有符号速度。', '',
      '| 条件 | 平均速度 μm/s | 中心净位移 μm | B 电位标准差均值 mV | 后4秒≥5峰细胞 /18 | 推进门槛 |',
      '|---|---:|---:|---:|---:|---|']
    labels={'baseline':'无扰动实时基线','off_baseline':'无扰动反馈关闭基线','live_pert':'扰动＋实时反馈','replay_base_pert':'扰动＋基线电流重放','off_pert':'扰动＋反馈关闭','replay_pert':'扰动＋自身电流重放'}
    for n,m in s['metrics'].items():
        seed=n.split('_')[0][1:]; label=labels[n.split('_',1)[1]]
        lines.append(f"| seed{seed} {label} | {fmt(m['late_mean_signed_speed_um_s'])} | {fmt(m['late_centroid_displacement_um'])} | {fmt(m['late_B_mean_voltage_sd_mV'])} | {sum(n>=5 for n in m['late_B_peak_count'])} | {'通过' if m['propulsion_threshold_passed'] else '未通过'} |")
    lines += ['', '推进门槛沿用平均有符号速度至少 10 μm/s、净位移至少 20 μm。扰动运动含外部做功，即使某项增加也不能作为自主推进验收。实时与重放之间的速度差在两个匹配种子中分别保存；只有两个初值种子，不作总体统计显著性或生理结论。', '',
      '## 与分支 A 的数值误差核对', '']
    if 'windows' in numerical:
        state='通过' if numerical['branch_A_convergence_accepted'] else ('未通过' if numerical['branch_A_complete'] else '尚未完成；现有配对未通过全部检查')
        lines += [f"已读取分支 A 的 {numerical['branch_A_new_runs']} 次完整运行汇总，神经步长收敛验收结果为 **{state}**。下面使用无扰动实时基线的 0.025 ms 与 {numerical['reference_dt_ms']} ms 原始轨迹差，和 seed13 同一扰动实时/基线重放效应比较。分支 A 的 0.025 ms 基线与本分支复用基线的所有数值数组逐元素一致。", '',
          '| 窗口 s | B 效应/步长差 RMS | 电流效应/步长差 RMS | 平均速度效应/步长差 |',
          '|---|---:|---:|---:|']
        for key,label in [('during','2–2.3'),('recovery','2.3–4'),('late','4–8')]:
            w=numerical['windows'][key];lines.append(f"| {label} | {fmt(w['B_effect_to_dt_difference'])} | {fmt(w['feedback_effect_to_dt_difference'])} | {fmt(w['speed_effect_to_dt_difference'])} |")
        w=numerical['windows']['late']
        lines += ['',f"4–8 秒的 B 电位效应 RMS 为 {fmt(w['B_effect_RMS_mV'])} mV，对应步长差为 {fmt(w['B_voltage_dt_RMS_mV'])} mV；感觉电流效应 RMS 为 {fmt(w['feedback_effect_RMS_pA'])} pA，对应步长差为 {fmt(w['feedback_dt_RMS_pA'])} pA。平均速度效应为 {fmt(w['mean_speed_effect_um_s'])} μm/s，步长引起的平均速度差为 {fmt(w['mean_speed_dt_difference_um_s'])} μm/s。"]
        if numerical.get('phase_cell_crosscheck'):
            p=numerical['phase_cell_crosscheck']
            lines += ['',f"对可配对细胞，晚期平均峰时偏移相对 A 同细胞最大峰时差（分母下限取 A 的 10 ms 保存间隔）的比值为 {fmt(min(v['effect_to_dt_peak_difference_with_sampling_floor'] for v in p))}–{fmt(max(v['effect_to_dt_peak_difference_with_sampling_floor'] for v in p))}。逐细胞结果保存在 `numerical-crosscheck.json`；未按偏移大小筛选可配对细胞。"]
        lines += ['',f"就现有无扰动步长配对而言，晚期 B 活动与感觉电流效应分别约为数值差异的 {fmt(w['B_effect_to_dt_difference'])} 倍和 {fmt(w['feedback_effect_to_dt_difference'])} 倍，量级分离明显；晚期平均速度差仅约 {fmt(w['speed_effect_to_dt_difference'])} 倍，不能把这项小运动差异作为已收敛的精确估计。实时反馈在本比较中的速度差为负，没有观察到推进改善。"]
        lines += ['', '这些比值衡量与现有数值差异的分离程度，不能当作扰动条件的严格误差界。A 检查无扰动系统的神经步长，仍保持身体/交换 1 ms；本分支没有额外重跑扰动条件的细步长。峰时比较还须同时考虑 A 保存的逐细胞峰时差，不能因电位效应大就自动接受精确相位数值。']
    else:
        lines += ['完成报告时，分支 A 尚无可用的步长配对结果。本分支没有自行追加数值收敛运行；0.025 ms 的精确峰时和小运动差异因此暂不能作为已收敛结论。A 完成后可重跑 `feedback_numerical.py` 和本报告生成器作只读核对。']
    lines += ['', '## 数据、来源与重算', '',
      '- 原始数据、冻结协议、全部负结果及可重算汇总：`results/b-motor-feedback-causality-2026-11-09/`。',
      '- 每个新条件：`trace.npz`（共享 10 ms 状态、1 ms 电流）、`exchange-audit.npz`（1 ms B 电位、长度、候选电流、外部/实际命令与因果时钟）、`record.json`（完整参数与 SHA）。',
      '- `source-verification.json` 记录 94 个已核对的旧归档来源；新目录保存运行源码、kernel、连接/参数、身体程序及数据快照，无需读取云端旧源。',
      '- 只读重算：使用现有环境运行 `analysis/neuromechanics/b_motor_validation/feedback_summarize.py`；数值对照运行 `feedback_numerical.py`；中文报告运行 `feedback_report.py`。所用 A 原始配对轨迹及摘要在本分支 `numerical-reference/` 保存只读副本，离线重算不依赖 A 目录。',
      '- 复现包与归档校验：`docs/reports/assets/b-motor-feedback-causality-2026-11-09/reproduction.zip`，归档 CRC、逐文件 SHA 和解压后独立重算结果见新结果目录的 `archive-validation.json`。', '',
      '历史数据、共同运行器、README、主线进度和分支 A 文件均保持原样。']
    REPORT.write_text('\n'.join(lines)+'\n')
    print(REPORT)

if __name__=='__main__':main()
