"""Render the completed fixed-budget study as a Chinese, source-linked report."""
import json
from pathlib import Path
import hashlib

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / 'results/full302-joint-numerics-2026-11-12'
REPORT = PROJECT / 'docs/reports/full302-joint-numerics-2026-11-12.md'
LABELS = {
    'B_frequency': 'B频率最大相对差', 'B_amplitude': 'B振幅最大相对差',
    'B_mean': 'B均值最大差 / mV', 'B_trajectory': 'B无平移RMS最大值 / mV',
    'B_peak_phase': 'B双向最近峰最大偏差 / s',
    'calcium_statistics': '逐肌肉钙统计归一化差', 'calcium_trajectory': '逐肌肉钙归一化RMS',
    'drive_statistics': '逐驱动通道统计归一化差', 'drive_trajectory': '逐驱动通道归一化RMS',
    'body_shape_trajectory': '内在身体形状RMS / μm', 'body_shape_amplitude': '身体横向振幅差 / μm',
    'side_strain_trajectory': '侧长应变RMS', 'mean_speed': '平均有符号速度差 / μm·s⁻¹',
    'speed_trajectory': '速度逐时RMS / μm·s⁻¹', 'centroid_displacement': '质心位移差 / μm',
    'feedback_trajectory': '反馈电流RMS / pA',
}
def read(p): return json.loads(p.read_text())
def number(x): return '—' if x is None else f'{x:.7g}'
def yes(x): return '通过' if x else '未通过'
def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    s = read(ROOT/'summary.json'); v = read(ROOT/'validation.json')
    assert s['final_budget_complete']
    assert v['summary_sha256'] == hashlib.sha256((ROOT/'summary.json').read_bytes()).hexdigest()
    if not v['independent_candidate_audit_passed']:
        assert v['incomplete_or_rhythm_failures_retained'] and not s['neural_step_convergence_accepted']
        lines=['# 完整302联合数值候选：保留失败的阶段报告','',
            f"已执行{s['new_full_runs']}个预定条件，其中{s['complete_new_runs']}条完成8秒。存在未完成轨迹或原节律有效性条件失败，不能给出整体收敛通过结论；未重跑、未调参。",'',
            '共享机制实际作用于284个通用C1神经元和95个肌肉；18个B点原型不变。8秒、seed13、0.9 nS NMJ、0.1 nS AVB耦合、100 pA/strain且±1 pA反馈、1 ms身体交换均保持锁定。','',
            table(['条件','完整运行','原节律有效','原因'],[[name,yes(m['complete']),yes(m['valid']),m.get('reason') or '—'] for name,m in s['metrics'].items()]),'',
            '原始失败、已有样本和异常记录均保留。缺失轨迹或缺失节律不会转换为通过；数值独立检查未宣称通过。完整且可比较的配对结果如下。','']
        for key,c in s['comparisons'].items():
            lines += [f'## {key}','']
            if c['comparable']:
                lines += [table(['判据','值','限','结果'],[[LABELS[g],number(e['value']),number(e['limit']),yes(c['gates'][g])] for g,e in c['errors'].items()]),'']
            else:lines += [c['reason'],'']
        lines += ['## 验证与边界','',
            '协议、来源及运行输入的SHA已经核对。失败条件阻止完整数值验收，未以总体均值或历史轨迹代替。本轮只细化神经步长，身体1 ms时钟固定；未启动额外修复、训练或新条件。','',
            '- [完整结果及失败原因](../../results/full302-joint-numerics-2026-11-12/summary.json)',
            '- [独立检查记录](../../results/full302-joint-numerics-2026-11-12/validation.json)',
            '- [原始资料归档](assets/full302-joint-numerics-2026-11-12/reproduction.zip)',
            '- [最终归档完整性与解压复算](../../results/full302-joint-numerics-2026-11-12/archive-validation.json)','']
        REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text('\n'.join(lines));return
    co, fi = s['latest_pair_ms']
    keys = [f'{mode}_{co:g}_vs_{fi:g}' for mode in ['off','live']]
    initial = ['off_0.025_vs_0.0125','live_0.025_vs_0.0125']
    moving=[name for name,m in s['metrics'].items() if m['propulsion_gate_passed']]
    motion_result=('所有条件均未达到推进要求。' if not moving else '达到工程推进门槛的条件：'+', '.join(moving)+'。')
    lines = ['# 完整302网络、95肌肉与身体：联合数值修正迁移验证', '',
             '日期：2026-11-12。候选：`joint_exp_direct`。', '',
             f"本轮共完成{s['complete_new_runs']}/{s['new_full_runs']}条新的8秒运行，无重跑。最终相邻步长{co:g}→{fi:g} ms的整体验收**{yes(s['neural_step_convergence_accepted'])}**。" +
             '运动仍按原门槛单独判断，'+motion_result, '',
             '## 回答研究问题', '']
    for mode, key in zip(['off','live'], keys):
        c=s['comparisons'][key]; ratios=s['numerical_error_migration_ratios'][key]
        ca=ratios['calcium_trajectory']; dr=ratios['drive_trajectory']
        lines += [f"- **{mode}**：最终16项中{sum(c['gates'].values())}/16项通过。钙逐时误差为旧实现同一步长配对的{number(ca['ratio'])}倍，驱动为{number(dr['ratio'])}倍。" +
                  ('未通过项：' + '、'.join(LABELS[g] for g in c['failed_gates']) + '。' if c['failed_gates'] else '本模式所有原定容差通过。')]
    lines += ['', '组件中约91%的误差下降没有作为完整模型的验收条件，也没有直接外推。这里的比值来自新候选内部步长差与旧实现内部同一步长差的比较；闭环结果与孤立组件的变化幅度需分别报告。共享机制同时作用于**284个通用C1神经元和95个肌肉**，完整模型变化无法单独归因于肌肉。', '',
              '## 冻结条件及预检', '',
              '- 302神经元、95肌肉、4665条化学连接、1192对电连接；18个B类点机制原型不变，B细胞旧密度通道gmax为零。',
              '- 每条8秒、seed13；NMJ 0.9 nS，AVB—B每个B总耦合0.1 nS；反馈off/live，100 pA/strain并截断±1 pA；身体与交换步长固定1 ms；统计窗口4–8秒。',
              '- Python3.12.14、NEURON9.0.2、NumPy2.3.5、SciPy1.16.3；固定步长secondorder=0、CVODE关闭。18个B初值与历史版一致，302电位及95肌肉钙初始数组逐位一致。',
              '- 唯一机制变更为四个MOD内9处：5个Ca/K门控及CaPool的直接导数、三条fopen0直接STATE幂读出。生成的状态和电流函数与已验证exp_direct相同；其余机制函数与父版相同。保留h_fcond、rates、NMJ缓存、CaPool同步和截断。',
              '- 运行器沿用冻结循环；只隔离路径并加外层校验／失败保存，无额外fcurrent或rates。零时刻预检未推进神经或身体。',
              f"- 运行前协议SHA256：`{s['protocol_sha256']}`。协议、来源、执行输入及运行器均保存独立哈希。", '',
              '## 运行与推进结果', '',
              '下表全部为本轮新运行。推进门槛同时要求4–8秒平均有符号速度≥10 μm/s、该窗口质心位移≥20 μm。', '']
    rows=[]
    for name,m in s['metrics'].items():
        rows.append([name, yes(m['complete']), min(m['B_peak_counts']), number(m['late_speed_um_s']),
                     number(m['late_displacement_um']), number(m['body_frame_lateral_amplitude_um']),yes(m['propulsion_gate_passed'])])
    motion_scale_small=all(abs(m['late_speed_um_s'])<.1 and m['late_displacement_um']<.5 for m in s['metrics'].values())
    lines += [table(['条件','完整8秒','18B最少峰数','有符号速度 μm/s','位移 μm','内在横向振幅 μm','推进'],rows), '',
              ('上述运动量仍接近零。' if motion_scale_small else motion_result)+'数值容差的运动差量与实际推进门槛分别判断。', '',
              '## 原16项判据逐项结果', '',
              '数值均为原定义的最差细胞／通道或指定全体RMS；未作相位平移。钙与驱动按参考通道q95−q05归一化并保留原尺度下限。表内“值≤限”逐项判断，不用群体均值覆盖失败。', '']
    for key,c in s['comparisons'].items():
        lines += [f'### {key}', '', table(['判据','值','限','结果'],
            [[LABELS[g], number(e['value']), number(e['limit']), yes(c['gates'][g])] for g,e in c['errors'].items()]), '']
    lines += ['0.025→0.0125 ms首轮触发了既定补跑规则：off未通过钙／驱动逐时误差；live还有B节律、电位与峰时、钙／驱动统计、应变和反馈项。因此为两种模式各补一条0.00625 ms，合计6条，未增加其他条件。', '',
              '### 最终相邻配对的18个B细胞', '']
    for mode,key in zip(['off','live'],keys):
        c=s['comparisons'][key];d=c['details'];a=s['metrics'][f'{mode}_dt{co:g}'];b=s['metrics'][f'{mode}_dt{fi:g}']
        rows=[]
        for i,n in enumerate(a['B_names']):
            rows.append([n,a['B_peak_counts'][i],b['B_peak_counts'][i],number(d['B_frequency_relative'][i]),
                number(d['B_amplitude_relative'][i]),number(abs(a['B_mean_mV'][i]-b['B_mean_mV'][i])),
                number(d['B_unshifted_RMS_mV'][i]),number(d['B_bidirectional_peak_offset_max_s'][i])])
        lines += [f'**{mode}**', '',table(['B细胞','粗峰数','细峰数','频率相对差','振幅相对差','均值差 mV','RMS mV','峰偏差 s'],rows),'']
    lines += ['### 最终相邻配对的肌肉与驱动失败通道', '',
              '每项阈值均为0.05。驱动共48×2个身体侧向通道；段号为1–48，D/V分别为背／腹侧。完整逐通道数组保存在summary及differences，以下列出所有超限通道。','']
    muscles=read(ROOT/'connection-plan.json')['muscles']
    for mode,key in zip(['off','live'],keys):
        d=s['comparisons'][key]['details'];rows=[]
        for field,label,names in [('calcium_scaled_stats_max_per_channel','钙统计',muscles),
                                  ('calcium_scaled_RMS_per_channel','钙逐时',muscles),
                                  ('drive_scaled_stats_max_per_channel','驱动统计',[f'{j//2+1}{"D" if j%2==0 else "V"}' for j in range(96)]),
                                  ('drive_scaled_RMS_per_channel','驱动逐时',[f'{j//2+1}{"D" if j%2==0 else "V"}' for j in range(96)])]:
            values=d[field];bad=[f'{names[i]}={number(x)}' for i,x in enumerate(values) if x>.05+1e-12]
            rows.append([label,f'{len(bad)}/{len(values)}','；'.join(bad) if bad else '无'])
        lines += [f'**{mode}**','',table(['指标','超限数','通道与数值'],rows),'']
    lines += ['## 肌肉组件改善向完整模型的迁移', '',
              '以下比值=候选步长误差÷历史旧实现相同步长误差。小于1表示这项误差变小，大于1表示变大；它不等于同一步长新旧轨迹的差。零误差分母记为—。', '']
    ratio_rows=[]
    for g in LABELS:
        ratio_rows.append([LABELS[g]] + [number(s['numerical_error_migration_ratios'][k][g]['ratio']) for k in initial+keys])
    lines += [table(['指标','off初轮','live初轮','off最终','live最终'],ratio_rows),'',
              '反馈闭环把肌肉、身体与B细胞输入连在一起，因此组件误差减少不要求闭环轨迹误差也按相同比例下降。这个机制解释与当前现象相容；本轮只比较预定候选，未分离各共享神经元与肌肉的贡献，也未检验其他原因。', '',
              '## 同一步长的新旧实现差：单独标注的模型变化', '',
              '六条旧轨迹是11月9日同参数历史对照，均独立复制并校验原始SHA；没有充当本轮新运行。下表为候选减旧实现的轨迹差或运动差，包含数值实现变化产生的总体轨迹改变，不是候选自身步长误差。','']
    rows=[]
    for name,c in s['historical_model_changes'].items():
        rows.append([name,number(c['all302_voltage_RMS_mV']),number(c['B_voltage_RMS_mV']),number(c['muscle_calcium_RMS_mM']),
                     number(c['muscle_drive_RMS']),number(c['body_frame_shape_RMS_um']),number(c['speed_mean_change_um_s']),number(c['displacement_change_um'])])
    lines += [table(['条件','302电位RMS mV','18B电位RMS mV','钙RMS mM','驱动RMS','形状RMS μm','速度变化 μm/s','位移变化 μm'],rows),'',
              '## 独立复核与交付边界','',
              f"独立NumPy计算没有导入主分析公式或NEURON：复算{v['runs_checked']}条新轨迹、{v['pair_gate_values_checked']}个配对判据值及同数目的门槛／通过状态，核对{v['migration_ratio_values_checked']}个旧新误差比、{v['historical_same_parameter_model_changes_checked']}个同参数历史对照，并重建{v['sampled_feedback_values_reconstructed']}个采样反馈值。原判据和预算未变，独立计算与主结果一致。独立复算通过不表示科学收敛通过。", '',
              '只细化神经积分步长；身体内部求解器和1 ms交换固定。状态每10 ms采样，反馈每1 ms采样；峰时验收沿用10 ms标准时标，不宣称解析了更细的相位。未拟合积分阶数、未证明所有时钟收敛。工程B类原型仍不是生理拟合；本轮没有趋温测试。'+motion_result+'本轮结束，不启动额外修复、训练或新实验。','']
    certificate=ROOT/'archive-payload-validation.json'
    if certificate.exists():
        a=read(certificate);assert a['offline_reanalysis_identical'] and a['all_payload_sha256_passed']
        lines += ['归档流程已从独立解压目录完成ZIP CRC、逐文件SHA256和离线重算；离线重算的summary与独立验证结果逐字节一致。最终ZIP的哈希和最终复算凭据写在外置`archive-validation.json`及assets校验文件中，避免归档哈希自引用。','']
    else: lines += ['归档及独立解压复算尚待完成；完成记录将写入归档校验凭据。','']
    lines += ['- [完整结果与全部逐项数据](../../results/full302-joint-numerics-2026-11-12/summary.json)',
              '- [独立数值检查](../../results/full302-joint-numerics-2026-11-12/validation.json)',
              '- [协议](../../results/full302-joint-numerics-2026-11-12/protocol.json)／[生成代码核验](../../results/full302-joint-numerics-2026-11-12/generated-code-validation.json)／[零时刻预检](../../results/full302-joint-numerics-2026-11-12/preflight.json)',
              '- [独立归档](assets/full302-joint-numerics-2026-11-12/reproduction.zip)／[最终归档验证](../../results/full302-joint-numerics-2026-11-12/archive-validation.json)', '']
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(REPORT),summary_sha256=v['summary_sha256'],final_budget_complete=True)))


if __name__=='__main__': main()
