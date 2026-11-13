"""Render the scoped Chinese review from independently saved results."""
import json
from joint_review_compute import REPO, OUT, A, OLD, read, save, digest

REPORT=REPO/'docs/reports/full302-joint-numerics-review-2026-11-12.md'
LABELS={
    'B_frequency':'B 频率相对差','B_amplitude':'B 振幅相对差','B_mean':'B 均值差 / mV',
    'B_trajectory':'B 无平移 RMS / mV','B_peak_phase':'B 双向峰时差 / s',
    'calcium_statistics':'钙逐通道统计缩放误差','calcium_trajectory':'钙逐通道轨迹缩放 RMS',
    'drive_statistics':'驱动逐通道统计缩放误差','drive_trajectory':'驱动逐通道轨迹缩放 RMS',
    'body_shape_trajectory':'内在横向形状 RMS / μm','body_shape_amplitude':'内在横向振幅差 / μm',
    'side_strain_trajectory':'侧长应变轨迹 RMS','mean_speed':'平均有符号速度差 / μm/s',
    'speed_trajectory':'速度轨迹 RMS / μm/s','centroid_displacement':'质心位移差 / μm',
    'feedback_trajectory':'反馈电流轨迹 RMS / pA'}


def value(x):return f'{x:.8g}'


def main():
    source=read(OUT/'source-protocol-review.json');parent=read(OUT/'parent-crosscheck.json')
    own=read(OUT/'independent-results.json') if (OUT/'independent-results.json').exists() else None
    cross=read(OUT/'A-summary-crosscheck.json') if (OUT/'A-summary-crosscheck.json').exists() else None
    historical=read(OUT/'historical-comparison.json') if (OUT/'historical-comparison.json').exists() else None
    done=bool(own and own['adaptive_batch_complete'] and cross and cross['passed'] and source['all_available_checks_pass'] and not source['generated_code_pending'])
    status=dict(stage='independent_data_review_complete' if done else ('awaiting_candidate_raw' if not source['generated_code_pending'] else 'awaiting_candidate_raw_and_generated_preflight'),
        source_review_complete=source['source_review_complete'],available_source_checks_pass=source['all_available_checks_pass'],
        generated_code_pending=source['generated_code_pending'],parent_formula_crosscheck_passed=parent['passed'],
        candidate_raw_runs_reviewed=own['new_full_runs'] if own else 0,
        candidate_operational_acceptance=own['neural_step_convergence_accepted'] if own else None,
        scientific_runs_started_by_B=0,simulator_imported_by_B=False,compiled_by_B=False,
        other_chat_messages_sent=0)
    save(OUT/'review-status.json',status)
    lines=['# 完整302网络与身体联合数值修正：独立复核','',
        '本报告来自分支 B 的源码只读检查与原始 NPZ 独立复算。本分支未加载 NEURON、编译或启动科学仿真，也未修改 A 或历史结果。','']
    if own:
        if own['adaptive_batch_complete']:
            lines += [f"候选已审阅 {own['new_full_runs']} 条运行记录，其中 {own['valid_full_runs']} 条通过轨迹有效性检查。按原协议的最终相邻步长对 {own['latest_pair_ms']} ms，操作性收敛验收{'通过' if own['neural_step_convergence_accepted'] else '未通过'}。",'']
            unresolved=[]
            for mode in ('off','live'):
                key=f"{mode}_{own['latest_pair_ms'][0]:g}_vs_{own['latest_pair_ms'][1]:g}"
                if key in own['comparisons'] and own['comparisons'][key]['failed_gates']:
                    unresolved.append(mode+'：'+'、'.join(LABELS[k] for k in own['comparisons'][key]['failed_gates']))
            if unresolved:lines += ['最终未通过项为'+'；'.join(unresolved)+'。','']
            if historical:
                transfer=[]
                for mode in ('off','live'):
                    key=f"{mode}_{own['latest_pair_ms'][0]:g}_vs_{own['latest_pair_ms'][1]:g}"
                    ratios=historical['candidate_over_parent_step_errors'].get(key,{})
                    if all(k in ratios for k in ('calcium_trajectory','drive_trajectory')):
                        calcium=ratios['calcium_trajectory']['candidate_over_parent']
                        drive=ratios['drive_trajectory']['candidate_over_parent']
                        transfer.append(f"{mode} 模式钙／驱动轨迹缩放 RMS 为历史父版相同步长对的 {value(calcium)}／{value(drive)} 倍")
                if transfer:lines += ['改善迁移按完整系统实际数值判断：'+'；'.join(transfer)+'。这些是各自参考尺度下的步长误差倍率，不是同 dt 的新旧模型轨迹差。','']
            if own['metrics'] and not any(m['propulsion_gate_passed'] for m in own['metrics'].values()):
                lines += ['所有有效候选运行均未达到持续推进门槛；数值变化尚未带来原标准下的推进成功。','']
        else:
            lines += [f"候选已审阅 {own['new_full_runs']} 条运行记录，预定批次尚未完整；当前不作最终收敛结论。",'']
    else:
        lines += ['当前仅完成源码／协议复核和历史数据计算交叉核对；新候选原始轨迹尚未就绪，不能判断改善是否迁移。','']
    lines += ['## 冻结范围与实现核对','',
        f"四个机制文件与 11 月 10 日 exp_direct 候选一致；相对 11 月 9 日父版恰有九处允许改动：五个 Ca/K 门变量及 CaPool 改用直接导数，三处 fopen0 改为 STATE 幂。其余机制、HOC、完整运行器、网络与身体实现、参数和身体二进制均按文件 SHA 核对。当前可用源码检查{'全部通过' if source['all_available_checks_pass'] else '存在未通过项'}。",'',
        '共享密度机制的实际作用范围是 **284 个通用 C1 神经元及 95 个肌肉**。18 个 B 类工程原型的密度 gmax 关闭，其 BMotorPoint 与配置保持原样。因此本轮完整模型变化不能只归因于肌肉。','',
        'h(cai) 的赋值式抑制因子、NMJ 电流缓存、CaPool 的 cai 同步及负浓度截断顺序保留。Ca 通道私有反转电位仍为 40 mV；离子缓冲区的 eca 不能代替它。初始化语句不变，直接 STATE 读出造成的原生初始电流差属于候选总体变化。','',
        '新包装层仅固定路径、在构造后断言人口与默认 secondorder=0，并在异常后保存已有样本。冻结积分循环逐字一致，没有额外 fcurrent、rates、推进调用或初始化钩子。', '']
    if source['generated_code_pending']:
        lines += ['实际新生成代码仍待齐备：'+', '.join(source['generated_code_pending'])+'。当前源码通过不能替代此项。','']
    else:
        lines += ['四个新生成机制的 STATE、电流、rates 与初始化函数已独立提取，与已验证的 exp_direct 生成函数逐项核对。','']
    lines += ['## 判据与独立性','',
        '验收数值、4–8 s 窗口和定义均沿用 11 月 9 日。先 off/live × 0.025/0.0125 ms，共四条；任一模式任一判据失败，双模式都追加 0.00625 ms，共最多六条，不重跑。新协议仅把不完整／无效轨迹也触发追加写得更明确，参考成员仍为更细的相邻步长。','',
        '18 个 B 细胞分别计算频率、分位振幅、均值、原位 RMS 和双向近邻峰时；相位使用原定 10 ms 采样刻度，另查实际神经时钟漂移，不做平移拟合。95 肌肉和 96 驱动槽逐通道计算均值、标准差、q05、q95 及轨迹误差。形状先去质心再按实际头尾轴转到内在坐标；近零速度、位移、应变和电流保留绝对门槛。','',
        '独立程序直接读取原始 NPZ，不导入 A 分析函数。另重建肌肉钙→驱动及身体侧长→感受应变→反馈电流，并核对人口、连接、有限值、电压范围、时钟、SHA 和来源锁。', '',
        '时钟证据分两层：原始数据可复查实际神经记录时刻与声明的交换刻度；逐交换身体同步和 IDA 不超前依赖冻结运行器中的逐步断言及 A 的零步预检。NPZ 未保存逐交换身体／IDA 时刻，B 没有从原始数据独立重建这一未记录序列。', '',
        f"历史父版六条轨迹的 192 个判据数值已交叉核对，最大浮点差 {parent['max_abs_numeric_difference']:.5g}；全部判定相同。该检查验证复算实现，历史轨迹不充当新候选基线。",'']
    if own:
        lines += ['## 候选自身的步长比较','',
            f"四条初始运行是否触发更细步长：{own['fine_required']}；4+2 批次是否完整：{own['adaptive_batch_complete']}。",'']
        for key,pair in own['comparisons'].items():
            lines += [f'### {key}','', '| 判据 | 误差 | 原定上限 | 判定 |','|---|---:|---:|---|']
            for metric,item in pair['errors'].items():
                lines.append(f"| {LABELS[metric]} | {value(item['value'])} | {value(item['limit'])} | {'通过' if item['passed'] else '**未通过**'} |")
            lines += ['']
            mode=key.split('_')[0];coarse,fine=key.split('_')[1],key.split('_')[-1]
            metric=own['metrics'][f'{mode}_dt{coarse}'];detail=pair['details']
            if 'B_bidirectional_peak_offset_max_s' in detail:
                phase=detail['B_bidirectional_peak_offset_max_s'];worst=max(range(18),key=lambda j:phase[j])
                counts=detail['phase_peak_counts'][worst]
                lines += [f"最大双向近邻峰时差出现在 {metric['B_names'][worst]}，4–8 s 峰数分别为 {counts[0]}／{counts[1]}。该值按峰序列逐向匹配计算，未解释为统一时间延迟。",'']
            if 'calcium' in detail:
                j=detail['calcium']['worst_trajectory_channel']
                k=detail['drive']['worst_trajectory_channel']
                lines += [f"钙轨迹最差肌肉为 {metric['muscle_names'][j]}；驱动轨迹最差为第 {k//2+1} 个身体段的{'背侧' if k%2==0 else '腹侧'}槽。完整逐通道统计与 RMS 保存在独立计算 JSON 中。",'']
        lines += ['## 运动与数值验收分别判断','',
            '| 模式、dt / ms | 4–8 s 平均速度 / μm/s | 4–8 s 质心位移 / μm | 持续推进门槛 |','|---|---:|---:|---|']
        for n,m in own['metrics'].items():
            lines.append(f"| {n} | {value(m['late_speed_um_s'])} | {value(m['late_displacement_um'])} | {'通过' if m['propulsion_gate_passed'] else '未通过'} |")
        lines += ['','推进要求有符号速度 ≥10 μm/s 且位移 ≥20 μm。速度差很小或收敛通过，并不满足这一运动标准。','']
    if historical and own and own['adaptive_batch_complete']:
        lines += ['## 历史父版与新候选的区别','',
            '同 dt 的历史父版／候选差是整体模型变化；候选内部相邻 dt 差才用于本次验收。以下保留最终步长对的关键误差，倍率为新候选／历史父版，尺度分别按各自细步长参考计算。','',
            '| 模式 | 指标 | 父版误差 | 候选误差 | 候选／父版 |','|---|---|---:|---:|---:|']
        for mode in ('off','live'):
            key=f"{mode}_{own['latest_pair_ms'][0]:g}_vs_{own['latest_pair_ms'][1]:g}"
            for metric in ('B_trajectory','B_peak_phase','calcium_trajectory','drive_trajectory','feedback_trajectory'):
                q=historical['candidate_over_parent_step_errors'][key][metric]
                ratio=value(q['candidate_over_parent']) if q['candidate_over_parent'] is not None else '不适用（父版近零）'
                lines.append(f"| {mode} | {LABELS[metric]} | {value(q['parent'])} | {value(q['candidate'])} | {ratio} |")
        lines += ['']
    if cross:
        lines += [f"独立结果与 A 汇总交叉核对：{cross['numeric_scalars_compared']} 个判据数值，最大差 {cross['max_abs_numeric_difference']:.5g}；全部判定一致。",'']
    lines += ['## 结论范围与可复查材料','',
        '本轮只细化神经 dt，身体／交换 1 ms 及 IDA 设置冻结。无论最终是否通过，都仅适用于预定操作性容差；没有验证所有时钟的收敛或渐近阶数。18 个 B 仍为工程原型，本轮没有开展参数搜索、训练或新的实验阶段。','',
        '来源：[任务交接](../full302-joint-numerics-2026-11-12.md)、[独立源码／协议复核](../../results/full302-joint-numerics-review-2026-11-12/source-protocol-review.json)、[历史复算交叉核对](../../results/full302-joint-numerics-review-2026-11-12/parent-crosscheck.json)、[工具失败记录](../../results/full302-joint-numerics-review-2026-11-12/review-attempts.json)。']
    if own:lines += ['','[候选独立复算](../../results/full302-joint-numerics-review-2026-11-12/independent-results.json)']
    if historical:lines += ['','[历史／候选分开比较](../../results/full302-joint-numerics-review-2026-11-12/historical-comparison.json)']
    if cross:lines += ['','[A 汇总交叉核对](../../results/full302-joint-numerics-review-2026-11-12/A-summary-crosscheck.json)']
    if done:lines += ['','复算包保留独立程序、计算结果和来源检查，原始轨迹引用已冻结的候选／父版归档；[依赖归档与 SHA256](../../results/full302-joint-numerics-review-2026-11-12/archive-dependencies.json)。解压后的复算仅使用 NumPy／SciPy，禁止导入 NEURON 和读取原工作区。最终 ZIP 哈希及实际验证日志另存包外，避免自引用。']
    REPORT.write_text('\n'.join(lines)+'\n')
    print(json.dumps(status,ensure_ascii=False))


if __name__=='__main__':main()
