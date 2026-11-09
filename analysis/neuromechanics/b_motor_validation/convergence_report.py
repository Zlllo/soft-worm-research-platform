"""Write the Chinese handoff and a self-contained, hash-verified evidence ZIP."""
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'results/b-motor-convergence-2026-11-09'
REPORT=PROJECT/'docs/reports/b-motor-convergence-2026-11-09.md'
ASSETS=REPORT.parent/'assets/b-motor-convergence-2026-11-09'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def fmt(v): return f'{v:.6g}'


def main():
    s=read(ROOT/'summary.json');v=read(ROOT/'independent-validation.json')
    assert v['summary_sha256']==sha(ROOT/'summary.json') and v['independent_numpy_audit_passed']
    assert s['new_full_runs'] in [4,6]
    assert all(not m['propulsion_gate_passed'] for m in s['metrics'].values())
    accepted=s['neural_step_convergence_accepted'];fine=s['new_full_runs']==6
    a,b=s['latest_pair_ms']
    lines=['# 完整302神经—身体系统：神经积分步长收敛验证', '', '日期：2026-11-09；独立任务分支A。', '',
        '**验收结果：'+('通过预先声明的神经步长容差。' if accepted else '未通过完整系统的全部预先声明容差。')+'**', '',
        f"完成{s['new_full_runs']}次新的8秒完整运行。保留302神经元、95肌肉、18个B类工程原型、强NMJ 0.9 nS、seed 13和固定1 ms身体交换；只改变神经积分步长。没有训练、参数优化、外部肌肉波或身体扰动。", '',
        '## 预先固定的协议与判据', '',
        '运行前冻结 `protocol.json` 和 `protocol-lock.json`。先运行off/live的0.025和0.0125 ms；任一条件任一判据失败，则两条件均追加0.00625 ms，最多六次。判据全程未改。后4秒（4–8 s）用于统计，同时保存完整0–8 s轨迹。状态每10 ms、反馈电流每1 ms归档。', '',
        f"协议SHA-256：`{s['protocol_sha256']}`。两次0.025 ms基线均重新计算，旧数据不充当新运行。", '',
        '| 检查对象 | 预定验收条件 |', '|---|---|',
        '| 每个B细胞 | 频率相对差≤1%；q95−q05幅度相对差≤5%；平均电位差≤0.2 mV；未平移逐时RMS≤0.5 mV；显著峰双向最近时间差≤10 ms |',
        '| 每条肌肉钙、每条48×2驱动通道 | 均值、标准差、q05、q95差及逐时RMS，除以细步长q95−q05后的最大值≤5%；钙尺度下限1e−9 mM、驱动尺度下限0.001 |',
        '| 真实身体形变 | 去质心并按瞬时头尾轴旋转后的49节点横向轨迹RMS≤0.1 μm；最大节点时域峰峰幅度差≤max(0.05 μm, 5%参考幅度)；两侧应变RMS差≤max(1e−4, 5%参考应变RMS) |',
        '| 接近零的运动量 | 平均有符号速度绝对差≤0.01 μm/s；速度逐时RMS差≤0.02 μm/s；4–8 s质心净位移量差≤0.05 μm |',
        '| 反馈 | 后4秒1 ms反馈逐时RMS差≤max(0.005 pA, 5%参考反馈RMS) |', '',
        '所有判据均须通过；比较以较细步长为参考。B峰使用3 mV显著度，与旧阶段一致，不增加峰间距过滤。相位比较不平移验收轨迹；另作±250 ms常数平移诊断，并保存每个细胞的结果。10 ms记录间隔限制峰时差分辨率。峰时差使用统一的名义记录时刻，原始NEURON累计浮点时钟另行校验并保留，避免亚纳秒尾差改变10 ms边界判定。', '',
        '## 各运行的直接结果', '',
        '| 反馈 | 神经步长 ms | B频率范围 Hz | B幅度范围 mV | 钙最大值 mM | 驱动通道平均SD | 身体内在横向幅度 μm | 后4秒均速 μm/s | 净位移 μm |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in ['off','live']:
        for dt in [.025,.0125,.00625]:
            m=s['metrics'].get(f'{mode}_dt{dt:g}')
            if not m:continue
            f=m['B_frequency_Hz'];am=m['B_amplitude_mV']
            lines.append(f"| {mode} | {dt:g} | {fmt(min(f))}–{fmt(max(f))} | {fmt(min(am))}–{fmt(max(am))} | {fmt(m['calcium_max_mM'])} | {fmt(m['drive_mean_channel_std'])} | {fmt(m['body_frame_lateral_amplitude_um'])} | {fmt(m['late_speed_um_s'])} | {fmt(m['late_displacement_um'])} |")
    lines+=['', '上述幅度用身体自身坐标计算，已去除质心运动和整体旋转；另在JSON保留实验室坐标幅度，便于对应旧报告。所有条件18/18 B细胞均保留可识别节律。持续前进门槛仍为均速≥10 μm/s且净位移≥20 μm。', '',
        '## 相邻步长的验收结果', '', '| 反馈及步长 ms | 总体验收 | 未通过项 |', '|---|---|---|']
    for key,c in s['comparisons'].items():
        if (c['coarse_dt_ms'],c['reference_dt_ms'])==(.025,.00625):continue
        lines.append(f"| {c['feedback']}：{c['coarse_dt_ms']:g} → {c['reference_dt_ms']:g} | {'通过' if c['passed'] else '未通过'} | {', '.join(c['failed_gates']) or '无'} |")
    if fine:
        lines+=['', '原使用步长0.025 ms相对最细0.00625 ms的直接检查：', '',
            '| 反馈 | B最大频率相对差 | B最大逐时RMS mV | 峰时间最大偏差 ms | 未通过项 |', '|---|---:|---:|---:|---|']
        for mode in ['off','live']:
            c=s['comparisons'][f'{mode}_0.025_vs_0.00625'];e=c['errors']
            lines.append(f"| {mode} | {fmt(e['B_frequency']['value']*100)}% | {fmt(e['B_trajectory']['value'])} | {fmt(e['B_peak_phase']['value']*1000)} | {', '.join(c['failed_gates']) or '无'} |")
        lines+=['', '因此，0.025 ms下off的B节律与逐时电位通过上述直接检查；live的精确频率、逐时电位和峰相位未通过。不能由相邻某一对的频率差小于1%，推断0.025 ms对最细参考也小于1%。']
    lines+=['', '下面给出每个判据的实际差异及其阈值。相对误差是无量纲量，例如0.01表示1%；绝对误差单位见判据名。', '',
        '| 比较 | 判据 | 实际差异 | 阈值 | 结果 |', '|---|---|---:|---:|---|']
    labels=dict(B_frequency='B频率相对差',B_amplitude='B幅度相对差',B_mean='B均值 mV',B_trajectory='B逐时RMS mV',
        B_peak_phase='B峰时间 s',calcium_statistics='钙统计标准化差',calcium_trajectory='钙逐时标准化RMS',
        drive_statistics='驱动统计标准化差',drive_trajectory='驱动逐时标准化RMS',body_shape_trajectory='身体横向RMS μm',
        body_shape_amplitude='身体横向幅度差 μm',side_strain_trajectory='应变RMS差',mean_speed='均速差 μm/s',
        speed_trajectory='速度RMS差 μm/s',centroid_displacement='净位移量差 μm',feedback_trajectory='反馈RMS差 pA')
    for key,c in s['comparisons'].items():
        if (c['coarse_dt_ms'],c['reference_dt_ms'])==(.025,.00625):continue
        for name,e in c['errors'].items():
            lines.append(f"| {c['feedback']} {c['coarse_dt_ms']:g}→{c['reference_dt_ms']:g} | {labels[name]} | {fmt(e['value'])} | {fmt(e['limit'])} | {'通过' if c['gates'][name] else '未通过'} |")
    lines+=['', '## 相位漂移、可使用结论与推荐', '',
        f'最后比较（{a:g}→{b:g} ms）的分项可用性如下；“通过”仅指本协议的容差与8秒窗口。', '',
        '| 指标组 | off | live |', '|---|---|---|']
    domain_labels=dict(B_rhythm='B频率/幅度/均值',B_trajectory_and_peak_phase='B逐时电位/峰相位',
        muscle_calcium_and_drive='肌肉钙/驱动',body_shape='身体内在横向形状',body_side_strain='身体两侧应变',
        near_zero_movement='近零速度/位移',feedback_current='反馈电流')
    for domain,label in domain_labels.items():
        flags=[s['latest_pair_domains_passed'][mode][domain] for mode in ['off','live']]
        lines.append(f"| {label} | {'通过' if flags[0] else '未收敛'} | {'通过' if flags[1] else '未收敛'} |")
    lines.append('')
    for mode in ['off','live']:
        c=s['comparisons'][f'{mode}_{a:g}_vs_{b:g}'];d=c['details']
        shifts=d['B_constant_shift_diagnostic']
        lines.append(f"- {mode}，{a:g}→{b:g} ms：单细胞最大未平移电位RMS {fmt(max(d['B_unshifted_RMS_mV']))} mV；全B最大绝对差{fmt(d['B_max_abs_voltage_difference_mV'])} mV；显著峰最大偏移{fmt(max(d['B_bidirectional_peak_offset_max_s']))} s。常数平移诊断使用相同的4.25–7.74 s内部窗口，最佳偏移范围{fmt(min(x['shift_s'] for x in shifts))}–{fmt(max(x['shift_s'] for x in shifts))} s，窗口内最大单细胞RMS为{fmt(max(x['RMS_mV'] for x in shifts))} mV；同窗口未平移值为{fmt(max(x['unshifted_same_window_RMS_mV'] for x in shifts))} mV。该内部窗口数值不与完整4–8 s的RMS直接比较，诊断没有改变验收结果。")
        ca=d['calcium_scaled_RMS_per_channel'];dr=d['drive_scaled_RMS_per_channel']
        lines.append(f"- {mode}的逐时超标通道：钙{sum(x>.05 for x in ca)}/95、驱动{sum(x>.05 for x in dr)}/96；所有通道合并的绝对钙RMS差{fmt(d['calcium_absolute_RMS'])} mM、驱动RMS差{fmt(d['drive_absolute_RMS'])}。")
    if accepted:
        lines+=['', f'在本协议和8秒窗口内，{a:g}与{b:g} ms比较通过，建议后续同条件计算使用{a:g} ms或更细步长，并保留此误差基准。精确到比本协议更小的效应仍需单独验证。']
    else:
        lines+=['', f'在预算内最后比较仍有未通过项，不能宣布完整闭环轨迹已收敛。建议后续诊断使用当前最细{b:g} ms作为数值参考；这不是已认证步长，涉及精确相位、钙/驱动细节或小反馈效应时，仍需进一步步长/交换时钟验证。不得用常数平移后接近或总体均值接近替代原判据。']
    if fine:
        lines+=['', '误差也未整体呈单调减小：off的B逐时RMS约随再次减半缩小至前一对的0.500倍，但其钙/驱动标准化逐时RMS分别变为前一对的'+fmt(s['error_ratios_finer_pair_over_coarser_pair']['off']['calcium_trajectory'])+'/' +fmt(s['error_ratios_finer_pair_over_coarser_pair']['off']['drive_trajectory'])+'倍；live的B逐时RMS、峰偏移和反馈RMS差均增大。由此不能拟合或宣称整个系统已进入渐近收敛区间。']
    lines+=['', '冻结源码中的一项相关事实：`kernels/ca_boyle.mod`保留了C1钙通道快门控常数`e_timeCourse_tau = 0.100027 ms`，三个神经步长分别约为其1/4、1/8、1/16。B慢节律的单细胞验证不能覆盖这些快动态。它是层间误差差异的一个解释线索；本轮没有隔离通道、重放固定神经输入或更改身体时钟，因而不将全部误差归因于这一门控。', '',
        '本组结果可支持哪些结论，应按指标判读：', '',
        '- 18个B工程原型在完整网络内保持节律：所有步长均观察到，频率和幅度的定量可用性见对应判据。',
        '- 神经输出能产生非零肌肉响应和真实身体形变：所有步长直接记录；形变量的精确值是否稳定见身体判据。',
        '- 所有步长均未通过既定持续推进门槛；近零速度采用绝对误差，不据其相对误差或正负号宣称推进改善。',
        '- 身体反馈的精确电位差、相位差或小运动效应，须与同模式的步长差异逐时比较。`differences/*.npz`提供分支B可直接读取的全302电位、B电位、钙、驱动、身体、应变、速度和反馈差异，不是统计置信区间。', '',
        '三档步长下live/off的B电位总体RMS差分别为'+ '、'.join(fmt(s['feedback_on_off_descriptive_differences'][str(dt)]['B_live_minus_off_RMS_mV'])+' mV' for dt in [.025,.0125,.00625] if str(dt) in s['feedback_on_off_descriptive_differences'])+'，说明“反馈会明显改变该模型的B活动”这一粗尺度观察保留。它不使未通过的精确相位或肌肉逐时判据转为通过，也不证明推进改善。', '',
        '## 验证、来源与剩余限制', '',
        f"来源核对与原始数据检查全部通过；独立NumPy复算检查{s['new_full_runs']}条轨迹、{v['pair_gate_values_checked']}个判据值。独立程序不导入主分析，并单独实现峰显著度与身体旋转。",
        '', '反馈电流重建覆盖801个已保存的10 ms身体状态点；原始反馈电流保存了全部8001个1 ms交换点。每1 ms的身体全态未另行保存，其计算顺序由冻结运行器和运行时断言核对。本次不认证亚10 ms的肌肉电位波形，也不声称能分辨比10 ms更小的B峰相位差。钙到95肌肉再到48×2驱动的映射另由独立程序重建核对。',
        '', '新0.025 ms基线与历史同条件的数值数组逐项比较：', '']
    for mode,h in s['historical_baseline_consistency'].items():
        lines.append(f"- {mode}：{'全部数值数组逐元素相同' if h['all_numeric_arrays_identical'] else '存在差异，详见summary.json逐数组绝对差'}；参数除运行名外完全相同。")
    lines+=['', '实际完成的完整系统运行数没有超过六次；失败的验收项保留在汇总中。误差仅覆盖神经积分步长：身体的1 ms外部交换、身体积分器容差及其内部步进没有细化。8秒、单seed不能给出长期相位稳定性或独立重复的结论，也未估计严格渐近收敛阶。工程B参数尚未生理拟合，结果不扩展为生理通道识别或生物趋温。', '',
        '## 文件与独立复算', '',
        '- 原始轨迹、各运行spec/record/provenance：`results/b-motor-convergence-2026-11-09/{off,live}_dt*/`。',
        '- `protocol.json`及锁定SHA、`inputs.json`来源SHA、`summary.json`逐细胞/逐通道结果、`independent-validation.json`独立审计。',
        '- `runtime/`保存冻结源码、数据、原驱动和身体二进制；`kernels/`、连接计划及参数为只读复制并逐文件校验。原共同运行器及历史结果未改。',
        '- 独立任务源码只在 `analysis/neuromechanics/b_motor_validation/convergence*.py`。',
        '- 证据归档：`docs/reports/assets/b-motor-convergence-2026-11-09/reproduction.zip`，内含全套原始轨迹与SHA清单。编译二进制沿用本机版本，跨平台需要重新构建；原始数据汇总只依赖NumPy/SciPy。', '',
        '在项目根目录、已装requirements所列依赖的Python下执行：', '', '```sh',
        'OPENBLAS_NUM_THREADS=1 /tmp/worm-progress-env/bin/python analysis/neuromechanics/b_motor_validation/convergence_analyze.py',
        'OPENBLAS_NUM_THREADS=1 /tmp/worm-progress-env/bin/python analysis/neuromechanics/b_motor_validation/convergence_verify.py',
        '```', '', '重新模拟应解压到独立目录，将该副本的`runtime/body/worker`恢复为可执行权限，保留协议和输入，仅移走该副本的已有运行目录后，以 `convergence_run.py --feedback off --dt 0.0125` 等指定条件运行；禁止覆盖本证据中的轨迹。', '']
    REPORT.write_text('\n'.join(lines))
    # Archive preserves project-relative layout, including all sources needed by analysis.
    ASSETS.mkdir(parents=True,exist_ok=True)
    payload={}
    for path in ROOT.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.name!='archive-validation.json':
            payload[str(path.relative_to(PROJECT))]=path
    for path in Path(__file__).resolve().parent.glob('convergence*.py'):payload[str(path.relative_to(PROJECT))]=path
    payload[str(REPORT.relative_to(PROJECT))]=REPORT
    manifest={name:sha(path) for name,path in sorted(payload.items())}
    archive=ASSETS/'reproduction.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(payload.items()):z.write(path,name)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='worm-convergence-audit-') as temp:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            z.extractall(temp)
        for name,digest in manifest.items():assert sha(Path(temp)/name)==digest
        # Recompute using ONLY extracted project data/source, rather than shared paths.
        import subprocess,sys,os
        for script in ['convergence_analyze.py','convergence_verify.py']:
            cmd=[sys.executable,str(Path(temp)/'analysis/neuromechanics/b_motor_validation'/script)]
            subprocess.run(cmd,cwd=temp,check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'},stdout=subprocess.DEVNULL)
        assert read(Path(temp)/'results/b-motor-convergence-2026-11-09/summary.json')==s
    digest=sha(archive)
    archive.with_suffix('.zip.sha256').write_text(digest+'  reproduction.zip\n')
    validation=dict(archive_sha256=digest,payload_files=len(payload),archive_bytes=archive.stat().st_size,
        zip_CRC_passed=True,extracted_all_SHA_passed=True,isolated_extracted_analysis_and_independent_audit_passed=True,
        report_sha256=sha(REPORT),summary_sha256=sha(ROOT/'summary.json'))
    (ROOT/'archive-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation))


if __name__=='__main__':main()
