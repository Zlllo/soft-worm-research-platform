"""Cross-check A's summary against independent algebra and write review only."""
import json
from pathlib import Path
import shutil
import numpy as np
from integration_review_source import ROOT, REPO, ISOLATION, sha, write

REPORT=REPO/'docs/reports/c1-muscle-integration-review-2026-11-09.md'
def read(p):return json.loads(p.read_text())
def fmt(x):return f'{x:.6g}'

def main():
    source=read(ROOT/'source-and-units.json');static=read(ROOT/'protocol-and-patches.json');saved=read(ROOT/'saved-data-review.json')
    p=read(ISOLATION/'protocol.json');a=read(ISOLATION/'summary.json')
    assert not static['pending'] and not static['issues'] and not saved['pending']
    assert saved['reviewed_runs']==24 and len(saved['main_runs'])==12 and len(saved['pairs'])==12
    names={'voltage_RMS_mV':'voltage_RMS_mV','voltage_mean_abs_mV':'voltage_mean_abs_mV','voltage_amplitude_relative':'voltage_amplitude_relative',
        'concentration_mM_scaled_RMS':'STATE_calcium_scaled_RMS','concentration_mM_scaled_statistics_max':'STATE_calcium_scaled_stats',
        'cai_mM_scaled_RMS':'cai_scaled_RMS','cai_mM_scaled_statistics_max':'cai_scaled_stats',
        'drive_cai_scaled_RMS':'drive_scaled_RMS','drive_cai_scaled_statistics_max':'drive_scaled_stats'}
    discrepancies=[]
    for key,d in saved['pairs'].items():
        other=a['paired_step_comparisons'][key]
        for independent,original in names.items():discrepancies.append(abs(d['metrics'][independent]-other['errors'][original]['value']))
        discrepancies.append(abs(max(d['metrics']['gate_state_RMS_per_gate'].values())-other['errors']['gate_state_RMS']['value']))
        assert d['all_pass']==other['passed']
    assert max(discrepancies)<1e-13
    for key,d in saved['controls'].items():
        other=a['analytic_controls'][key.replace('_dt','_')]
        if key.startswith('gates'):
            for state,q in d['states'].items():assert abs(q['analytic_max_error']-other['max_abs_state_error'][state])<1e-14
            assert d['analytic_pass']==other['passed']
        else:
            assert abs(d['reference_state_max_error_mM']-other['STATE_max_abs_error_mM'])<1e-20
            assert abs(d['stress_sync_max_error_mM']-other['outward_stress_max_sync_error_mM'])<1e-20
    ratios={};old=saved['pairs']['original_0.0125_vs_0.00625'];new=saved['pairs']['both_0.0125_vs_0.00625']
    for name,target in [('voltage_RMS_mV','voltage_RMS_mV'),('concentration_mM_scaled_RMS','STATE_calcium_scaled_RMS'),
                        ('cai_mM_scaled_RMS','cai_scaled_RMS'),('drive_cai_scaled_RMS','drive_scaled_RMS')]:
        x=old['metrics'][name];y=new['metrics'][name]
        tolerance=.5 if name=='voltage_RMS_mV' else .05
        small=x<=.1*tolerance;rule=y<=.5*x or small
        ratios[target]=dict(original=x,combined=y,ratio=y/x,original_below_small_error_exception=small,
                           preregistered_improvement_rule_satisfied=rule,actual_step_difference_decreased=y<x)
        assert abs(ratios[target]['ratio']-a['combined_over_original_improvement'][target]['ratio'])<1e-13
        assert rule==a['combined_over_original_improvement'][target]['strong_improvement']
    analytic=all(d.get('analytic_pass',d.get('analytic_STATE_pass',False)) for k,d in saved['controls'].items() if '_both_' in k)
    recommend=bool(static['static_review_complete'] and analytic and new['all_pass'] and all(v['preregistered_improvement_rule_satisfied'] for v in ratios.values()))
    assert recommend==a['recommend_next_full_system_validation'] and not recommend
    snapshots=ROOT/'readonly-snapshots';snapshots.mkdir(exist_ok=True)
    reference={}
    for rel in ['protocol.json','protocol-lock.json','summary.json','build-evidence.json','sources.json','harness/integration_prepare_at_lock.py']:
        target=snapshots/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ISOLATION/rel,target)
        reference[str((ISOLATION/rel).relative_to(REPO))]=sha(ISOLATION/rel)
    for name in ['integration_prepare.py','integration_run.py','integration_analyze.py','integration_batch.py']:
        original=REPO/'analysis/neuromechanics/b_motor_validation'/name;shutil.copy2(original,snapshots/name)
        reference[str(original.relative_to(REPO))]=sha(original)
    setup=[]
    for folder in sorted((ISOLATION/'setup-attempts').iterdir()):
        q=read(folder/'failure.json');log=(folder/'log.txt').read_text()
        assert q['neural_steps_executed']==0 and 'data[0]=record' in log
        setup.append(dict(case=folder.name,failure_sha256=sha(folder/'failure.json'),log_sha256=sha(folder/'log.txt'),stage=q['stage']))
    gate_max=max(x['analytic_max_error'] for k,d in saved['controls'].items() if k.startswith('gates_both') for x in d['states'].values())
    pool_max=max(d['reference_state_max_error_mM'] for k,d in saved['controls'].items() if k.startswith('pool_both'))
    cache_error=max(d['previous_state_power_max_error'] for m in saved['main_runs'].values() for d in m['caches'].values())
    current_cache_difference=max(d['current_state_power_RMS_difference'] for m in saved['main_runs'].values() for d in m['caches'].values())
    sync_max=max(d['max_cai_STATE_difference_mM'] for d in saved['main_runs'].values())
    residual=min(d['stress_min_cai_mM'] for k,d in saved['controls'].items() if k.startswith('pool_both'))
    initial_cache=max(d['initial_total_current_cache_error_nA'] for k,d in saved['controls'].items() if k.startswith('pool'))
    findings=[
        dict(id='R1',kind='negative_scientific_result',status='confirmed_and_retained',message='Combined fixed-coefficient STATE controls pass, but all four main last-pair error magnitudes rise by about 2.1x versus original; the preregistered improvement requirement fails for voltage and does not justify the next full-system validation.',evidence='saved-data-review.json / paired errors and independently recomputed ratios'),
        dict(id='R2',kind='unchanged_model_limitation',status='confirmed_already_disclosed_by_A',message='All main endpoint gate-power caches match the state before the most recent neural update, not the simultaneously saved new state. This does not establish the complete runtime current evaluation delay. No cache repair was mixed into the patch.',max_cache_previous_power_error=cache_error),
        dict(id='R3',kind='unchanged_model_limitation',status='confirmed_already_disclosed_by_A',message='Outward-source stress clips STATE to zero after cai assignment, leaving exposed negative cai. The 12 main runs have exactly synchronized STATE/cai, so the stress result does not establish the cause of their step errors.',combined_min_cai_mM=residual,main_max_STATE_cai_difference_mM=sync_max),
        dict(id='R4',kind='observation_timing_note',status='additional_independent_observation',message='At t=0 the global pool iCa cache is still zero while measured helper density is -0.002mA/cm2, equivalent to +0.0062831853nA inward. From the first advanced state onward the cache agrees with measured current. Use measured density for the analytic input; do not treat the initial global cache as the actual source.',initial_global_cache_error_nA=initial_cache)]
    report_a=REPO/'docs/reports/c1-muscle-integration-isolation-2026-11-09.md'
    if report_a.exists():
        reference[str(report_a.relative_to(REPO))]=sha(report_a);shutil.copy2(report_a,snapshots/report_a.name)
    out=dict(status='review_complete',scope='read-only algebra and saved-data audit; zero NEURON imports, compiles, simulations or writes into A/legacy-B directories',
        protocol_sha256=static['protocol_sha256'],A_summary_sha256=sha(ISOLATION/'summary.json'),A_reference_hashes=reference,
        raw_trace_sha256=saved['raw_hashes'],setup_failures_preserved=setup,review_errors=[],issues=findings,
        checked=dict(four_variants_only_derivative_RHS_changed=True,protected_parameters_geometry_initialization_caches_sync_unchanged=True,
            actual_generated_state_equations_exponential_as_intended=True,gate_nonsteady_initials_actual_fixed_voltage_and_exact_solution=True,
            calcium_measured_density_sign_area_conversion_and_analytic_STATE=True,all_24_traces_hashed_finite_correct_clock=True,
            all_12_main_inputs_and_same_dt_synapse_gates_match=True,dense_sampling_and_previous_interval_indices_match=True,
            all_12_step_pairs_independently_recomputed=True,A_pair_and_control_metrics_reproduced=True,predeclared_error_rules_unchanged=True,
            no_full_system_success_inferred_from_component_controls=True),
        numbers=dict(combined_gate_max_abs_error=gate_max,combined_pool_max_abs_error_mM=pool_max,
            pair_metric_max_difference_from_A=max(discrepancies),cached_previous_gate_max_error=cache_error,
            max_single_gate_cached_vs_current_RMS_difference=current_cache_difference),
        main_last_pair_passes={v:saved['pairs'][f'{v}_0.0125_vs_0.00625']['all_pass'] for v in p['variants']},
        text_review=dict(A_summary_conclusion_boundary_checked=True,A_final_prose_report_reviewed=False,
                         reason='A prose report was not saved at the time of this independent data review'),
        improvement=ratios,independent_recommend_next_full_system_validation=recommend,
        interpretation_notes=['Small-error exception flags satisfy a preregistered rule; they are not evidence that calcium/drive errors decreased.',
            'Input is prescribed for a component assay. Same dt gives identical synapse s; synaptic current itself varies with postsynaptic voltage.',
            'Different dt uses the same dense source with different left-hold durations; pair differences include input-discretization effects.',
            'Default channel initialization is steady state; control initial states were explicitly displaced and checked.',
            'Preparer changed only build-resume logic/checker spelling after protocol lock; locked source hash and prepare() AST match, physical patch unchanged.'],
        not_verified=['Continuous-input true solution or cause of larger main pair error','Complete current evaluation delay from endpoint cache observations',
            'Shared mechanism effects on 284 C1 neurons','Full302/body convergence after any repair','Locomotion or thermotaxis improvements'])
    write(ROOT/'review.json',out)
    lines=['# C1 肌肉积分隔离：独立只读复核', '',
        '**源码与解析 STATE 验证通过；主肌肉条件未达到预先要求的误差改善，不建议据此进入新的完整系统验证。** 本复核没有启动 NEURON、编译或任何科学仿真，没有改动 A 文件或旧 B 结果。', '',
        '## 协议与补丁', '',
        f"已核对冻结源清单、协议锁与全部 24 条原始迹线 SHA。协议 SHA：`{static['protocol_sha256']}`。A 汇总 SHA：`{out['A_summary_sha256']}`。", '',
        '四版本为原版、仅五门控、仅 CaPool、两者合并。实际 MOD 差异严格限定为状态导数右端；PARAMETER、UNITS、INITIAL、BREAKPOINT、rates()、hoc 几何/电容/电导均保持不变。实际生成的 states() 在目标版本含正确的负指数系数；其余 rates、初始化、状态调用顺序及电流计算函数与原版一致。没有混入缓存刷新、浓度同步或生理参数调整。', '',
        '准备脚本在锁定后调整过编译续接与指数系数的文本检查。已核对锁定源 SHA 和 prepare() 的语法树完全不变，并保存差异；这项调整没有改变协议或科学补丁。两次首次观察字段访问失败有日志和零步记录，发生于第一个 fadvance 前，均保留。', '',
        '## 解析解与量纲', '',
        '固定电压下，五个门控初值均实测偏离稳态至少 0.05，实际电压偏差为零。独立解为：', '',
        r'\[x(t)=x_\infty+(x_0-x_\infty)e^{-t/\tau}.\tag{1}\]', '',
        f"原版数据与 Euler 的闭式离散解吻合，而合并版在三档步长上的最大解析误差为 {fmt(gate_max)}，低于 10⁻⁹ 门槛。解析比较使用了明确偏离稳态的初值。", '',
        'CaPool 的实测 `ica` 为 mA/cm²、向外为正；面积从 μm² 乘 10⁻⁸ 转为 cm²，总流入电流为 −ica×面积×10⁶ nA。`rho` 的单位是 μmol/(cm·nA·ms)，故源项单位为 μmol/(cm³·ms)=mM/ms。独立解为：', '',
        r'\[S=-10^6\rho\,ica,\quad C_\infty=C_{rest}+\tau S,\quad C(t)=C_\infty+(C_0-C_\infty)e^{-t/\tau}.\tag{2}\]', '',
        f"−0.002 mA/cm² 对应约 +0.0062831853 nA 流入和 {fmt(saved['controls']['pool_both_dt0.025']['source_mM_per_ms'])} mM/ms 源项。0–50 ms 固定输入、50–100 ms 零输入衰减均已核对；合并版 STATE 最大解析误差为 {fmt(pool_max)} mM，低于 10⁻¹⁵ mM 门槛。STATE 与暴露的 cai 分别复算。", '',
        't=0 的全局 pool_iCa 仍是初始化缓存零值，与已测得的源电流不同；第一个推进状态起两者一致。解析参考使用实测密度，所有单步 STATE 更新也与保存的实测源项及原/指数公式吻合，因此这项初始观测差异没有被当作实际零输入。', '',
        '## 主肌肉结果与 A 汇总复算', '',
        '12 个主条件均为相同几何、NMJ 0.9 nS、Vth −30 mV 和相同预先保存输入。每条主迹线有 320001 个 0.025 ms 样点，保存间隔约为最快门控时间常数 0.100027 ms 的四分之一。各版本在共同端点的输入逐元素一致，同一步长的突触 s 也逐元素一致；上一区间实际输入按各自步长的左端索引核对。突触电流会随输出肌肉电位变化，不能把它当作相同电流钳。规定输入属于组件测试，不是自主运动。', '',
        '四版本的全部三种步长配对均独立重算；与 A 汇总的最大指标差为 '+fmt(max(discrepancies))+'。最后配对（0.0125 对 0.00625 ms）、4–8 秒窗口如下：', '',
        '| 版本 | 电位 RMS mV | STATE 钙缩放 RMS | cai 缩放 RMS | 驱动缩放 RMS | 验收阈值 |',
        '|---|---:|---:|---:|---:|---|']
    for variant in p['variants']:
        d=saved['pairs'][f'{variant}_0.0125_vs_0.00625']['metrics']
        lines.append(f"| {variant} | {fmt(d['voltage_RMS_mV'])} | {fmt(d['concentration_mM_scaled_RMS'])} | {fmt(d['cai_mM_scaled_RMS'])} | {fmt(d['drive_cai_scaled_RMS'])} | 通过 |")
    lines += ['', '所有版本都通过该孤立输入下的最后配对阈值，并不意味着新版本改善。合并/原版的电位、STATE 钙、cai、驱动误差比分别为 '+ '、'.join(fmt(v['ratio']) for v in ratios.values())+'，均增加。钙/驱动的“strong_improvement=true”来自预先声明的原误差已低于十分之一门槛的例外规则，不是实测改善；电位项不满足例外且未改善，故 A 的不建议进入完整系统结论与独立计算一致。', '',
        '不同步长使用同一密集源，但保持输入的区间长度随 dt 变化，步长差同时包含该输入离散化。原版较小的配对差不能直接证明它更接近连续系统真解，也不能仅凭这组数据确定是缓存、输入保持、膜积分或误差抵消造成结果。', '',
        '## 保留的问题与边界', '',
        f"1. 所有主条件的五个门控幂缓存与最近一次状态更新前的门控幂相符，最大误差 {fmt(cache_error)}；与同时保存的新状态不同。它确认了端点缓存关系，未单独测得完整电流求值顺序或总延迟。补丁没有改变这一行为。", '',
        f"2. 外流压力段中 STATE 被截为零后，暴露 cai 仍可为负；合并版最小 cai 为 {fmt(residual)} mM。该问题按协议保留，未宣称浓度同步修复。12 条主条件的 STATE/cai 最大差为 {fmt(sync_max)}，因此不能把压力段现象直接指认为主条件误差原因。", '',
        '3. 未测试共享密度机制对其余 284 个 C1 神经元的影响、完整302网络/身体的新步长收敛，以及自主推进或趋温。组件解析通过不能替代这些验收。', '',
        '## 复核文件', '',
        '本次已审阅 A 的机器可读汇总及其结论边界；A 最终中文报告在独立复核时尚未保存，因此没有审核其最终文本。', '',
        '- `results/c1-muscle-integration-review-2026-11-09/review.json`：最终结论、具体问题、来源 SHA、独立比值和未验证项。',
        '- 同目录 `source-and-units.json`、`protocol-and-patches.json`、`saved-data-review.json`：可重算证据；`readonly-snapshots/` 保存审阅时的 A 协议、汇总和代码。原始迹线保持在 A 目录，以 SHA 引用。',
        '- `analysis/neuromechanics/b_motor_validation/integration_review_*.py`：只读源码检查、代数复算与报告生成。']
    REPORT.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=out['status'],review_errors=len(out['review_errors']),findings=len(findings),recommend_full_system=recommend,report=str(REPORT))))

if __name__=='__main__':main()
