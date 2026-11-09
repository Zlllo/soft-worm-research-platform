"""Audit and report the completed, fixed-budget body topology comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from body_topology_calibration import check, PARENT, REPO
from diagnose_navigation_failures import digest, write
from thermotaxis.navigation_window_policy import WindowActorCritic


def balanced_by_run(y, score, run):
    """Recalculate from saved predictions independently of the fitting helper."""
    values = []
    for rid in np.unique(run):
        mask = run == rid
        positive, negative = mask & (y == 1), mask & (y == 0)
        assert positive.any() and negative.any()
        values.append(((score[positive] >= 0).mean() + (score[negative] < 0).mean()) / 2)
    return float(np.mean(values))


def main(root, output):
    d = json.loads((root / 'design.json').read_text())
    check(d)
    evaluation = json.loads((root / 'evaluation.json').read_text())
    assert evaluation['complete'] and len(evaluation['rows']) == 84
    meta = json.loads((root / 'new-body-scenes.json').read_text())
    assert digest(root / 'new-body-scenes.npz') == meta['sha256']
    with np.load(root / 'new-body-scenes.npz') as z:
        data = {k: z[k] for k in z.files}
    assert len(data['label']) == meta['windows'] and len(np.unique(data['run'])) == 16
    for case in d['new_body_cases']:
        folder = root / case['path']
        run = json.loads((folder / 'run.json').read_text())
        build = json.loads((folder / 'build.json').read_text())
        assert run['complete'] and run['max_solver_ahead_s'] <= 1e-10
        assert run['factor'] == case['factor'] and run['onset_s'] == case['onset_s']
        assert run['feedback'] == case['feedback']
        assert digest(folder / 'trajectory.npz') == run['trajectory_sha256'] == meta['physical_run_sha256'][case['path']]
        assert digest(folder / 'worm-load-worker.cc') == build['generated_sha256']
        assert digest(folder / 'worker') == build['binary_sha256']
    # The physical runner reuses this locked parent design and its native libraries.
    body = json.loads((REPO / 'results/body-load-stage2-2026-11-03/design.json').read_text())
    for path, sha in {**body['source_sha256'], **body['inputs_sha256']}.items():
        assert digest(REPO / path) == sha, path

    seen = set()
    for row in evaluation['rows']:
        s, init = row['spec'], row['initialization']
        key = (init, s['id'])
        assert key not in seen
        seen.add(key)
        folder = root / 'fits' / f'{init}-{s["id"]:03d}'
        record = json.loads((folder / 'record.json').read_text())
        assert record['complete'] and record['spec'] == s and record['initialization'] == init
        assert row['old'] == record['old']
        assert digest(folder / 'model.pt') == record['model_sha256']
        torch.manual_seed(s['seed'])
        model = WindowActorCritic(s['name'], s['interface'])
        if init == 'BC':
            model.load_state_dict(torch.load(PARENT / f'fit-{s["id"]:03d}' / 'BC.pt', map_location='cpu', weights_only=True))
        start = model.state_dict()
        end = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
        changed = [k for k in start if not torch.equal(start[k], end[k])]
        assert changed == record['changed_names']
        assert set(changed) <= set(record['trainable_names'])
        assert not any(k.startswith('critic.') or k == 'log_std' for k in changed)
        with np.load(folder / 'new-predictions.npz') as z:
            score = z['score']
        assert score.shape == data['label'].shape and np.isfinite(score).all()
        for family, mask in [('all', np.ones(len(score), bool)), ('noisy', data['noise'] > 0),
                             ('matched_load', data['family'] == 0), ('novel_load', data['family'] == 1)]:
            value = balanced_by_run(data['label'][mask], score[mask], data['run'][mask])
            assert abs(value - row['new'][family]['balanced_accuracy']) < 1e-12
    assert seen == {(init, s['id']) for init in d['initializations'] for s in d['specs']}

    names = list(dict.fromkeys(s['name'] for s in d['specs']))
    masks = {}
    for name in names:
        if name.startswith('gru') or name == 'isolated8':
            continue
        model = WindowActorCritic(name, 'restricted')
        key = tuple(zip(model.pre.tolist(), model.post.tolist()))
        masks.setdefault(key, []).append(name)
    duplicates = [v for v in masks.values() if len(v) > 1]
    groups = []
    contrasts = []
    for init in d['initializations']:
        rows = [r for r in evaluation['rows'] if r['initialization'] == init]
        for name in names:
            selected = sorted((r for r in rows if r['spec']['name'] == name), key=lambda r: r['spec']['seed'])
            groups.append(dict(initialization=init, name=name, seeds=[r['spec']['seed'] for r in selected],
                scores={f: [r['new'][f]['balanced_accuracy'] for r in selected]
                        for f in ('all', 'noisy', 'matched_load', 'novel_load')},
                old=[r['old']['all']['balanced_accuracy'] for r in selected]))
        for seed in (7, 17, 27):
            selected = [r for r in rows if r['spec']['seed'] == seed]
            real = next(r for r in selected if r['spec']['name'] == 'real8')
            rewired = [r for r in selected if r['spec']['name'].startswith('rewired')]
            assert len(rewired) == 10
            contrasts.append(dict(initialization=init, seed=seed,
                real_minus_mean_rewired={f: real['new'][f]['balanced_accuracy'] - float(np.mean([r['new'][f]['balanced_accuracy'] for r in rewired]))
                                        for f in ('matched_load', 'novel_load')}))
    regression = (root / 'regression.log').read_text()
    assert '210 passed, 14 warnings' in regression
    restored = json.loads((root / 'restore-check.json').read_text())
    assert restored['archive_restored_evaluation_matches'] and restored['models'] == 84 and not restored['new_training']
    summary = dict(complete=True, status='stage_paused_after_evaluation', fits=84, physical_runs=16,
        windows=meta['windows'], family_windows={f: int((data['family'] == code).sum())
            for f, code in [('matched_load', 0), ('novel_load', 1)]}, groups=groups,
        paired_seed_contrasts=contrasts, duplicate_labelled_chemical_masks=duplicates,
        verification=dict(locked_sources_and_inputs=True, model_hashes=84, actual_parameter_freezes=84,
            fresh_prediction_metrics_recomputed=84, native_trajectory_source_and_binary_hashes=16,
            project_tests_passed=210, warnings=14, restored_archive_evaluation_matches=True),
        limits=d['notes'])
    output.mkdir(parents=True, exist_ok=True)
    write(root / 'summary.json', summary)
    write(output / 'summary.json', summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True)
    for ax, init in zip(axes, d['initializations']):
        for i, name in enumerate(names):
            group = next(g for g in groups if g['initialization'] == init and g['name'] == name)
            for family, offset, color, marker in [('matched_load', -.12, '#1764ab', 'o'), ('novel_load', .12, '#cf6c16', '^')]:
                value = np.array(group['scores'][family]) * 100
                ax.scatter(value, np.full(3, i + offset), color=color, s=24, marker=marker, alpha=.8,
                           label=family.replace('_', ' ') if i == 0 else None)
                ax.plot([value.mean()], [i + offset], marker='|', color='black', markersize=12)
        ax.set_title(init + ' initialization')
        ax.set_xlim(47, 101)
        ax.axvline(50, color='gray', linewidth=.8, linestyle='--')
        ax.set_xlabel('Macro-run balanced accuracy (%)')
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    axes[1].legend(loc='lower left')
    fig.suptitle('Fixed budget: 1,024 updates; all three seeds retained\nNew pulse phases / fields / noise; one deterministic body parameter set')
    fig.tight_layout()
    fig.savefig(output / 'topology-comparison.png', dpi=160)
    plt.close(fig)

    def avg(init, name, family):
        g = next(g for g in groups if g['initialization'] == init and g['name'] == name)
        return 100 * np.mean(g['scores'][family])

    table = ['| 布局 | BC：原负载 | BC：新负载 | 随机：原负载 | 随机：新负载 |',
             '|---|---:|---:|---:|---:|']
    for name in names:
        table.append('| ' + name + ' | ' + ' | '.join(f'{avg(init, name, family):.2f}%' for init in d['initializations'] for family in ('matched_load', 'novel_load')) + ' |')
    table.append('| 全部10张重排图均值 | ' + ' | '.join(f'{np.mean([avg(init, n, f) for n in names if n.startswith("rewired")]):.2f}%' for init in d['initializations'] for f in ('matched_load', 'novel_load')) + ' |')
    seeds = ['| 起点 | 布局 | 原负载：种子7／17／27 | 新负载：种子7／17／27 |', '|---|---|---|---|']
    for g in groups:
        seeds.append('| ' + g['initialization'] + ' | ' + g['name'] + ' | ' + ' | '.join('／'.join(f'{100*x:.2f}%' for x in g['scores'][f]) for f in ('matched_load', 'novel_load')) + ' |')
    contrast_lines = []
    for init in d['initializations']:
        c = [r for r in contrasts if r['initialization'] == init]
        contrast_lines.append(f'- {init}：原负载真实图减重排均值分别为' + '、'.join(f'{100*r["real_minus_mean_rewired"]["matched_load"]:+.2f}' for r in c) + '个百分点；三种子平均' + f'{100*np.mean([r["real_minus_mean_rewired"]["matched_load"] for r in c]):+.2f}' + '个百分点。')
    duplicate_text = ('真实图与10张重排图在命名节点下的化学连接掩码均不同。' if not duplicates else
                      '固定命名节点下化学连接掩码完全重复的分组为：' + json.dumps(duplicates, ensure_ascii=False) + '。')
    report = f'''# 身体方向解码：等预算布局对照与新身体病例

登记日期：2026-11-03；完成核验：2026-11-05。状态：**84次拟合、16次新身体模拟和最终评估已完成，按用户要求阶段性暂停**。

真实8节点结构在新相位／温场上的方向解码可达到较高准确率，但没有稳定超过全部10张重排图。真实图BC起点原负载均值为{avg('BC','real8','matched_load'):.2f}%，随机起点为{avg('random','real8','matched_load'):.2f}%。低分可以由功能参数状态及任务接口产生，不能只用“只有8个节点”解释；本轮结果也没有给出真实布局独有的任务优势。

## 设计和评价边界

保留真实8节点、全部10张化学重排图、隔离节点、GRU2和GRU16，各3个种子。BC沿用各自同预算的旧运动学行为克隆模型；random从随机权重开始，两类起点分别报告。每个模型以相同种子批次进行1024次更新，batch256、Adam 0.003，全部actor功能参数可训练；节点、图掩码、8节点模型的0.3秒时间常数、critic和探索参数固定。GRU更新记忆和读出，是容量参考，不是参数量匹配的细胞模型。

训练仅使用原身体数据相位0、2、4、6。旧相位3、7结果保留为诊断。最终检查点直接用于本次新病例，没有根据新病例选择种子或检查点。

新身体模拟为相位3.5／7.5 × 前部阻力倍数0.75／1.5／2／3 × 实时反馈／名义反馈回放，共16次20秒模拟。预先将1.5／2定义为主要的原负载范围组，0.75／3定义为次要的新强度组，各8次。温场方向11.25°+45°k、噪声种子0／149／159／169也是本次新取值。得到{meta['windows']:,}个四秒历史窗口，同一身体运行的温场和噪声版本不充当独立动物。

所有病例共用一套确定性的身体参数与起始状态；“新身体病例”指新扰动相位、强度和采样条件。准确率是每条身体运行内先计算平衡准确率，再对运行及3个训练种子平均。标签是身体哪侧更暖；没有训练身体趋温策略。受限模型在AFD接收温度、AIY接收实际曲率及其因果差分、从RIA读出，属于工程输入投影。

## 全部布局结果

{chr(10).join(table)}

![全部种子与布局](assets/body-topology-calibration-2026-11-03/topology-comparison.png)

{chr(10).join(contrast_lines)}

以上为同训练种子的配对描述；没有把窗口数量当成独立重复计算显著性。重排图全部保留，未筛掉低分结果。{duplicate_text}所有非隔离图均保留温度／运动输入到RIA的图可达性；这种可达性不能解释不同权重下的具体分数。隔离模型实际关闭跨节点和gap通信，RIA没有温度／运动输入，始终50%符合该接口的结构预期。

## 对当前解释的影响

1. 解剖结构与功能参数要分开。真实图不加节点即可在本次任务中获得较好解码；前一步内部参数1024次更新90.17%、仅输入和读出4096次63.21%的结果，结合本轮多布局结果，更支持先检查输入配合与内部状态。两次实验起点和新病例不同，百分比不能直接作同一对照。
2. 当前任务没有稳定区分真实图与保留通路的重排图。多个布局在学习有效权重后均能完成计算；真实图的细胞身份、符号、效能及统一时间常数仍是简化实现，不能把这里的排名解释为活体网络排名。
3. 身体提供了运动与采样的时间结构。此前匹配运动参考显著改善解码、连续延迟使冻结解码下降，是时序对应的证据。本轮显示多种回路可学习利用这种信息；它没有单独分离弹性、阻力或神经反馈各自的因果贡献。原负载实时反馈恢复未获优势的负结果继续保留。

## 全部训练种子

{chr(10).join(seeds)}

## 核验、复现和暂停位置

已逐一核对84个模型哈希，并从各自真实起点重新比较全部参数：只有声明的参数发生改变，critic、探索参数和图缓冲区保持不变。独立读取保存预测，重算全部84个模型的四类新病例指标，均与评估记录一致。16条轨迹、生成原生源码及二进制哈希通过检查，原生求解器未越过控制时钟；登记的源码、输入和父级身体依赖哈希一致。项目tests/中**210项通过，14条已有依赖弃用警告**。

另在独立临时目录解压归档、恢复相对路径，不访问原工作区模型和数据，重新评价84个最终模型，evaluation.json与原结果逐字节一致；未重新训练。

复现归档包含本次模型、预测、轨迹、设计、日志，以及登记的训练数据和BC起点。解压后将evidence目录内容恢复到results/body-topology-calibration-2026-11-03，其余目录保留相对路径，即可重做evaluate。身体模拟重建仍需原Boyle固定提交及Sundials构建环境；原构建记录保留机器路径用于溯源，换目录应重新构建worker。归档不含原生二进制，完整原生依赖与二进制的核验在原工作区进行。

```bash
PYTHONPATH=src python3 analysis/body_topology_calibration.py evaluate
PYTHONPATH=src MPLCONFIGDIR=/tmp/worm-topology-report python3 analysis/summarize_body_topology_calibration.py
```

以上是原工作区的评价和报告命令；summarize额外核验本地原生二进制及父级构建环境，在仅恢复归档的目录中先运行evaluate即可。完整重新拟合需在恢复依赖后使用另一个root，按prepare、physical、train、evaluate顺序执行，避免覆盖本轮记录。

[完整汇总JSON](assets/body-topology-calibration-2026-11-03/summary.json) · [复现ZIP](assets/body-topology-calibration-2026-11-03/reproduction.zip) · [逐文件SHA256](assets/body-topology-calibration-2026-11-03/manifest.json)

本阶段到此暂停。后续AFD→AIY与强度匹配通路干预仅有草稿，未执行、没有实验结果；本次没有增加节点或启动新的训练阶段。
'''
    (REPO / 'docs/reports/body-topology-calibration-2026-11-03.md').write_text(report)
    print(json.dumps(dict(fits=84, runs=16, windows=meta['windows'], duplicates=duplicates, complete=True)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('results/body-topology-calibration-2026-11-03'))
    p.add_argument('--output', type=Path, default=Path('docs/reports/assets/body-topology-calibration-2026-11-03'))
    a = p.parse_args()
    torch.set_num_threads(1)
    main(a.root, a.output)
