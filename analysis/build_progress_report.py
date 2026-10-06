"""Build the teacher-facing narrative and canonical portable-report input.

Run after phase1_progress_scan.py; the HTML reader is packaged separately.
"""
from pathlib import Path
import json
import shutil
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/phase1-progress-2026-10-04'
REPORTS = ROOT / 'docs/reports'
ASSETS = REPORTS / 'assets/phase1-progress-2026-10-04'


def audited_report_queries(summaries):
    """Materially aggregate raw pairs with SQLite for native reader provenance.

    Bootstrap intervals remain Python analysis outputs and are explicitly
    identified as such; SQL does not claim to generate them.
    """
    connection = sqlite3.connect(ASSETS / 'report-evidence.sqlite')
    connection.row_factory = sqlite3.Row
    connection.executescript('DROP TABLE IF EXISTS trials; DROP TABLE IF EXISTS intervals; DROP TABLE IF EXISTS arrival_ci; DROP TABLE IF EXISTS grid; '
                            'CREATE TABLE trials(condition TEXT,side TEXT,noise REAL,tau REAL,replicate INTEGER,grp TEXT,reached INTEGER,entry REAL,restricted REAL,residence REAL,rms REAL); '
                            'CREATE TABLE intervals(side TEXT,metric TEXT,lo REAL,hi REAL,display TEXT); '
                            'CREATE TABLE arrival_ci(side TEXT,grp TEXT,display TEXT); '
                            'CREATE TABLE grid(side TEXT,grp TEXT,t REAL,phase INTEGER);')
    for name, summary in summaries.items():
        side = {'confirmation-cold':'冷侧','confirmation-hot':'热侧'}.get(name, '')
        pairs = json.loads((DATA / name / 'paired-records.json').read_text(encoding='utf8'))['paired_replicates']
        for pair in pairs:
            for group, label in [('response','响应'),('no_response','无响应')]:
                row = pair[group]
                restricted = row['first_entry_time_s'] if row['entered_comfort_region'] else 180.
                connection.execute('INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                                   (name,side,summary['noise_K'],summary['slow_memory_s'],pair['replicate'],label,
                                    int(row['entered_comfort_region']),row['first_entry_time_s'],restricted,
                                    row['comfort_residence_fraction'],row['rms_temperature_error_K']))
        if not side:
            continue
        for group, label in [('response','响应'),('no_response','无响应')]:
            ci = summary['groups'][group]['reached_wilson95']
            connection.execute('INSERT INTO arrival_ci VALUES (?,?,?)',(side,label,f'{ci[0]:.1%}—{ci[1]:.1%}'))
            for t in summary['groups'][group]['survival']['time_s']:
                if t > 0:
                    connection.execute('INSERT INTO grid VALUES (?,?,?,?)',(side,label,t,0))
                connection.execute('INSERT INTO grid VALUES (?,?,?,?)',(side,label,t,1))
        for metric, precision, label in [('restricted_first_entry_s',2,'限制首次进入时间 / s'),('comfort_residence_fraction',4,'驻留比例'),('rms_temperature_error_K',4,'RMS 温差 / K')]:
            d = summary['paired_difference'][metric]
            lo, hi = d['bootstrap95']
            display = f"{d['mean']:+.{precision}f}（{lo:+.{precision}f}—{hi:+.{precision}f}）"
            connection.execute('INSERT INTO intervals VALUES (?,?,?,?,?)',(side,label,lo,hi,display))
    queries = {
        'confirmation': '''SELECT side, grp AS "group", COUNT(*) AS pairs, AVG(reached) AS reached,
arrival_ci.display AS reached_ci, AVG(restricted) AS restricted_s, AVG(residence) AS residence, AVG(rms) AS rms_K
FROM trials JOIN arrival_ci USING(side,grp) WHERE side <> '' GROUP BY side,grp ORDER BY side,grp''',
        'differences': '''WITH paired AS (
SELECT a.side, a.restricted-b.restricted AS restricted, a.residence-b.residence AS residence, a.rms-b.rms AS rms
FROM trials a JOIN trials b ON a.condition=b.condition AND a.replicate=b.replicate
WHERE a.side<>'' AND a.grp='响应' AND b.grp='无响应'),
long_delta AS (
SELECT side,'限制首次进入时间 / s' AS metric,restricted AS delta FROM paired UNION ALL
SELECT side,'驻留比例',residence FROM paired UNION ALL SELECT side,'RMS 温差 / K',rms FROM paired)
SELECT side,metric,AVG(delta) AS difference,lo AS ci_low,hi AS ci_high,display,COUNT(*) AS pairs,4000 AS bootstrap_resamples
FROM long_delta JOIN intervals USING(side,metric) GROUP BY side,metric ORDER BY side,metric''',
        'survival': '''SELECT grid.t AS time_s,
AVG(CASE WHEN trials.reached=1 AND (trials.entry<grid.t OR (grid.phase=1 AND trials.entry=grid.t)) THEN 0.0 ELSE 1.0 END) AS survival,
grid.side || '·' || grid.grp AS series,COUNT(*) AS pairs,180 AS window_s
FROM grid JOIN trials ON grid.side=trials.side AND grid.grp=trials.grp
GROUP BY grid.side,grid.grp,grid.t,grid.phase ORDER BY series,grid.t,grid.phase''',
        'scan': '''SELECT 'σ=' || printf('%g',a.noise) || ' K；τ慢=' || printf('%g',a.tau) || ' s' AS condition,
a.noise AS noise_K,a.tau AS slow_memory_s,AVG(a.residence-b.residence) AS residence_difference,COUNT(*) AS pairs
FROM trials a JOIN trials b ON a.condition=b.condition AND a.replicate=b.replicate
WHERE a.side='' AND a.grp='响应' AND b.grp='无响应'
GROUP BY a.noise,a.tau ORDER BY a.noise,a.tau'''
    }
    datasets, sources = {}, []
    for name, sql in queries.items():
        datasets[name] = [dict(r) for r in connection.execute(sql)]
        (ASSETS / f'{name}.sql').write_text(sql+';\n',encoding='utf8')
        sources.append(dict(id='query-'+name,label='本地配对仿真与统计证据：'+name,
                            path=f'docs/reports/assets/phase1-progress-2026-10-04/{name}.sql',
                            query=dict(sql=sql,engine='SQLite',language='sql',
                                       description='从保存的逐对轨迹指标计算组均值、配对差或首次进入生存曲线；区间来自 phase1_progress_scan.py 的 Wilson/整对 bootstrap 输出。',
                                       tables_used=['trials','arrival_ci','intervals','grid'],
                                       filters=['静态线性温度场；180 s；确认每侧128对，扫描每格32对'],
                                       source_files=['analysis/phase1_progress_scan.py','results/phase1-progress-2026-10-04/*/paired-records.json','docs/reports/assets/phase1-progress-2026-10-04/report-evidence.sqlite'])))
    connection.commit()
    connection.close()
    return datasets,sources


def main():
    summaries = json.loads((DATA / 'summary.json').read_text(encoding='utf8'))
    ASSETS.mkdir(parents=True, exist_ok=True)
    for filename in ['summary.json', 'design.json', 'numerical-gate.json', 'manifest.json',
                     'confirmation-survival.png', 'confirmation-survival.svg',
                     'exploratory-scan.png', 'exploratory-scan.svg']:
        shutil.copyfile(DATA / filename, ASSETS / filename)
    lines = ['本轮完成 9 格×32 对探索扫描和冷、热各 128 对参考实验，共 544 对、1,088 条轨迹。下表到达率括号为 Wilson 95% 区间，其他列为全体轨迹的均值。', '',
             '| 起点与组别 | 到达率（95% 区间） | 限制平均首次进入 / s | 驻留比例 | RMS 温差 / K |',
             '|---|---:|---:|---:|---:|']
    group_rows, difference_rows, survival_rows, scan_rows = [], [], [], []
    for side, side_label in [('cold', '冷侧'), ('hot', '热侧')]:
        summary = summaries['confirmation-' + side]
        for group, label in [('response', '响应'), ('no_response', '无响应')]:
            row = summary['groups'][group]
            m, ci = row['means'], row['reached_wilson95']
            lines.append(f"| {side_label}，{label} | {m['reached']:.1%}（{ci[0]:.1%}—{ci[1]:.1%}） | {m['restricted_first_entry_s']:.2f} | {m['comfort_residence_fraction']:.4f} | {m['rms_temperature_error_K']:.4f} |")
            group_rows.append(dict(side=side_label, group=label, reached=m['reached'],
                                   reached_ci=f'{ci[0]:.1%}—{ci[1]:.1%}', pairs=128,
                                   restricted_s=m['restricted_first_entry_s'],
                                   residence=m['comfort_residence_fraction'], rms_K=m['rms_temperature_error_K']))
            curve = row['survival']
            previous = curve['survival_probability'][0]
            for t, value in zip(curve['time_s'], curve['survival_probability']):
                # Duplicate event x values preserve a step curve in a line renderer.
                if t > 0:
                    survival_rows.append(dict(time_s=t, survival=previous, series=side_label + '·' + label, pairs=128, window_s=180))
                survival_rows.append(dict(time_s=t, survival=value, series=side_label + '·' + label, pairs=128, window_s=180))
                previous = value
    lines += ['', '使用整对重采样计算的配对差（响应减无响应）如下；括号为 4,000 次 bootstrap 的百分位 95% 区间。时间差为负、驻留差为正、RMS 差为负，分别表示寻找更快、驻留更多和温度误差更小。', '',
              '| 起点 | 限制首次进入时间差 / s | 驻留比例差 | RMS 温差差 / K |', '|---|---:|---:|---:|']
    for side, label in [('cold', '冷侧'), ('hot', '热侧')]:
        cells = []
        for metric, precision, metric_label in [('restricted_first_entry_s', 2, '限制首次进入时间 / s'),
                                               ('comfort_residence_fraction', 4, '驻留比例'),
                                               ('rms_temperature_error_K', 4, 'RMS 温差 / K')]:
            d = summaries['confirmation-' + side]['paired_difference'][metric]
            ci = d['bootstrap95']
            cell = f"{d['mean']:+.{precision}f}（{ci[0]:+.{precision}f}—{ci[1]:+.{precision}f}）"
            cells.append(cell)
            difference_rows.append(dict(side=label, metric=metric_label, difference=d['mean'],
                                        ci_low=ci[0], ci_high=ci[1], display=cell, pairs=128, bootstrap_resamples=4000))
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '参考参数的冷热两侧都出现一致方向的改善，上述三项配对区间均未跨零。可以报告“在当前模型和给定参数下，趋势记忆控制提高了有限时窗趋温表现”。这还不解释身体柔软性，也不证明跨环境的普适性。', '',
              '![首次进入舒适区的生存曲线](assets/phase1-progress-2026-10-04/confirmation-survival.png)', '',
              '图 1：纵轴为尚未首次进入舒适带的轨迹比例，下降更快表示进入更早。每侧 128 对，180 s 行政删失；实线为响应、虚线为无响应。Cold/Hot 表示冷/热侧。', '',
              '![噪声与慢记忆的探索扫描](assets/phase1-progress-2026-10-04/exploratory-scan.png)', '',
              '图 2：每格 32 对；左为驻留比例差，右为限制首次进入时间差，均为响应减无响应。本次九格驻留改善范围为 +0.1701—+0.4624，限制首次进入时间缩短约 41.99—96.02 s。高噪声 0.25 K 的改善幅度在本次扫描中较弱。该扫描用于提出后续问题，不进行参数最优性或多重比较显著性声明。', '',
              '数值加速前已比较物理步长 0.01 s 与 0.1 s，保留测温和控制周期。构建检查四组噪声/记忆极端条件，独立验证另查八组冷热/噪声/种子配置；结果在误差阈值内一致。加速降低了分段匀速运动的重复计算次数，没有改变传感或控制信息量。']
    narrative_path = REPORTS / 'progress-2026-10-04.md'
    narrative = narrative_path.read_text(encoding='utf8')
    # Replace this one section; preserve unrelated prose across reruns.
    a = narrative.index('## 本轮结果')
    b = narrative.index('## 结果的适用范围与验收')
    narrative = narrative[:a] + '## 本轮结果\n\n' + '\n'.join(lines) + '\n\n' + narrative[b:]
    narrative_path.write_text(narrative, encoding='utf8')
    for key, s in summaries.items():
        if not key.startswith('scan-'):
            continue
        d = s['paired_difference']['comfort_residence_fraction']
        scan_rows.append(dict(condition=f"σ={s['noise_K']:g} K；τ慢={s['slow_memory_s']:g} s", noise_K=s['noise_K'],
                              slow_memory_s=s['slow_memory_s'], residence_difference=d['mean'],
                              ci_low=d['bootstrap95'][0], ci_high=d['bootstrap95'][1], pairs=s['replicates']))
    title = narrative.splitlines()[0][2:]
    blocks = [dict(id='title', type='markdown', body='# ' + title + '\n\n2026 年 10 月 4 日 · 物理学本科科研阶段汇报')]
    sections = narrative.split('\n## ')[1:]
    for i, section in enumerate(sections):
        heading, _, body = section.partition('\n')
        if heading == '复现材料':
            continue  # Native source affordances carry provenance; md retains full reproducibility.
        if heading == '本轮结果':
            blocks += [dict(id='results', type='markdown', sourceId='statistics', body='## 冷热两侧均改善了参考参数下的有限时窗趋温表现\n\n每侧 128 对，观察 180 s；响应组与无响应组共享配对随机条件。到达率用 Wilson 95% 区间，差值用整对 bootstrap 95% 区间。限制时间纳入所有未到达轨迹；驻留为有限观察窗比例，RMS 为各条轨迹时间 RMS 的均值。'),
                       dict(id='groups', type='table', tableId='group-table'),
                       dict(id='differences', type='table', tableId='difference-table'),
                       dict(id='finding-interpretation', type='markdown', sourceId='statistics', body='上述时间、驻留和 RMS 配对差的区间均未跨零。结果支持当前模型、当前参数下的机制改善，尚不支持身体柔软性或跨环境普适性的结论。'),
                       dict(id='survival-chart', type='chart', chartId='survival'),
                       dict(id='scan-intro', type='markdown', sourceId='statistics', body='## 九格探索扫描提供了噪声与记忆的后续研究区间\n\n每格 32 对；横轴列出噪声与慢记忆，纵轴是响应减对照的驻留比例。九格差值均为正；高噪声条件优势减弱。未做多重比较校正，不能根据最高格点选择最优参数。'),
                       dict(id='scan-chart', type='chart', chartId='scan')]
        else:
            # Static figure images belong to supporting md; portable reader uses native charts.
            blocks.append(dict(id=f'section-{i}', type='markdown', body='## ' + heading + '\n' + body))
    datasets, query_sources = audited_report_queries(summaries)
    sources = [dict(id='statistics', label='固定设计统计实验及逐对指标汇总', path='docs/reports/assets/phase1-progress-2026-10-04/summary.json'),
               dict(id='design', label='运行前固定的实验设计', path='configs/phase1_progress_scan.json')]
    sources += query_sources
    tables = [dict(id='group-table', title='冷热参考实验的组均值与到达区间', dataset='confirmation', sourceId='query-confirmation',
                   defaultSort=dict(field='side', direction='asc'), columns=[dict(field=f, label=l, **opts) for f,l,opts in [
                       ('side','起点',{}),('group','组别',{}),('pairs','配对数',{}),('reached','到达率',dict(format='percent')),('reached_ci','95% 区间',{}),
                       ('restricted_s','限制时间 / s',{}),('residence','驻留比例',{}),('rms_K','RMS / K',{})]]),
              dict(id='difference-table', title='响应减对照的配对差及 95% 区间', dataset='differences', sourceId='query-differences',
                   defaultSort=dict(field='side', direction='asc'),columns=[dict(field=f,label=l) for f,l in [('side','起点'),('metric','指标'),('display','配对均值差（95% 区间）')]])]
    charts = [dict(id='survival', title='冷热起点的首次进入舒适带生存曲线', subtitle='每侧 128 对；观察 180 s；纵轴为尚未进入的轨迹比例，阶梯下降表示首次到达',
                   type='line',dataset='survival',sourceId='query-survival',encodings=dict(x=dict(field='time_s',type='quantitative',label='观察时间 / s'),
                   y=dict(field='survival',type='quantitative',label='尚未到达比例'),color=dict(field='series',type='nominal',label='起点与组别'))),
              dict(id='scan',title='噪声与慢记忆九格扫描的驻留比例差',subtitle='每格 32 对；响应减无响应；探索性格点，无多重比较校正',
                   type='bar',dataset='scan',sourceId='query-scan',encodings=dict(x=dict(field='condition',type='nominal',label='噪声 σ 与慢记忆 τ'),
                   y=dict(field='residence_difference',type='quantitative',label='驻留比例差')))]
    artifact = dict(surface='report',manifest=dict(version=1,surface='report',title=title,blocks=blocks,sources=sources,charts=charts,tables=tables),
                    snapshot=dict(version=1,status='ready',datasets=datasets),sources=sources)
    (REPORTS / 'progress-2026-10-04.artifact.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print('Generated narrative, canonical report artifact and archived key evidence')


if __name__ == '__main__':
    main()
