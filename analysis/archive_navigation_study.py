"""Publish inspectable data, weights, source snapshots, pilots and checksums."""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import zipfile

import numpy as np
import torch


ROOT=Path('results/navigation-factorial-2026-10-25');OUT=Path('docs/reports/assets/navigation-study-2026-10-25')


def add_tree(z,root):
    for p in sorted(Path(root).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store':z.write(p,str(p))


def main():
    data=json.loads((ROOT/'results.json').read_text());summary=json.loads((ROOT/'summary.json').read_text())
    assert data['complete'] and len(data['runs'])==105
    assert all(data['source_hashes_unchanged'].values()) and all(data['input_hashes_unchanged'].values())
    OUT.mkdir(parents=True,exist_ok=True)
    for n in ['design.json','results.json','summary.json']:shutil.copyfile(ROOT/n,OUT/n)
    for source,dest in [('results/navigation-mechanisms-2026-10-25/results.json','mechanism-results.json'),
        ('results/navigation-numerical-refinement-2026-10-25.json','numerical-refinement.json'),
        ('results/navigation-teacher-regeneration-2026-10-25.json','teacher-regeneration.json'),
        ('results/navigation-blind-control-2026-10-25.json','blind-control.json'),
        ('results/navigation-initialization-audit-2026-10-25.json','initialization-audit.json'),
        ('results/navigation-example-feasibility-audit-2026-10-25.json','example-feasibility-audit.json'),
        ('results/navigation-feasible-2026-10-25/results.json','feasibility-results.json'),
        ('results/navigation-signal-audit-2026-10-25/results.json','signal-audit.json'),
        ('results/navigation-graph-isomorphism-audit-2026-10-25.json','graph-isomorphism-audit.json')]:shutil.copyfile(source,OUT/dest)
    sampling=Path('results/navigation-sampling-2026-10-25')
    rows=[json.loads((sampling/f'evaluation-{s["id"]:03d}.json').read_text()) for s in data['design']['specs']]
    (OUT/'sampling-results.json').write_text(json.dumps(dict(design=json.loads((sampling/'design.json').read_text()),runs=rows),indent=2)+'\n')
    zero=Path('results/navigation-zero-sweep-2026-10-25')
    zero_rows=[json.loads((zero/f'evaluation-{s["id"]:03d}.json').read_text()) for s in data['design']['specs']]
    (OUT/'zero-sweep-results.json').write_text(json.dumps(dict(design=json.loads((zero/'design.json').read_text()),runs=zero_rows),indent=2)+'\n')
    with zipfile.ZipFile(OUT/'models-and-sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        add_tree(z,ROOT);add_tree(z,sampling);add_tree(z,'results/navigation-examples-2026-10-25');add_tree(z,'results/navigation-zero-sweep-2026-10-25');add_tree(z,'results/navigation-feasible-2026-10-25');add_tree(z,'results/navigation-signal-audit-2026-10-25')
        extra=['analysis/summarize_navigation_factorial.py','analysis/plot_navigation_study.py','analysis/write_navigation_report.py',
            'analysis/archive_navigation_study.py','analysis/navigation_sampling_controls.py','analysis/navigation_examples.py',
            'analysis/plot_navigation_examples.py','analysis/navigation_blind_control.py','analysis/navigation_no_sweep_control.py','analysis/navigation_feasibility_replication.py','analysis/navigation_signal_audit.py','analysis/report_navigation_feasible.py','analysis/navigation_feasible_reference.py','analysis/audit_navigation_graphs.py','docs/navigation-protocol-2026-10-25.md']
        for p in extra:z.write(p,'additional-source/'+p)
        for p in sorted(Path('tests').glob('test_navigation*.py')):z.write(p,'additional-source/'+str(p))
    with zipfile.ZipFile(OUT/'teacher-data.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in data['design']['input_sha256']:z.write(p,p)
    with zipfile.ZipFile(OUT/'mechanism-source-models.zip','w',zipfile.ZIP_DEFLATED) as z:
        add_tree(z,'results/navigation-mechanisms-2026-10-25');add_tree(z,'data/reference/elife68848')
        z.write('tests/test_motor_feedback.py','additional-source/tests/test_motor_feedback.py')
    pilots=['navigation-preflight','navigation-a2c-preflight','navigation-a2c-long-preflight',
        'navigation-bc-preflight','navigation-gru16-bc-preflight','navigation-window-preflight',
        'navigation-window-ppo-preflight','navigation-factorial-pretraining-plan0','navigation-mechanisms-initial-assay']
    with zipfile.ZipFile(OUT/'development-pilots.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in pilots:
            root=Path('results')/(name+'-2026-10-25')
            if root.exists():add_tree(z,root)
        for p in ['analysis/train_navigation.py','analysis/navigation_teacher.py','analysis/navigation_window_learning.py',
            'analysis/navigation_gru16_preflight.py','src/thermotaxis/navigation_policy.py']:
            z.write(p,'additional-source/'+p)
        for p in Path('results').glob('navigation-*-plan-2026-10-25.json'):z.write(p,str(p))
    index=dict(fits=105,unique_setting_groups=35,training_seeds=[7,17,27],shared_teacher_episodes=32,
        teacher_control_interactions_per_model=38400,learner_control_interactions_per_model=262144,
        primary_rollouts=summary['primary_cases'],frozen_transfer_rollouts=summary['transfer_cases'],
        center_sensor_rollouts=summary['sampling_cases'],supplemental_zero_sweep_rollouts=1680,physics_replication_RL_fits=42,physics_replication_rollouts=6048,unique_primary_case_seeds=32,
        paired_transfer_case_seeds=16,failed_fits=summary['failed_fits'],
        pilot_notes=['All development cases retired.','Pilot prefix/checkpoint reruns are not independent fits.',
            'Plan0 predates a zero-event-jitter validation fix and contains no training.',
            'Initial mechanism assay differs only by missing paired no-pulse analysis; no extra independent evidence.'],
        runtime=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__))
    (OUT/'archive-index.json').write_text(json.dumps(index,indent=2)+'\n')
    (OUT/'README.md').write_text('''# 本轮材料与复现

先读根目录报告`docs/reports/navigation-study-2026-10-25.md`与执行协议。JSON包含全部种子，不仅均值；时间未达到保留null。图中的范围不是置信区间。`results.json`保留原始60°压力条件；`feasibility-results.json`是随后因速度下界问题而进行的15°、新病例复查，不把它伪装成原预登记结果。

- `models-and-sources.zip`：主105组拟合的315个模型检查点，以及复用示范检查点的42组几何复查所新增84个RL检查点、训练曲线、全部评估、固定例图轨迹、原始源码快照与补充分析脚本。路径从`results/`开始，解压到仓库根目录即可恢复。
- `teacher-data.zip`：共享教师数据与两种感觉编码的实际数组，路径从`results/`开始；SHA256与设计一致。
- `mechanism-source-models.zip`：AFD现象学检查、Ji2021源模型Python移植、原始附件ZIP、归一化阅读副本及许可／出处。
- `development-pilots.zip`：包含未学会、不稳定的预检与源代码。重复前缀和初版计划不计独立证据。

无需重训即可读取`summary.json`、`results.json`、`sampling-results.json`、`mechanism-results.json`。若重建图表，先将前两个ZIP解压到仓库根目录，再解压机制模型ZIP；保持记录的源码及PyTorch/NumPy版本。

```bash
PYTHONPATH=src python3 analysis/summarize_navigation_factorial.py
MPLCONFIGDIR=/tmp/worm-matplotlib PYTHONPATH=src python3 analysis/plot_navigation_study.py
MPLCONFIGDIR=/tmp/worm-matplotlib PYTHONPATH=src python3 analysis/plot_navigation_examples.py
```

若完整重训，在一个新的输出位置或移走已恢复的实验目录后，运行`analysis/navigation_factorial.py prepare`；`navigation_teacher.data()`会按固定种子重建共享缓存。准备结束后执行全部fit，再执行evaluate，最后collect。`navigation_sampling_controls.py prepare`也应在评估之前运行，随后evaluate补充中心测温记录。不得把公开结果读过的病例再称为新的最终确认集。

物理对象仍是规定运动学；原始运动反馈模型用归一化空间单位。函数核验不是实测细胞动力学标定，模型结果不是活体实验结论。
''')
    for p in OUT.glob('*.zip'):
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None,p
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('archived',index,'file_count',len(manifest),flush=True)


if __name__=='__main__':main()
