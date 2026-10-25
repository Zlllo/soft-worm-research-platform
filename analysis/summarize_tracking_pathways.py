"""Archive a completed full-I/O pathway diagnostic with explicit reuse counts."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


ROOT=Path('results/tracking-pathways-2026-10-25')
OUT=Path('docs/reports/assets/tracking-pathways-2026-10-25')


def main():
    d=json.loads((ROOT/'results.json').read_text());assert d['complete'] and len(d['runs'])==15
    assert all(d['source_hashes_unchanged'].values())
    summary={}
    for mode,treatment in [('intact','reused_baseline'),('no_chemical','frozen'),('no_gap','frozen'),('isolated','frozen'),('isolated','scratch_refit')]:
        rows=[r for r in d['runs'] if r['mode']==mode and r['treatment']==treatment];assert len(rows)==3
        item={}
        for scene in ['static','moving']:
            vals=[]
            for row in rows:
                ref=next(r for r in d['runs'] if r['mode']=='intact' and r['training_seed']==row['training_seed'])
                for a,b in zip(row['conditions'],ref['conditions']):
                    for k in ['scene','phase_rad','field','case_seed','initial_state']:assert a[k]==b[k]
                cells=[c for c in row['conditions'] if c['scene']==scene]
                vals.append(float(np.mean([c['metrics']['moving_tracking_fraction'] for c in cells])))
            item[scene]=dict(mean=float(np.mean(vals)),by_training_seed={str(r['training_seed']):v for r,v in zip(rows,vals)})
        summary[mode+'__'+treatment]=item
    baseline=json.loads(Path('results/tracking-timescale-2026-10-25/design.json').read_text())
    checks={str(s):hashlib.sha256((ROOT/f'parent-real8-{s}.pt').read_bytes()).hexdigest()==baseline['checkpoints'][f'real8-{s}-B16-U2048.pt'] for s in [7,17,27]}
    assert all(checks.values())
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['design.json','results.json']:shutil.copyfile(ROOT/name,OUT/name)
    (OUT/'summary.json').write_text(json.dumps(dict(groups=summary,parent_checkpoint_checks=checks,new_trajectories=144,reused_trajectories=36),indent=2)+'\n')
    with zipfile.ZipFile(OUT/'models-and-sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(ROOT))
        for name in ['analysis/summarize_tracking_pathways.py','tests/test_tracking_pathways.py']:
            z.write(name,'additional-source/'+name)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows=[]
    labels=[('原真实图','intact__reused_baseline'),('冻结后关闭全部化学连接（含自环）','no_chemical__frozen'),('冻结后关闭gap','no_gap__frozen'),('冻结后隔离节点，保留自环','isolated__frozen'),('隔离节点，从头训练','isolated__scratch_refit')]
    for label,key in labels:
        rows.append('| '+label+' | '+' | '.join(f"{summary[key][scene]['mean']*100:.2f}%" for scene in ['static','moving'])+' |')
    text='''# 输入输出接口与节点间连接依赖检查

日期：2026-10-25。3次新拟合、144条新轨迹，另复用36条完整模型基线轨迹。所有原模型均来自相同32条静态示范、2048次更新、0.3秒节点时间常数和full输入输出接口。

## 问题与方法

full接口允许输入直接进入全部节点、从全部节点读出动作。检查这条路径是否使任务无需节点间传递。

冻结干预分别关闭全部化学连接、gap连接，或全部节点间传递；隔离节点时保留两个化学自环与各节点的泄漏动态。另从相同初始化规则开始训练隔离节点模型，所有3次拟合完成后才评估。该模型保留71个nominal参数，但16个化学跨节点权重和4个gap参数被屏蔽，实际只有51个可影响输出的参数；不称有效容量匹配。

沿用病例10501／10502，线性／径向场，动态波幅0.05 K、周期60秒，初相0与π/2。每种模型／种子4条静态、8条动态轨迹。重复诊断病例不作为独立确认性测试。

## 结果

表为三个训练种子的平均追踪时间比例，各种子的值见摘要。

| 处理 | 静态追踪 | 动态追踪 |
| --- | --- | --- |
'''+ '\n'.join(rows)+'''

已有控制器确实依赖化学连接／节点间传递。冻结后隔离节点造成三个训练种子均下降，因此“接口允许旁路”不能直接等同于“训练完成后没使用连接”。

隔离节点重训的静态值分别100%、81.61%、61.59%，动态分别15.64%、15.69%、11.52%。其中一个种子说明当前静态病例上存在无节点间传递的解，但这组训练没有复制原模型的动态迁移。该结果仍受有效容量和优化预算限制，不能证明隔离节点模型在任何训练下都做不到。

去掉gap后动态均值提高至86.31%，但三个种子中一个下降、两个提高；这只提示该批模型的gap耦合未呈现统一迁移收益，尚未检验去gap后重新训练，也不能推断真实gap生物功能无益。

因此目前支持“连接参与计算”，尚未支持“真实布局优于其他布局”。下一步应结合[功能机制审计与任务设计](../mechanism-audit-and-task-plan-2026-10-25.md)，检查细胞动态与行为需求是否匹配。

## 验证与复现

单元测试确认隔离restricted接口后读出不再依赖温度或相位，而full接口仍可使用输入；检查被屏蔽跨节点权重梯度为零、自环保留、选择性屏蔽不改另一种连接、恒等模型精确复现输出。原模型检查点与前批SHA256一致，全部源码哈希与配对初态检查通过。

```bash
PYTHONPATH=src python3 analysis/probe_tracking_pathways.py
python3 analysis/summarize_tracking_pathways.py
```

首次拟合依赖原固定教师数据缓存及前批结果。无需重训即可读取归档审查：

[设计](assets/tracking-pathways-2026-10-25/design.json)、[完整结果](assets/tracking-pathways-2026-10-25/results.json)、[摘要](assets/tracking-pathways-2026-10-25/summary.json)、[源码和6个原／新检查点](assets/tracking-pathways-2026-10-25/models-and-sources.zip)、[SHA256](assets/tracking-pathways-2026-10-25/manifest.json)。
'''
    Path('docs/reports/tracking-pathways-2026-10-25.md').write_text(text)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
