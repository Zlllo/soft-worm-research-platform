# 本轮材料与复现

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
