# 完整系统联合候选：数据、复核与恢复入口

日期：2026-11-12；本阶段6条候选运行已结束，无重跑、无参数搜索／训练。完整系统全项收敛与推进均未通过；详见[主线综合结果](../../full302-joint-numerics-main-2026-11-12.md)。

- [A原始数据、源代码与分析](https://github.com/Zlllo/soft-worm-research-platform/releases/download/research-mechanisms-2026-11-13/full302-joint-numerics-2026-11-12.zip)（约95.3 MB），[SHA256](../full302-joint-numerics-2026-11-12/reproduction.zip.sha256)。
- [B独立复核包](https://github.com/Zlllo/soft-worm-research-platform/releases/download/research-mechanisms-2026-11-13/full302-joint-numerics-review-2026-11-12.zip)（约379 KB），[SHA256](../full302-joint-numerics-review-2026-11-12/reproduction.zip.sha256)。
- [旧父版原始包](https://github.com/Zlllo/soft-worm-research-platform/releases/download/research-stage-2026-11-09/b-motor-convergence-2026-11-09-reproduction.zip)：B的完整父版交叉核对依赖此包，哈希和步骤见其archive-dependencies.json。
- [主线补充验收包](https://github.com/Zlllo/soft-worm-research-platform/releases/download/research-mechanisms-2026-11-13/full302-joint-numerics-main-2026-11-12-main-evidence.zip)：独立原始数组计算、关键旧文件保护、闭环／通道分布描述、最终验收及本报告。

在新的空目录解压A包及主线包，使用Python3.12、NumPy2.3.5、SciPy1.16.3运行以下命令，仅复算已保存数据，不启动NEURON或身体程序：

```sh
OPENBLAS_NUM_THREADS=1 python results/full302-joint-numerics-main-2026-11-12/parent_pair_audit.py
OPENBLAS_NUM_THREADS=1 python results/full302-joint-numerics-main-2026-11-12/parent_feedback_audit.py
OPENBLAS_NUM_THREADS=1 python results/full302-joint-numerics-main-2026-11-12/parent_motor_description.py
OPENBLAS_NUM_THREADS=1 python results/full302-joint-numerics-main-2026-11-12/parent_transient_description.py
```

第一项独立重算6组配对的全部16判据；第二项按保存时刻真实侧长重建反馈，不补造未保存的1ms身体样本。后两项是明确标注的探索性／事后描述，不改变验收，也不构成相位拟合后通过或混沌证明。

A包自己的复算入口为joint_analyze.py、joint_verify.py。B的完整复算需把父版包、A包与B包解压到同一空根目录，再按其archive-dependencies.json的6个命令顺序运行。

所有未通过项、先前实现、候选参数和来源均保留。没有把候选替换成默认主模型，也没有在本阶段追加新的修复或训练。研究已由本次“继续推进”恢复，当前状态为这一大阶段完成；旧暂停标记仅描述旧快照。
