# 本阶段公开材料与原始数据索引

日期：2026-11-09。研究暂停，本次整理新增科学运行数为0。

先读[综合总结](../../research-stage-summary-2026-11-09.md)，再读[完整系统验收](../../b-motor-validation-2026-11-09.md)、[组件隔离](../../c1-muscle-integration-isolation-2026-11-09.md)与[独立复核](../../c1-muscle-integration-review-2026-11-09.md)。

## 阅读结论与核对证据

- [当前验收状态](acceptance.json)：3/5，60%；完整系统收敛与持续推进未通过。
- [组件原始汇总](evidence/c1-muscle-integration-isolation-2026-11-09/summary.json)、[独立检查](evidence/c1-muscle-integration-isolation-2026-11-09/independent-validation.json)、[主线发布前复算](evidence/c1-muscle-integration-isolation-2026-11-09/publication-parent-audit.json)。
- [独立分支复核](evidence/c1-muscle-integration-review-2026-11-09/review.json)：保留未改善的负结果，以及正文复核范围的限制。
- [完整系统主线验收](evidence/b-motor-validation-2026-11-09/main-acceptance.json)及其旁边的独立归档审查记录。
- [原始数值实现诊断](evidence/b-motor-convergence-2026-11-09/numerical-diagnostic.md)。

## 较大原始归档

下列文件作为[本仓库公开Release附件](https://github.com/Zlllo/soft-worm-research-platform/releases/tag/research-stage-2026-11-09)保存，避免把大体积仿真数据加入Git历史。[下载清单](release-assets.json)记录下载地址、准确字节数、原始项目路径与SHA-256。原始ZIP内容未重写；较小归档仍按原路径保留在仓库。

| 数据归档 | 大小（MB） | 下载 |
|---|---:|---|
| b-motor-convergence-2026-11-09 | 72.6 | [冻结原始归档](https://github.com/Zlllo/soft-worm-research-platform/releases/download/research-stage-2026-11-09/b-motor-convergence-2026-11-09-reproduction.zip) |
| b-motor-feedback-causality-2026-11-09 | 153.6 | [冻结原始归档](https://github.com/Zlllo/soft-worm-research-platform/releases/download/research-stage-2026-11-09/b-motor-feedback-causality-2026-11-09-reproduction.zip) |
| body-load-stage2-2026-11-03 | 259.0 | [冻结原始归档](https://github.com/Zlllo/soft-worm-research-platform/releases/download/research-stage-2026-11-09/body-load-stage2-2026-11-03-reproduction.zip) |
| body-topology-calibration-2026-11-03 | 128.1 | [冻结原始归档](https://github.com/Zlllo/soft-worm-research-platform/releases/download/research-stage-2026-11-09/body-topology-calibration-2026-11-03-reproduction.zip) |
| c1-muscle-integration-isolation-2026-11-09 | 664.3 | [冻结原始归档](https://github.com/Zlllo/soft-worm-research-platform/releases/download/research-stage-2026-11-09/c1-muscle-integration-isolation-2026-11-09-reproduction.zip) |
| full302-closed-loop-2026-11-05 | 63.2 | [冻结原始归档](https://github.com/Zlllo/soft-worm-research-platform/releases/download/research-stage-2026-11-09/full302-closed-loop-2026-11-05-reproduction.zip) |
| full302-mechanisms-2026-11-07 | 74.2 | [冻结原始归档](https://github.com/Zlllo/soft-worm-research-platform/releases/download/research-stage-2026-11-09/full302-mechanisms-2026-11-07-reproduction.zip) |

下载后可放回清单的original_path，以兼容旧报告记录的本地路径；也可直接解压到项目根目录。所有归档内的results路径均是解压后的路径，仓库本身不提交整棵results目录。

## 离线复算与重新仿真

最新肌肉组件归档中包含325个载荷及MANIFEST.json；解压后在项目根目录执行：

```sh
OPENBLAS_NUM_THREADS=1 python analysis/neuromechanics/b_motor_validation/integration_analyze.py
OPENBLAS_NUM_THREADS=1 python analysis/neuromechanics/b_motor_validation/integration_verify.py
```

上述两项仅复算已有数据，不启动NEURON。固定环境为Python 3.12.14、NumPy 2.3.5；科学运行还使用NEURON 9.0.2，需另行编译机制。原macOS arm64本机程序不能视为其他平台已验证；平台迁移须重建并记录新来源。当前保持暂停，不要把准备脚本视作自动继续的指令。

源代码、参数锁、准备失败、科学失败与分析脚本均随对应冻结归档保留。此目录的evidence是阅读用副本，不替代各归档内的原始文件。[源码恢复记录](source-recovery.json)说明了三个仅在发布副本中从冻结归档恢复的云占位文件，原工作目录未覆盖。

## 本次同步检查

发布副本代码检查为215项通过、10项原生身体集成检查因未配置专用程序而跳过；未重跑科学仿真。见[本次检查记录](publication-validation.json)与[测试输出](publication-tests.txt)。最新组件归档另外解压复算，汇总逐字节一致。生成SVG及补丁上下文中的空白按原样保留。

公开仓库中的历史报告仅调整必要的下载或汇总链接；冻结原文、报告哈希与原始数据仍保留在各ZIP中。
