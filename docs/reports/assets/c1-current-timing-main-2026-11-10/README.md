# 2026-11-10综合验收与暂停入口

先读[本阶段结果](../../c1-current-timing-main-2026-11-10.md)。本阶段已完成的科学条件为12个小控制、6个新增孤立肌肉主条件及6个旧基线复用；完整302/身体/训练新增为0。

- [分支A完整原始归档](https://github.com/Zlllo/soft-worm-research-platform/releases/download/research-mechanisms-2026-11-13/c1-current-timing-2026-11-10.zip)与[SHA-256](../c1-current-timing-2026-11-10/reproduction.zip.sha256)：原始迹线、来源、参数、两版未执行协议记录及实际v2、代码、分析与独立审计。
- [分支B独立复核归档](https://github.com/Zlllo/soft-worm-research-platform/releases/download/research-mechanisms-2026-11-13/c1-current-timing-review-2026-11-10.zip)：独立算法、官方源码依据、原始数据和已保留的工具失败。
- [主线验收补充包](https://github.com/Zlllo/soft-worm-research-platform/releases/download/research-mechanisms-2026-11-13/c1-current-timing-main-2026-11-10-main-evidence.zip)：主线独立计算、归档审查、综合报告和最终暂停记录。它不重复存储全部大体积迹线，需与上面的原始归档配合使用。

离线复算不会启动NEURON：先将A原始归档解压到独立目录，再将主线补充包解压到同一根目录。使用Python 3.12及NumPy 2.3.5执行：

```sh
OPENBLAS_NUM_THREADS=1 python results/c1-current-timing-main-2026-11-10/parent_control_audit.py
OPENBLAS_NUM_THREADS=1 python results/c1-current-timing-main-2026-11-10/parent_main_audit.py
```

前者独立核对12个小控制及实际通道反转电位；后者独立核对12条主数据与120项步长指标。分支各自的分析／审计命令见其报告。原始数据、失败记录、旧完整系统未通过结果均不改写。

按用户“完成目前后暂停”，后续完整网络复验、修复及训练均不自动启动。恢复时应先读本阶段综合报告、`results/c1-current-timing-main-2026-11-10/final-acceptance.json`与两分支的`stage-pause.json`。
