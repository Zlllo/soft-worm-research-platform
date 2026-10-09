# 软体线虫项目进度汇报材料

整理日期：2026年10月6日。此文件夹汇集现有阶段报告、展示图表和验证记录；原始文件继续保留在项目的 `docs/` 中。归档日期不代表重新运行了全部历史实验。

## 建议展示顺序

本次向老师汇报可先使用[阶段进展汇报与讲述稿](阶段进展汇报.md)，正文以CCB连续中心线和DDPG训练、冻结评估为主，保留3张易读配图和训练回放链接；新力学简要说明，详细验证作为补充。该稿于2026年10月9日更新。

1. [阶段进展总览](阶段进展总览.md)：初版与当前工作对照，平台改造、训练状态和下一步安排。
2. [最新身体力学与策略接入报告](docs/reports/progress-2026-10-04-body.md)：规定形变身体、介质耗散、四方向执行和冻结策略迁移。
3. [点模型有限采样与长时趋温报告](docs/reports/progress-2026-10-04-closure.md)：局部感知、记忆和长时统计结果。
4. [点模型统计汇报页面](docs/reports/progress-2026-10-04.html)：用于浏览器展示；页面对应最初544对统计批次，尚未包含最新身体成果。
5. [项目结构图](docs/project-structure-map.md)：各阶段与平台的关系。
6. [原平台算法与可调参数说明](原平台算法与可调参数说明.md)：模型、算法、连续动作入口，以及各滑块的含义和实际生效范围。

## 展示图表

| 展示内容 | 文件 |
| --- | --- |
| 连续身体的周期位形 | [身体位形](docs/reports/assets/phase2-body-2026-10-04/body-shapes.png) |
| 身体推进和累计介质耗散 | [轨迹与耗散](docs/reports/assets/phase2-body-2026-10-04/trajectory-energy.png) |
| 不同幅值与阻力比 | [驱动参数扫描](docs/reports/assets/phase2-body-2026-10-04/amplitude-drag-scan.png) |
| 时间与空间数值收敛 | [收敛图](docs/reports/assets/phase2-body-2026-10-04/numerical-convergence.png) |
| 四方向命令的实际身体执行 | [四方向轨迹](docs/reports/assets/phase2-four-action-2026-10-04/four-action-bridge.png) |
| 冻结Q表迁移，8.70秒离域终止 | [迁移轨迹](docs/reports/assets/phase2-four-action-2026-10-04/frozen-policy-transfer.png) |
| 点模型首次进入舒适区 | [首次进入曲线](docs/reports/assets/phase1-progress-2026-10-04/confirmation-survival.png) |
| 点模型噪声与记忆比较 | [参数扫描](docs/reports/assets/phase1-progress-2026-10-04/exploratory-scan.png) |
| 长时舒适区驻留 | [后期驻留](docs/reports/assets/phase1-boundary-stationarity-2026-10-04/late-occupation.png) |

## 训练与验证证据

- [已保存Q-learning训练记录](docs/reports/assets/legacy-policy-bridge-2026-10-04/training-records.json)：60轮的起点、终点和累计奖励。
- [精确Q表及训练来源](docs/reports/assets/legacy-policy-bridge-2026-10-04/new-training-interface-qtable.json)：固定种子新训练的接口样本，不是组员历史权重。
- [冻结旧策略接入方法](docs/legacy-policy-bridge.md)：训练条件、观测差异、坐标映射和限制。
- [身体机械独立验证](docs/validation/phase2-body-validation-2026-10-04.md)。
- [策略桥接独立验证](docs/validation/phase2-bridge-validation-2026-10-04.md)。
- 其余批次的统计及理论验证位于 `docs/validation/`；逐批设计、汇总和关键记录位于 `docs/reports/assets/`。

## 使用说明

原平台已完成多项功能扩展，启动入口仍为项目根目录下的 `程序/app.py`。在项目根目录运行 `python3 -m streamlit run 程序/app.py`，可展示原平台的身体选择、离散/连续动作、训练图、动画与日志。最新 `src/thermotaxis/` 身体力学核心尚未接入该界面，应使用此归档中的图表及报告展示。

汇报中的“训练样本”“机制统计”“力学验证”分别按各自实验条件解释。点模型没有强化学习训练，身体机械基准没有趋温控制；目前没有新力学身体成功趋温训练与复杂环境泛化的完整结果。

完整实验输出仍在项目根目录的 `results/` 中，默认不上传GitHub；这里保留用于汇报的关键证据。代码、配置和实验脚本分别在项目根目录的 `src/`、`configs/`、`analysis/` 和 `程序/` 中。
