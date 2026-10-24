# 软体仿生线虫研究平台

2026-10-24 普通网络：去均值 GRU16 已在三个训练种子的二维基础评估中达到 100% 追踪时间比例；原始 GRU 追加示范与优化后为 91.92%。头摆频率迁移暴露了 MLP 的局限。当前为监督模仿学习，真实回路的样本效率与物理约束收益仍待对照。见 [训练结果、迁移与完整材料](docs/reports/neural-tracking-2026-10-24.md)。

2026-10-24 二维场景：持续推进、头部摆动测温和有限转向响应已实现，444条验证完成。这是规定运动学基准。见 [场景验证](docs/reports/isothermal-scene-2026-10-24.md)。

2026-10-24 编码与可学习性：42组编码×拓扑探索及15次监督可学习性对照已完成，简化真实连接尚未显示优势。记录见 [编码与拓扑](docs/reports/encoding-topology-2026-10-24.md)、[可学习性](docs/reports/trainability-matrix-2026-10-24.md)和 [首批回路探索](docs/reports/connectome-pilot-2026-10-24.md)。

2026-10-24 主线：已导入修正版真实连接数据。下一项比较真实连接、度保持重排与普通记忆网络，再解释它们与物理约束的配合。见 [研究主线](docs/research-mainline.md)及 [场景设计](docs/scenario-design.md)。

2026-10-08：原可视化平台 CCB 连续测温、奖励、几何约束与 DDPG 导航已优化。单热源800轮/500步、三个随机种子、8起点冻结评估共24/24次进入中心0.75格内；该结果属于原平台额外信息基线，不是 SI 身体趋温结论。修改、失败记录及复现方法见 [连续导航验证报告](docs/reports/ccb-continuous-navigation-2026-10-08.md)。

2026-10-08 界面改版：极简浅色布局、轻量交互、训练后自动恢复配置、切换参数保留结果。界面说明与截图见 [界面改版记录](docs/reports/platform-ui-2026-10-08.md)。

2026-10-06 阶段汇报材料已集中整理到 [进度汇报_2026-10-06](进度汇报_2026-10-06/README.md)，包含进展总览、阶段报告、展示图表、训练样本及验证记录。

## 趋温重建主线（第0—2阶段）

组员 PR #1 已作为历史平台基线合并。新的研究主线在 `src/thermotaxis/` 中独立发展，研究局部含噪测温、短期记忆、身体运动与机械耗散怎样共同产生趋温。

第 0 阶段已经固定模型契约、SI 单位、独立的物理/传感/控制时钟、随机源和控制器观测权限。第 1 阶段新增一维含噪点模型、快慢记忆控制器，以及与无响应对照共享随机种子的配对集合实验。运行方式见 `docs/phase0-usage.md` 和 `docs/phase1-point-model.md`，完整约定见 `docs/model-contract.md`。

点模型用于回答“局部含噪测温与短期记忆能否产生趋温偏置”。它还没有身体形变与机械耗散，因此其结果属于机制基准，不能代替后续软体身体结论。项目结构和模块关系见 `docs/project-structure-map.md`，独立核验记录见 `docs/validation/phase0-phase1-validation.md`。

2026-10-04既有批次：544对统计实验、1,536对控制周期/时窗比较与256条连续局部弱响应轨迹已核验。参考180s下约47.7%—49.1%决策触及速率下限，弱响应式不定量预测参考强响应；该批次完整回归76项通过。

点模型最新批次：有限采样一阶系数与576条局部轨迹核验通过；三种域宽共192对3600秒轨迹、后期块与区间已复算，响应末块驻留约82%—83%，严格稳态仍是研究问题；该批次完整回归86项通过。

当前第二阶段：`src/thermotaxis/phase2_body.py`建立固定材料弧长、完整形变速度、RFT自由平移/旋转及介质耗散。16组驱动扫描、12组收敛比较与独立密网格核验完成，身体基准批次完整回归102项通过。参考1mm身体4秒推进0.37914mm；阻力工作值尚未标定。随后已建立有限响应四方向执行器、材料头部测温入口和独立时钟；旧Q表及DQN冻结接口已完成。历史权重暂未找到，本轮使用明确标为新训练的旧程序接口样本；实际新样本迁移于8.70s头部离域后终止，未完成导航；该身体桥接批次完整回归115项通过；合并批次完整回归见下文“当前状态”。身体局部记忆趋温及弹性尚待开展。

点模型材料见 `docs/phase1-memory-scaling.md`、`docs/phase1-sampling-response.md`、`docs/phase1-theory-sensitivity.md`、`docs/phase1-boundary-stationarity.md`。身体推导见 `docs/phase2-prescribed-body.md`，四方向执行见 `docs/phase2-four-action-bridge.md`，旧策略冻结接入见 `docs/legacy-policy-bridge.md`，最新老师续报见 `docs/reports/progress-2026-10-04-body.md`。既有离线HTML仍对应原统计批次。

运行第 1 阶段配对集合：

```bash
PYTHONPATH=src python3 -m thermotaxis phase1-ensemble \
  --config configs/phase1_reference.json \
  --replicates 100 \
  --output-root results \
  --repository .
```

这是一个基于 Python 的二维线虫趋温仿真平台。原始程序已经包含温度场、线虫身体、Q-Learning、DQN、Dueling DQN、迁移学习、课程学习和 Streamlit 可视化界面。当前仓库在此基础上继续改造，目标是逐步形成可用于比较不同身体表征、不同控制方式、复杂温度场和群体间接交互的研究平台。

GitHub 仓库：

- https://github.com/Zlllo/soft-worm-research-platform

## 当前状态

新的趋温主线当前包括：

- 第 0 阶段：严格配置、单位与信息边界、独立时钟和随机源、可复现结果包。
- 第 1 阶段：一维点模型、局部含噪测温、双时间尺度记忆、有界随机转向，以及共享种子的响应/无响应配对集合。
- 理论与诊断：连续无响应首次到达、离散概率传播、弱响应局部替代模型，以及控制周期/观察窗口和速率夹限诊断。身体力学与介质耗散已有独立模块，身体趋温统计尚未开展。
- 合并批次记录的完整回归：`PYTHONPATH=src python3 -m pytest tests 程序/test_core.py -q` 共176项通过，另有14条既有 matplotlib/pyparsing 弃用警告；原平台连续导航验证见 `docs/reports/ccb-continuous-navigation-2026-10-08.md`，SI身体核验见 `docs/validation/phase2-body-validation-2026-10-04.md`。

以下内容描述已经合并的旧平台基线：

旧平台历史基线已完成以下三个改造阶段；这里的阶段编号与上文趋温重建第 0—2 阶段分别使用：

- 第一阶段：整理可运行基础，修复测试入口，增加依赖说明和 pytest 测试。
- 第二阶段：身体模型解耦，把训练引擎从直接创建 `Worm2D` 改为通过身体模型工厂创建。
- 第三阶段：新增连续中心线身体和主动形变身体的第一版实现。

旧平台合并时的验证结果：

- `python3 -m pytest`：14 个旧平台测试通过
- `python3 程序/test_simple.py`：通过
- `py_compile`：通过

组员 PR #1 已作为历史平台基线合并；[PR #2](https://github.com/Zlllo/soft-worm-research-platform/pull/2) 已于 2026-10-09 合并到 `main`。当前 `main` 同时包含原平台 CCB/DDPG 连续导航与界面更新，以及 `src/thermotaxis/` 中第 0—2 阶段的研究代码和验证材料。`codex/phase0-foundation` 为保留的已合并开发分支。文中 14、76、86、102、115 项分别属于历史基线或阶段批次，不表示当前完整回归数量。

## 环境要求

- Python 3.8 到 3.11
- 推荐开发基准：Python 3.9
- 核心依赖：`numpy`
- 测试依赖：`pytest`
- 界面依赖：`streamlit`、`matplotlib`、`Pillow`
- 深度学习依赖：`torch`
- 动画导出建议安装 `ffmpeg`

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

## 运行方式

运行核心冒烟测试：

```bash
python3 程序/test_simple.py
```

运行 pytest：

```bash
python3 -m pytest
```

启动 Streamlit 界面：

```bash
streamlit run 程序/app.py
```

使用 DQN 或 Dueling DQN 前，需要确认 `torch` 已安装。

## 原始代码结构记录

以下是原始项目中已经存在的主要结构和职责。本节用于记录改造前的代码组织方式，方便后续写论文、报告或对比实验时回溯。

### `程序/app.py`

Streamlit 图形界面入口。

原始功能包括：

- 选择实验模式：标准训练、迁移学习、课程学习。
- 配置温度场类型、训练轮数、每轮步数、学习率、折扣因子、epsilon 衰减等参数。
- 配置线虫身体参数，例如节段数、节段长度、头部半径、身体宽度、转向角、前进速度等。
- 配置噪声参数，例如位置噪声、角度噪声、温度噪声。
- 选择学习方法：Q-Learning、DQN、Dueling DQN。
- 展示训练进度、奖励曲线、轨迹图、动画和部分统计结果。

这个文件主要负责用户交互，不直接实现核心物理或学习逻辑。

### `程序/simulation_engine.py`

训练流程引擎。

原始功能包括：

- `run_standard_simulation_engine(...)`：标准训练流程。
- `run_transfer_simulation_engine(...)`：迁移学习流程，先在源温度场训练，再到目标温度场测试。
- `run_curriculum_simulation_engine(...)`：课程学习流程，按多个温度场阶段训练，并留出一个阶段测试泛化能力。
- `zero_shot_testing_engine(...)`：零学习测试流程。
- `save_and_visualize_results(...)`：保存日志、Q 表、训练图和动画。
- 步数统计图、基础奖励图、简化动画等辅助结果生成。

原始问题是训练引擎直接创建 `Worm2D(...)`，训练流程和旧身体模型绑定较深，不利于替换身体表征。

### `程序/core/environment.py`

二维温度环境和实验配置。

主要内容：

- `ExperimentConfig`：实验名称、输出目录、日志文件、图表文件、动画文件、Q 表文件等路径配置。
- `Environment2D`：保存温度矩阵、环境尺寸、最佳温度点，并提供温度读取和状态向量生成。
- `get_state_vector(...)`：生成旧版 8 维状态，用于 Q-Learning 和神经网络输入。

原始状态向量仍是固定 8 维，后续计划会扩展为 `state_v2`。

### `程序/core/worm_body.py`

原始线虫身体和决策逻辑。

核心类是 `Worm2D`。它表示一条多节段线虫：

- 身体由多个离散节段组成，保存在 `body_segments` 中。
- 头部移动后，后续节段跟随。
- 每个节段可感知温度。
- 内部带有 Q 表、奖励累计、能量、肌肉疲劳、肌肉波、弹性约束等状态。
- 提供 `decide_move(...)` 执行动作选择、移动、奖励计算和 Q 表更新。
- 支持在 PyTorch 可用时使用 DQN/Dueling DQN。

需要特别说明：原始 `Worm2D` 不是连续软体模型，它更接近“离散节点链式身体”。它适合作为对照模型保留。

### `程序/core/neural_networks.py`

深度强化学习相关模块。

主要内容：

- `SimpleNeuralNetwork`
- `DuelingDQN`
- `DQNAgent`
- `ExperienceReplay`
- `PrioritizedReplayBuffer`
- `SumTree`

当前深度学习只依赖 PyTorch，没有引入外部强化学习框架。

### `程序/core/utils.py`

工具函数集合。

原始功能包括：

- 保存训练日志：`save_training_log(...)`
- 保存 Q 表：`save_q_table(...)`
- 初始化神经网络：`setup_neural_network(...)`
- 重置每轮线虫状态：`reset_worm_for_new_round(...)`
- 创建温度场：`create_temperature_environment(...)`
- 生成动态旋转温度场
- 生成堆叠状态：`get_stacked_state(...)`

### `程序/core/visualization.py`

训练结果可视化。

主要内容：

- 绘制温度场、训练轨迹、奖励曲线。
- 生成静态环境 GIF 动画。
- 生成动态环境 MP4 动画。
- 根据温度、肌肉疲劳、张力等状态改变身体显示效果。

### `程序/core/training_stats.py`

训练统计辅助模块。

主要内容：

- `StepTracker`：记录到达目标附近所需步数。
- `DualCenterTracker`：双热源场景下记录高温源和低温源访问情况。
- 对训练过程的步数、成功率、阶段表现做统计。

### `程序/test_simple.py`

脚本式冒烟测试。

覆盖内容：

- 温度场生成。
- 环境读取和 8 维状态向量。
- `Worm2D` 创建。
- 身体模型接口。
- 单步移动。
- 2 轮短训练。
- DQN 初始化。

### `程序/test_core.py`

后来新增的 pytest 测试文件。

当前覆盖内容：

- 静态温度场。
- 动态温度场更新。
- 旧身体创建。
- 旧身体约束指标。
- 连续中心线身体。
- 主动形变身体。
- 单步学习状态更新。
- 短训练。
- 指标 JSON 导出。
- DQN 初始化。
- 标准训练、迁移学习、课程学习的身体模型工厂接入。

## 旧平台早期改造记录

本节保留早期平台改造记录。后续 CCB 连续控制、观测、几何约束和界面更新以 [2026-10-08 连续导航报告](docs/reports/ccb-continuous-navigation-2026-10-08.md) 和 [界面改版记录](docs/reports/platform-ui-2026-10-08.md) 为准；以下历史能力描述不作为当前实现的完整清单。

### 1. 可运行基础整理

完成内容：

- 修复 `test_simple.py` 与当前 `Worm2D.__init__` 参数不一致的问题。
- 增加 `requirements.txt`，声明 numpy、pytest、matplotlib、streamlit、torch、Pillow。
- 增加 pytest 测试入口 `程序/test_core.py`。
- 让核心测试不依赖真实 Streamlit 和 Matplotlib，测试里使用轻量替身对象。
- 保留 `test_simple.py` 作为脚本式冒烟测试。

这部分让项目可以用明确命令验证，而不是只能从界面手动运行。

### 2. 身体模型接口和工厂

完成内容：

- 新增统一身体接口 `BodyModel`，要求身体模型提供：
  - `reset(...)`
  - `get_observation(...)`
  - `apply_action(...)`
  - `step_physics(...)`
  - `get_geometry(...)`
  - `get_metrics(...)`
- 将旧 `Worm2D` 保留为对照模型。
- 新增 `Worm2DModelAdapter`，把旧 `Worm2D` 包装为统一接口模型。
- 新增 `create_body_model(...)` 工厂函数。
- 新增 `create_training_body_model(...)`，训练引擎通过它读取 `training_params["body_model_type"]` 并创建身体模型。

现在训练流程不再必须直接写 `Worm2D(...)`。

### 3. 训练入口接入身体模型工厂

完成内容：

- 标准训练入口已经通过身体模型工厂创建身体。
- 迁移学习入口已经通过身体模型工厂创建源模型和目标模型。
- 课程学习入口已经通过身体模型工厂创建训练模型、测试模型和对照模型。
- 增加回归测试，确认三个训练入口都走身体模型工厂。

这让后续新增身体模型时，不需要重写三套训练流程。

### 4. 身体比较指标导出

完成内容：

- 扩展 `get_metrics(...)`，让身体模型返回可比较指标。
- 新增 `save_body_metrics(...)`。
- 每次保存实验结果时会额外生成 `body_metrics.json`。

当前指标包括：

- 实际身体长度
- 目标身体长度
- 身体长度误差
- 平均节段长度
- 节段长度误差
- 曲率均值
- 最大曲率
- 曲率约束违例次数
- 曲率约束违例率
- 能量
- 能量比例
- 肌肉疲劳
- 总奖励
- 当前步数
- 与最佳温度点的距离

这些指标用于比较旧链式身体、连续中心线身体和主动形变身体。

### 5. 连续中心线身体

新增 `ContinuousCenterlineBody`。

它的意义：

- 从旧的“多节段链条”推进到“连续中心线表征”。
- 身体仍然用采样点存储，但这些点代表一条固定弧长中心线。
- 不是完整有限元软体仿真，也不是连续介质求解器。
- 当前实现适合作为研究平台中的第二类身体表征。

当前能力：

- 固定采样点数量。
- 固定总弧长。
- 长度保持。
- 曲率限制。
- 阻尼。
- 能量耗散。
- 疲劳累积。
- 提供几何数据和指标。
- 支持 `apply_action(...)` 与 `step_physics(...)`。
- 支持标准训练短路径中的 `decide_move(...)`。

工厂名称：

- `continuous_centerline`
- `centerline`
- `continuous`

### 6. 主动形变身体

新增 `ActiveDeformationBody`。

它基于 `ContinuousCenterlineBody`，增加主动传播波。

当前能力：

- 波幅：`wave_amplitude`
- 频率：`wave_frequency`
- 相位：`wave_phase`
- 传播速度：`wave_speed`
- 波长：`wave_length`
- 波驱动推进增益：`propulsion_gain`
- 输出主动波相关指标。

工厂名称：

- `active_deformation`
- `active_wave`
- `wave_body`

需要说明：主动形变目前是研究平台的第一版波驱动近似模型，还没有和真实肌肉力学、接触力或流体阻力做严格耦合。

### 7. GitHub 仓库

已创建并推送到 GitHub：

- 仓库：`Zlllo/soft-worm-research-platform`
- 地址：https://github.com/Zlllo/soft-worm-research-platform
- 可见性：Public
- 分支：`main`

早期提交记录包括：

- `46f225c Establish worm simulation baseline`
- `fd4f851 Route standard training through body model factory`
- `3b4e77b Add standard training factory regression test`
- `c5f85ed Use body model factory across training engines`
- `33afcbb Export body comparison metrics`
- `70a3d3a Add continuous and active body models`

## 旧平台模型关系

以下三类模型属于 `程序/` 中的旧平台；`src/thermotaxis/` 中的 SI 规定形变身体另见上文趋温重建主线：

| 模型 | 工厂名称 | 表征方式 | 当前用途 |
| --- | --- | --- | --- |
| 旧多节段身体 | `worm2d` | 离散节点链条 | 对照模型、兼容旧训练 |
| 连续中心线身体 | `continuous_centerline` | 固定弧长采样中心线 | 身体表征对比 |
| 主动形变身体 | `active_deformation` | 中心线 + 主动传播波 | 主动形变控制研究 |

关键区别：

- `Worm2D` 是节点链式身体，不是连续身体。
- `ContinuousCenterlineBody` 是连续中心线的采样近似。
- `ActiveDeformationBody` 在连续中心线基础上加入主动波形变。

## 旧平台历史测试覆盖

旧平台历史基线的 pytest 测试数为 14 个；以下列表记录该批次覆盖范围，合并批次完整回归见“当前状态”。

覆盖范围：

- 温度场生成。
- 动态温度场更新。
- 环境尺寸和最佳点更新。
- 旧 `Worm2D` 创建。
- 旧身体单步移动和 Q 表更新。
- 2 轮短训练。
- DQN 初始化。
- 身体模型接口。
- 身体比较指标。
- `body_metrics.json` 导出。
- 连续中心线身体长度保持和曲率指标。
- 主动形变身体波参数推进。
- 标准训练使用身体模型工厂。
- 标准训练使用连续中心线身体短路径。
- 迁移学习和课程学习使用身体模型工厂。

## 结果输出

每次实验结果会按 `ExperimentConfig` 写入输出目录。

主要输出包括：

- `experiment.log`：训练日志。
- `q_table.txt`：Q 表文本输出。
- `training_results.png`：训练图表。
- `training_animation.gif` 或 `training_animation.mp4`：轨迹动画。
- `body_metrics.json`：身体模型比较指标。

其中 `body_metrics.json` 是本次改造新增的重要研究记录文件。

## 原平台扩展计划与实现状态

以下保留原平台扩展计划，并注明后续已实现的部分。该计划与 `src/thermotaxis/` 的局部测温信息契约分别使用，不能将位置、梯度或最佳点信息直接引入局部记忆控制器。

### 状态向量 `state_v2`

计划内容：

- 在旧 8 维状态 `state_v1` 之外，增加 `state_v2`。
- 包含温度梯度、位置、朝向、速度、曲率、能量、疲劳、约束违例、上一动作等信息。
- 神经网络输入维度由状态构造器返回，不再固定写 32。

### 动作空间系统

计划内容：

- 4 方向动作。
- 8 方向动作。
- 角度分箱。
- 连续朝向 + 步长。
- 身体波参数动作。

原平台已接入 CCB 相对转角与步长连续动作，ADB 保留波参数动作。这里列出的统一动作空间管理仍属于后续计划。

### 奖励函数配置化

计划内容：

- 温度收益。
- 目标保持。
- 路径效率。
- 能耗。
- 平滑性。
- 身体约束违例。
- 生存惩罚。

当前还没有统一的奖励配置文件和权重管理。

### 连续控制学习

计划内容：

- Q-Learning 只用于离散动作。
- DQN/Dueling DQN 用于离散动作和角度分箱动作。
- 连续控制使用本地 PyTorch Actor-Critic。
- 不引入外部强化学习框架。

原平台已具备本地 PyTorch Actor-Critic/DDPG 连续训练、检查点保存与冻结评估路径；目前验证结果限定于 CCB 单热源额外信息导航基线。

### 更复杂温度场泛化

计划内容：

- 单热源。
- 双热源。
- 环形热源。
- 斑点热源。
- 迷宫热源。
- 动态旋转热源。

当前项目已有多种温度场基础，但还没有完整实验矩阵、统一结果表和泛化评估脚本。

### 群体间接交互

计划内容：

- 多个线虫个体共享温度场。
- 增加信息素场。
- 信息素沉积、扩散、衰减。
- 每个个体只感知局部温度、局部信息素和邻近个体密度。
- 不使用显式通信。

当前还没有群体环境和信息素场。

### 平台实验管理

计划内容：

- 在 Streamlit 中选择身体模型。
- 在 Streamlit 中选择动作空间。
- 配置奖励权重。
- 配置群体规模。
- 配置信息素参数。
- 保存实验配置。
- 对比多个实验结果。

Streamlit 已提供身体模型与连续控制选择、奖励选项、结果保留和冻结评估展示；群体、信息素及统一实验对比管理仍属于后续计划。

### 正式实验矩阵

计划内容：

- 3 种身体模型。
- 4 类控制方式。
- 6 类温度场。
- 3 个随机种子。
- 群体规模 2、5、10。
- 信息素扩散和衰减低、中、高三档。

当前还没有自动化实验矩阵执行器。

## 当前重要限制

- 新增连续中心线身体是采样中心线模型，不是严格连续介质模型。
- 主动形变身体是波驱动近似模型，还没有真实软体接触力或流体耦合。
- 原平台不同路径使用不同观测与动作：旧离散路径保留兼容，CCB DQN 为 12 维观测并堆叠四帧，CCB Actor-Critic 为 14 维观测并输出连续动作。Streamlit 已提供身体模型和连续控制入口。
- CCB 导航评估使用同一单热源场、固定初始朝向，并关闭探索与位置噪声；观测含最佳点距离等额外信息。跨温度场及仅局部含噪测温条件下的导航泛化尚未验证。
- 点模型支持局部测温与短期记忆的机制研究，不包含身体形变和机械耗散；有限时稳定筛查不构成严格稳态证明。
- SI 身体的力学与耗散核验、冻结策略桥接不构成身体局部记忆趋温结论；身体趋温统计、弹性本构及绝对阻力标定仍待开展。
- 实验结果导出已有身体指标，但还没有统一批量分析脚本。

## 建议的下一步

当前优先在二维规定运动学场景中比较真实连接、度保持重排与普通记忆网络的学习曲线，随后交叉研究执行响应与动作限制。SI 身体的局部记忆控制与耗散比较保留为后续身体阶段，详见 [研究主线](docs/research-mainline.md)。

原平台另保留以下扩展计划；已有 Actor-Critic/DDPG 训练路径及身体模型、连续控制界面入口不再列为待实现事项：

1. 增加动作空间模块：先做 8 方向和角度分箱。
2. 增加 `state_v2`：让身体指标、速度、曲率和上一动作进入状态。
3. 完善统一奖励权重配置。
4. 实现群体与信息素场。

这样能让当前已经完成的三类身体模型真正进入统一实验矩阵。
