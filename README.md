# 软体仿生线虫研究平台

## 趋温重建主线（第 0 阶段）

组员 PR #1 已作为历史平台基线合并。新的研究主线在 `src/thermotaxis/` 中独立发展，研究局部含噪测温、短期记忆、身体运动与机械耗散怎样共同产生趋温。

第 0 阶段已经固定模型契约、SI 单位、独立的物理/传感/控制时钟、随机源和控制器观测权限。运行方式见 `docs/phase0-usage.md`，完整约定见 `docs/model-contract.md`。当前 phase-0 探针只验证可复现基础，不是趋温模拟结果。

这是一个基于 Python 的二维线虫趋温仿真平台。原始程序已经包含温度场、线虫身体、Q-Learning、DQN、Dueling DQN、迁移学习、课程学习和 Streamlit 可视化界面。当前仓库在此基础上继续改造，目标是逐步形成可用于比较不同身体表征、不同控制方式、复杂温度场和群体间接交互的研究平台。

GitHub 仓库：

- https://github.com/Zlllo/soft-worm-research-platform

## 当前状态

当前代码已经完成前三个阶段中的主要内容：

- 第一阶段：整理可运行基础，修复测试入口，增加依赖说明和 pytest 测试。
- 第二阶段：身体模型解耦，把训练引擎从直接创建 `Worm2D` 改为通过身体模型工厂创建。
- 第三阶段：新增连续中心线身体和主动形变身体的第一版实现。

当前验证结果：

- `python3 -m pytest`：14 个测试通过
- `python3 程序/test_simple.py`：通过
- `py_compile`：通过

最新代码已经推送到 GitHub 的 `main` 分支。

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

## 当前已经完成的改造内容

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
- 可见性：Private
- 分支：`main`

当前提交记录包括：

- `46f225c Establish worm simulation baseline`
- `fd4f851 Route standard training through body model factory`
- `3b4e77b Add standard training factory regression test`
- `c5f85ed Use body model factory across training engines`
- `33afcbb Export body comparison metrics`
- `70a3d3a Add continuous and active body models`

## 当前模型关系

当前有三类身体模型：

| 模型 | 工厂名称 | 表征方式 | 当前用途 |
| --- | --- | --- | --- |
| 旧多节段身体 | `worm2d` | 离散节点链条 | 对照模型、兼容旧训练 |
| 连续中心线身体 | `continuous_centerline` | 固定弧长采样中心线 | 身体表征对比 |
| 主动形变身体 | `active_deformation` | 中心线 + 主动传播波 | 主动形变控制研究 |

关键区别：

- `Worm2D` 是节点链式身体，不是连续身体。
- `ContinuousCenterlineBody` 是连续中心线的采样近似。
- `ActiveDeformationBody` 在连续中心线基础上加入主动波形变。

## 当前测试覆盖

当前 pytest 测试数：14 个。

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

## 按原计划还未完成的内容

以下内容还没有完成，后续可以按阶段继续推进。

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

当前只完成了身体模型侧对连续动作和波参数的基础接收，完整动作空间管理器还未实现。

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

当前还没有 Actor-Critic 训练路径。

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

当前训练引擎已支持 `body_model_type` 参数，但 UI 还没有完整开放这些新选项。

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
- 旧训练循环仍以旧 8 维状态和 4 动作为主。
- 新身体模型已能进入标准训练短路径，但还没有完成深度强化学习适配。
- Streamlit 页面还没有完整展示新身体模型的参数配置。
- 实验结果导出已有身体指标，但还没有统一批量分析脚本。

## 建议的下一步

建议后续按这个顺序继续：

1. 增加动作空间模块：先做 8 方向和角度分箱。
2. 增加 `state_v2`：让身体指标、速度、曲率和上一动作进入状态。
3. 将 Streamlit UI 接入身体模型选择和动作空间选择。
4. 实现奖励权重配置。
5. 实现 Actor-Critic 连续控制。
6. 实现群体与信息素场。

这样能让当前已经完成的三类身体模型真正进入统一实验矩阵。
