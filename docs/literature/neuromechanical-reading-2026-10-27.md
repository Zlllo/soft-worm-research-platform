# 神经—肌肉—身体模型：文献阅读与可复用内容

日期：2026-10-27。阅读范围为下列论文正文中与模型、接口、验证相关的方法和结果，以及所列公开源码。没有运行上游模拟器，也没有逐项复现补充材料。本文区分论文结果、当前源码观察和本项目提出的研究安排。

配套文档：[分阶段研究计划](../neuromechanical-research-plan-2026-10-27.md)。

## 1. 先回答当前问题

现有低分至少涉及回路范围、神经动态、行为出口、身体反馈和训练方法五个因素。仅增加节点，或者仅更换三维身体，都不能单独识别原因。值得复用的是已有的神经肌肉映射、可受肌肉驱动的身体、环境作用和测量方法；这些模块应先分别复现，再用于同条件比较。

“闭环”也需要说明具体含义：

1. **环境感觉闭环**：身体运动改变头部位置，头部位置改变局部刺激，刺激影响下一次神经活动。
2. **本体感觉闭环**：实际弯曲、伸长或应变返回运动回路。
3. **运动指令副本**：神经回路内部把运动指令传给感觉处理中间神经元。

三者可以分别存在。播放预存输入、输出神经活动驱动身体，只能确认前向耦合。过去动作、规定的摆头相位、实际身体弯曲也应分别记录。

## 2. 核心论文

### 2.1 c302：按研究问题选择回路范围和动态层级

Gleeson et al., 2018, *c302: a multiscale framework for modelling the nervous system of Caenorhabditis elegans*。

[论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6158223/)；[官方代码](https://github.com/openworm/c302)。重点阅读 Methods、Results 中的网络生成与 forward crawling 示例。

框架分别控制“选哪些细胞”和“采用多精细的细胞／突触模型”，支持 NeuroML 导出。C1/D1 使用连续传递突触。前行示例包含假定的周期输入，以及对本体感觉传播的功能近似；不能把该示例直接视作完整身体反馈复现。肌肉采用四列各24槽位，示例中的 MVL24 静默。

**本项目取用：** 命名与网络生成、连续突触和肌肉活动接口、从小回路到扩展回路的组织方式。先复现小型神经—肌肉响应，再核对完整数据版本。既有参数作为来源明确的起点，不称所有细胞均已实测校准。

### 2.2 Sibernetic：肌肉收缩怎样产生身体运动

Palyanov et al., 2018, *Three-dimensional simulation of the Caenorhabditis elegans body and muscle cells in liquid and gel environments for behavioural analysis*。

[论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6158221/)；[官方代码](https://github.com/openworm/sibernetic)。重点阅读 Methods 的 muscle cells、swimming/crawling 与 Discussion。

身体采用粒子方法，包含弹性外壳、内部压力与分布式收缩肌肉；通过指定激活模式研究游泳、爬行和姿态。论文的身体有96个可激活肌肉槽位，真实体壁肌肉数为95。身体运动示例本身不证明感觉反馈已经接通。官方 README 提供 c302/NEURON 接入方式，但仍有旧路径和测试模式硬编码说明。

**本项目取用：** 原始身体几何、肌肉到形变的映射、介质参数与身体测试。它适合作为独立身体参照。先测计算成本和版本兼容性，再决定是否承担大批训练；不先重写整套 SPH 求解器。

### 2.3 BAAIWorm：整合方法有用，感觉和肌肉读出需要单独处理

Zhao et al., 2024, *An integrative data-driven model simulating C. elegans brain, body and environment interactions*。

[论文](https://www.nature.com/articles/s43588-024-00738-w)；[代码](https://github.com/Jessie940611/BAAIWorm)；[论文归档代码](https://doi.org/10.5281/zenodo.13943857)。重点阅读 Fig. 3–5、Methods 的 body–environment、simulation configuration 和 reservoir configuration。

论文使用136个神经元、96肌肉槽位，四面体软体以 projective dynamics 求解，介质采用简化流体作用。80个运动神经元通过训练的线性矩阵映射到肌肉输出；这不是逐条复现解剖神经肌肉接头。其示例是化学吸引运动，感觉输入也经过工程编码。

**本项目取用：** 可受肌肉驱动的身体、在线交互组织和身体坐标测量。将它作为集成参考，而不是现成趋温控制器；更换感觉输入和神经肌肉读出时，需分别标记工程变更。其液体运动环境不能直接用于解释培养皿表面爬行。

### 2.4 Boyle–Berri–Cohen：较简洁的神经力学比较路线

Boyle, Berri & Cohen, 2012, *Gait Modulation in C. elegans: An Integrated Neuromechanical Model*。

[论文全文](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2012.00010/full)。重点阅读 Materials and Methods 的 body、muscle、neural model 与 Results 的 gait modulation。正文列出补充源码 S14，但本轮未下载运行。

模型把二维身体、肌肉和运动回路耦合，通过本体感觉与环境阻力相互作用研究前行步态变化。它提供计算成本较低的候选框架；其简化运动回路不能直接当成302节点连接组模型，也不能预设其已有完整倒退和转向能力。

**本项目取用：** 若三维求解成本过高，先评估原始补充源码的可运行性，再决定能否作为批量实验身体。保留其来源方程与参数，不把独立重建称原代码复现。

## 3. 两类反馈的实验依据

### 3.1 本体感觉

Wen et al., 2012, *Proprioceptive coupling within motor neurons drives C. elegans forward locomotion*。

[论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3508473/)。前行中，B型胆碱能运动神经元参与将前部弯曲传播到后部的本体感觉过程。

**转化成实验：** 对身体前部施加局部弯曲或改变负载，测量后部曲率的响应和延迟；比较实际形变反馈、无反馈、同统计的回放反馈。位置真值只供环境和评价器使用。

### 3.2 运动指令副本与趋温持续性

Ji et al., 2021, *Corollary discharge promotes a sustained motor state in a neural circuit for navigation*。

[论文全文](https://elifesciences.org/articles/68848)。研究涉及 RIM 依赖的运动状态信号、AIY与持续前行。其正趋温行为实验在薄层 NGM agar 上开展；Figure 7 的简化反馈模型已在本项目中独立检查过。

**转化成实验：** 允许前行／倒退后，用短暂与持续冷却区分感觉波动过滤与真正环境改变；运动指令副本和身体形变反馈分开操控。生物功能证据不能自动确定一条直接RIM→AIY突触的符号或强度。

## 4. 公开代码核查记录

以下是2026-10-27读取到的文件内容。SHA为整个文件的Git blob SHA，不是仓库提交号；实际复现时仍应固定提交及依赖。

| 文件 | blob SHA | 已核实的接口信息 |
|---|---|---|
| [BAAIWorm/worm_in_env.py](https://github.com/Jessie940611/BAAIWorm/blob/main/eworm/ghost_in_mesh_sim/worm_in_env.py) | `1d306f6f9e9b47b5f66821110abfd32d32bafddf` | 默认配置 `interaction_mode="offline"`、`proprioception=False`、`detail_control="fc"`；在线分支读取当前头部位置；默认感觉节点名单不含AFD |
| [BAAIWorm/worm_neural_network.py](https://github.com/Jessie940611/BAAIWorm/blob/main/eworm/ghost_in_mesh_sim/worm_neural_network.py) | `c2957d0072f578f165b56c04077bc697d0bd3350` | `fc`分支使用读出矩阵，并把信号裁剪到0–0.8；存在其他CNN分支，不能混同于论文主配置 |
| [Sibernetic/development README](https://github.com/openworm/sibernetic/blob/development/README.md) | `bf70971811663103f2c470e2e044d7760eccf4f4` | 提供c302和NEURON接入说明；NEURON桥测试模式的段列表有硬编码提示 |
| [c302 README](https://github.com/openworm/c302/blob/master/README.md) | `168fe9d8d363b3ad6fdf4a8567f61aba308d40be` | 提供小型神经—肌肉示例和多级模型生成入口，依赖pyNeuroML／NEURON等工具 |

BAAIWorm的感觉函数使用目标与头部距离定义局部浓度，再计算相邻时刻差分和归一化。用于趋温时应替换为环境内的头部局部温度采样；目标坐标不能进入策略。归一化初始历史全零时的分母也应专项核查。源码中的可选proprioception分支不等于已校准的实际应变反馈，需要追踪其信号来源。

BAAIWorm README公布的测试环境为 Ubuntu 20.04、NVIDIA 3090、CUDA 11.4、Python 3.8和OptiX；本机当前为macOS arm64。不能依据论文中的性能描述推断本机可直接运行，或者把显示帧率当成神经—身体完整训练的吞吐量。[环境说明](https://github.com/Jessie940611/BAAIWorm#system-requirements)

## 5. 与现有项目对接时的三个核对项

1. **95与96的映射。** 本轮读取本地 `cook2019_herm.json`，确认有473个细胞，其中体壁肌肉命名集合有95个，缺少MVL24。建立按细胞名的映射表；模型需要96槽位时，缺失槽位显式静默，不虚构解剖连接。保存两种命名空间及顺序。
2. **身体坐标与世界坐标。** 神经肌肉的背／腹、左／右与屏幕上的左右转向不是同一套坐标。先验证肌肉象限、身体姿态、头尾和传感点变换，再接温度场。
3. **不同来源数据不能默认相同。** 当前连接组固定为Cook修正版；c302参数、Sibernetic几何和BAAIWorm检查点有各自来源。复现原配置与接入当前数据应作为两个有版本记录的条件。

本轮未完成：上游环境安装、补充源码可运行性、模型单位与绝对时间校准、身体输入输出基准、独立趋温验证。它们是研究计划第一阶段的工作，不作为本次文献阅读的执行成果。
