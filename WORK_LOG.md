# 工作日志 — 2026-07-24 / 2026-08-08

## 概述

本次工作在软体仿生线虫研究平台上完成了以下内容：

1. **前端身体模型选择器** — Streamlit UI 支持三种身体模型切换
2. **Actor-Critic (DDPG) 训练模块** — 为连续身体模型新增连续动作强化学习
3. **Windows 兼容性修复** — GBK 编码崩溃 + 动画生成失败
4. **Python 3.14 + PyTorch 环境配置** — 安装依赖、验证全量测试

---

## 变更文件清单

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `程序/core/actor_critic.py` | **新增** | DDPG Actor-Critic 模块 |
| `程序/app.py` | 修改 | 身体模型选择器 + AC 训练选项 + 编码修复 |
| `程序/simulation_engine.py` | 修改 | AC 初始化集成 + 动画修复 + 编码修复 |
| `程序/core/worm_body.py` | 修改 | CCB 集成 AC + state_v2 + 编码修复 |
| `程序/core/utils.py` | 修改 | save_training_log/reset_worm 兼容 CCB/ADB |
| `程序/core/visualization.py` | 修改 | 动画输出 ffmpeg→pillow (GIF) |

---

## 详细变更

### 1. 前端：身体模型选择器 (`app.py`)

- **身体模型下拉框**：🪱 Worm2D / 📏 连续中心线 / 🌊 主动形变波
- **条件参数**：每种模型显示各自的参数滑块
  - Worm2D：节段数、节段长、头径、体宽、转向角、前进/后退/转向速度
  - ContinuousCenterlineBody：采样点数、总弧长、约束刚度、阻尼
  - ActiveDeformationBody：连续中心线参数 + 波幅、频率、传播速度、波长
- **学习算法**：根据身体模型显示可用算法
  - Worm2D：Q-Learning / DQN / Dueling DQN
  - CCB / ADB：Q-Learning / Actor-Critic
- **紧凑卡片**：训练前展示区根据模型类型显示不同参数

### 2. Actor-Critic (DDPG) 连续控制 (`core/actor_critic.py`)

- **Actor 网络**：15 维状态 → 2 维连续动作 `(heading, step_length)`
  - tanh 输出 + 可学习缩放因子
  - 隐藏层：2 × 128 ReLU
- **Critic 网络**：状态 + 动作 → Q 值
  - 双支路：状态支路 + 动作支路 → concat → 输出
- **经验回放缓冲区**：容量 50k，随机采样
- **OU 噪声**：Ornstein-Uhlenbeck 时间相关探索噪声
- **DDPGAgent**：完整训练循环
  - Critic 更新：MSE(Q(s,a), r + γ·Q'(s', μ'(s')))
  - Actor 更新：最大化 Q(s, μ(s))
  - 目标网络软更新：τ = 0.005

### 3. CCB 集成 Actor-Critic (`core/worm_body.py`)

- **`get_state_v2(env)`**：15 维连续状态向量
  - 位置 (归一化)、朝向 (cos/sin)、速度、曲率、温度/梯度、能量比
- **`setup_actor_critic()`**：初始化 DDPGAgent
- **`decide_move_actor_critic(env)`**：Actor 出动作 → 物理推进 → 奖励 → 经验回放 → 训练
- **`decide_move()` 自动分发**：AC 可用时走 AC，否则走 Q-learning

### 4. 兼容性修复

- **`utils.py` `save_training_log()`**：所有 Worm2D 专属属性用 `getattr` 防护，CCB/ADB 不再崩溃
- **`utils.py` `reset_worm_for_new_round()`**：操作 body_segments 后调用 `_sync_centerline_from_public_segments()` 同步内部状态
- **Windows GBK 编码**：`app.py`、`simulation_engine.py`、`utils.py` 顶部加 UTF-8 stdout 重配置，修复 emoji 打印崩溃
- **动画生成**：全部 `writer='ffmpeg'` 改为 `writer='pillow'` 输出 GIF，不再依赖 ffmpeg

---

## 验证结果

```
pytest:  14/14 passed
test_simple.py: 全部核心冒烟测试通过
streamlit: http://localhost:8502 正常运行
```

---

## 后续建议

1. **AC 超参数调优**：当前使用 DDPG 文献默认值，奖励函数是温度差分（非 Worm2D 的温度阶梯），建议做超参数搜索
2. **CCB 的 DQN 适配**：目前 CCB 只能用 Q-Learning + Actor-Critic，DQN/Dueling DQN 仅 Worm2D 支持
3. **角度分箱动作空间**：8/16 方向作为离散/连续之间的折中方案
4. **state_v2 扩展到 Worm2D**：让所有模型统一使用 15 维状态
5. **安装 ffmpeg**：如果想生成 MP4 视频而非 GIF，`choco install ffmpeg`（可选）

---

## 环境

- Python 3.14.0
- PyTorch 2.13.0 (CPU)
- Streamlit, Matplotlib, NumPy, Pillow
- Windows 11

---

## 2026-08-08: state_v2 统一重构

### 变更内容

将之前随意拼凑的 15 维 state_v2 替换为 **10 维统一结构**，三种身体模型（Worm2D / CCB / ADB）共用。

**新 10 维 state_v2 结构：**

| 维度 | 内容 | 来源 |
|---|---|---|
| 1-8 | 原 8 维环境状态向量 | `env.get_state_vector()` |
| 9 | 能量率 `energy / max_energy` | 身体属性 |
| 10 | 平均曲率 (归一化) | `_shape_metrics()` / 节段转角 |

堆叠 4 帧 → 40 维 NN 输入。NN 输入维度由 `worm.state_size * 4` 动态计算，不再硬编码。

### 修改文件

- `worm_body.py`: Worm2D 和 CCB 的 `get_state_v2()` 重写为 10 维；`ac_state_dim` 15→10
- `app.py`: state_v2 toggle 标签 (15→10 维)
- `simulation_engine.py`: state_v2 启用时 `state_size = 10`

### 变更理由

删掉的维度及原因：

| 删掉的维度 | 理由 |
|---|---|
| 坐标 x, y | 加坐标让 DQN 退化为"背地图"，破坏泛化 |
| 朝向 cos/sin | Worm2D 不需要；CCB 温度梯度已含方向信息 |
| 速度 vx, vy | 奖励只和温度有关，速度和奖励无因果链 |
| 长度误差 | 和温度奖励无关，物理约束强制执行 |
| 最大曲率 | 和均值高度相关，冗余 |
| 头部温度(单独) | 原 8 维已含（第 5 维） |
| 四方向梯度(单独) | 原 8 维已含（第 1-4 维） |

保留/新增维度的理由：

| 维度 | 理由 |
|---|---|
| 1-8 (原 8 维) | 经过验证的趋温任务核心状态 |
| 能量率 | AC 路径中能量出现在奖励惩罚项，有因果链 |
| 平均曲率 | AC 路径中曲率违例有惩罚；Worm2D DQN 可选关闭 |

---

## 2026-08-11: 奖励函数模块化（原版 / 含能量变体）

### 变更内容

按 README「奖励函数配置化」计划，把散落在 `worm_body.py` 里的奖励计算统一收口到新文件 **`core/reward_functions.py`**，后续只改这一个文件即可调整奖励。

**两个变体：**

| 变体 | 内容 | 说明 |
|---|---|---|
| `original`（不含能量） | Worm2D 原始温度阶梯，**逐位保留** | 出界 −10 / 120°C→5.0 阶梯 / 温差梯度 / +0.1；测试用独立参考实现模糊验证逐值相等 |
| `energy`（含能量，单步化） | 阶梯 + `−w_e·max(0, old_energy − energy) / max_energy` | 只对**当步消耗** ΔE 计费，不再混入历史时间信号 |

**两个具体修改点（按需求）：**

1. **单步化**：旧 CCB/ADB 用绝对能量 `(maxE − E)·0.001` 作惩罚，该值包含"从开局到现在的总消耗"时间信号；新方案按步计算 ΔE，只对当步消耗计费。
2. **改量级**：旧权重 0.001 相对温度奖励（0.1~5.0）可忽略；默认权重提到 **0.1**（前端滑块 0.0~0.5 可调）。

### 修改文件

- `core/reward_functions.py`：**新增**。`compute_reward(env, worm, variant, old_energy, energy_weight, stuck_penalty, constraint_penalty, target, movement)` + `apply_reward_config(worm, training_params)` + `DEFAULT_ENERGY_WEIGHT = 0.1`
- `core/worm_body.py`：删除 `Worm2D._calculate_reward` 和死代码 `calculate_reward_optimized`；三条奖励路径改调模块：
  - Worm2D `decide_move`：`compute_reward(..., target=(self.x, self.y))`，无附加惩罚
  - CCB `decide_move`（Q-learning）：统一阶梯（原为 ΔT 差分），无 stuck/constraint（保持原行为）
  - CCB `decide_move_actor_critic`：统一阶梯 + **stuck 0.5** 和 **constraint 违例率×0.5** 以参数传入（数值与触发条件原样保留，仅挪位置）
  - Worm2D/CCB `__init__` 增加 `reward_variant` / `reward_energy_weight` 默认属性
- `simulation_engine.py`：6 个线虫创建点（标准 1 / 迁移 2 / 课程 3）调用 `apply_reward_config`
- `app.py`：标准训练侧边栏新增「奖励变体」选择器（不含能量（原版）/ 含能量（单步化））+ 能量权重滑块；Worm2D 选含能量时提示"与原版逐值等价"；`training_params` 增加 `reward_variant` / `reward_energy_weight` 键
- `test_core.py`：新增 7 项测试（原版参考实现模糊测试 300 例逐值相等、能量公式数学测试、Worm2D 惰性模糊测试、stuck/constraint 测试、config 装配测试、CCB Q 路径冒烟测试）

### 行为影响

- **Worm2D**：奖励逐值不变（原版=原始；含能量变体因 ΔE≡0 与原始等价）
- **CCB / ADB**：温度部分从 ΔT 差分改为 Worm2D 阶梯（需重新基准）；能量项从绝对式改为单步式并放大权重
- AC 路径的 stuck / constraint 安全正则项原样保留

---

## 2026-08-18: 8 方向动作空间扩展（后端参数化 + 前端选择器）

### 变更内容

动作空间从"写死 4 方向"改为**参数化离散方向**（默认 4 保持历史行为，8 方向为新增，16 方向预留），前端新增动作空间选择器并与学习算法联动。

**方向编号统一：**

| | 编号 |
|---|---|
| 4 方向（历史编号，逐位不变） | 0上 1下 2左 3右 |
| n>4 方向（顺时针，公式生成） | 0上 1右上 2右 3右下 4下 5左下 6左 7左上，每 360°/n 一步 |

**关键设计决策：**

1. **对角不缩放**：8 方向对角 `(1,1)` 落在整数格点上，Q 表索引保持干净；缩放 1/√2 会引入浮点坐标截断伪影（一步对角 Q 状态不变、两步才进一格）。网格语义下对角移动快 √2 倍是合法策略，文档注明即可。
2. **Worm2D 只开 4/8**：16 方向含 8 个非格点方向，网格模型无法无损表示；16 方向留给 CCB（heading 连续，无此问题）。
3. **4 方向行为完全不变**：动作编号、落点、Q 表结构、NN 输入输出与旧版逐位一致，默认参数下零行为差异。

### 修改文件

- `core/worm_body.py`：新增 `_action_vectors()` / `_action_headings()`（4 方向历史表，n>4 公式生成）；Worm2D/CCB 的 `action_size` 读 `body_params["action_size"]`（默认 4）；两处 Q 表宽度动态化；`move()` / `step_physics` 兜底的 `moves` 动态化；`_select_action` 4 处 `random.choice([0,1,2,3])` → `random.randrange(action_size)`（修复 4-7 号动作永远探索不到的隐患）；CCB `_parse_action` headings 动态化；`Worm2D.setup_neural_network` 的 action_size 参数不再覆盖已有值
- `core/utils.py`：`setup_neural_network` 两处 `output_size=4` → `worm.action_size`；`save_q_table` 列数与标签动态（4 方向历史标签，n>4 顺时针标签，超 8 用"动作N"兜底）
- `app.py`：标准训练新增「🧭 动作空间」选择器（Worm2D: 4/8；CCB/ADB: 4/8/连续，默认 4）；算法联动过滤（离散→Worm2D: Q/DQN/Dueling、CCB/ADB: 仅 Q-Learning + "DQN/Dueling 未适配"提示；连续→仅 Actor-Critic）；`body_params` 新增 `action_size` 键；迁移/课程模式固定 4 方向
- `test_core.py`：+6 项测试（8 方向 Q 表宽/动作边界/8 落点/CCB 朝向 45°+冒烟/DQN 输出 8 维/save_q_table 8 列标签，4 方向历史编号回归），26/26 通过

### 已知瑕疵（文档注明，后续修复）

- **CCB 离散 Q 路径的能量-转向非 Markov**：Q 状态只有位置，但转向能耗依赖当前朝向（`turn_amount`）。4 方向时已存在，8 方向粒度变细影响变小；修复方案是 Q 状态加朝向，留作后续任务。
- **CCB/ADB 的 DQN/Dueling 未适配**：缺少神经网络回路（推理/回放/批训练/目标网络）与 `state_size` 组件，UI 中显示为"未适配"；本次 `output_size` 动态化后适配成本已降低，单独开一轮做。

### 验证结果

```
pytest:        26/26 passed（20 原有 + 6 新增）
test_simple.py: 全部通过
端到端冒烟:     8 方向 worm 完整训练回路（重置→决策→Q 更新）正常
```

### 后续建议

1. **16 方向 UI 选项**：CCB/ADB 打开 16（后端已支持，加个选项即可）
2. **CCB 的 DQN/Dueling 适配**：补神经网络回路 + `utils.setup_neural_network` 兼容 CCB 状态
3. **CCB Q 状态加朝向**：修复能量-转向 Markov 瑕疵
4. **state_v3 候选**：8 方向温度梯度（环境状态第 1-4 维扩到对角）

---

## 2026-08-23: 动画生成修复（matplotlib 3.9+ 兼容）

### 症状

多种动作空间 × 学习算法组合训练完成后，前端显示「⚠️ 未找到动画文件」。

### 根因（两层叠加）

1. **主因**：matplotlib 升级到 3.9+（当前环境 3.11）后 `plt.cm.get_cmap()` 被移除，`core/visualization.py` 两个动画函数（静态 `create_training_animation_2d` 第 94 行、动态 `create_training_animation_2d_dynamic_mp4` 第 297 行）一进循环就抛 `AttributeError` → `save_and_visualize_results` 捕获后走回退路径。
2. **次因**：回退函数 `create_simple_animation` 保存的文件名是 `simple_training_animation.gif`，而前端 `app.py` 的动画查找列表（`training_animation.gif/.mp4`、`worm_body_animation.gif/.mp4`）不含此名 → 即使回退成功也显示"未找到动画文件"。

说明：此故障影响**所有组合**（包括 worm2d Q-learning）；7 月 a0cef93 的"ffmpeg→pillow"修复解决的是另一个故障点（缺 ffmpeg），两者叠加不冲突。

### 修复

- `core/visualization.py`：两处 `plt.cm.get_cmap('viridis')` → `plt.cm.viridis`（色图对象直接使用，所有版本通用；pillow 保存逻辑未动）
- `app.py`：动画查找列表补充 `simple_training_animation.gif`，未来走回退路径时也能正常显示

### 验证

```
静态场动画 (CCB 历史)      → GIF 正常生成 (320KB)
动态场动画 (双中心旋转)     → GIF 正常生成 (921KB)
worm2d Q-learning 端到端   → save_and_visualize_results 输出 training_animation.gif (240KB)
pytest 26/26 + test_simple 全过
```
