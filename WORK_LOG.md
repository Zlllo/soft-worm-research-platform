# 工作日志 — 2026-07-24 / 2026-07-25

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
