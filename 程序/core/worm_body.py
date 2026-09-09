# --- File: worm_body.py (最终正确版本) ---
"""
线虫类模块 - 包含多节段线虫身体和决策逻辑
"""
import numpy as np
import math
import random
from collections import deque
import time
from abc import ABC, abstractmethod

# Actor-Critic 导入
try:
    from .actor_critic import DDPGAgent, TORCH_AVAILABLE as AC_TORCH_AVAILABLE
except ImportError:
    try:
        from actor_critic import DDPGAgent, TORCH_AVAILABLE as AC_TORCH_AVAILABLE
    except ImportError:
        DDPGAgent = None
        AC_TORCH_AVAILABLE = False

# 🔧 PyTorch导入检查
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch未安装，将仅使用Q-table方法")

# 🔧 修复神经网络导入
try:
    from neural_networks import PYTORCH_AVAILABLE
    if not TORCH_AVAILABLE:
        PYTORCH_AVAILABLE = False
    DQN_IMPORT_SUCCESS = True
except ImportError:
    try:
        from .neural_networks import PYTORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            PYTORCH_AVAILABLE = False
        DQN_IMPORT_SUCCESS = True
    except ImportError:
        print("⚠️ 无法导入神经网络模块，将使用Q-table方法")
        PYTORCH_AVAILABLE = False
        DQN_IMPORT_SUCCESS = False

# 🔧 修复utils导入
try:
    from utils import get_stacked_state
except ImportError:
    try:
        from .utils import get_stacked_state
    except ImportError:
        print("⚠️ 无法导入utils模块，使用内置方法")
        # 创建一个简单的替代函数
        def get_stacked_state(state_history, stack_size=4):
            """简单的状态堆叠函数 - 修复版"""
            try:
                if not state_history:
                    return np.zeros(8 * stack_size)
                
                # 确保 state_history 是列表或可迭代的
                if not hasattr(state_history, '__iter__'):
                    return np.zeros(8 * stack_size)
                
                state_list = list(state_history)
                
                if len(state_list) == 0:
                    return np.zeros(8 * stack_size)
                
                if len(state_list) < stack_size:
                    # 获取最后一个状态用于填充
                    last_state = state_list[-1]
                    if not hasattr(last_state, '__len__'):
                        return np.zeros(8 * stack_size)
                    
                    # 用最后一个状态填充不足的部分
                    padding = [last_state for _ in range(stack_size - len(state_list))]
                    full_list = state_list + padding
                else:
                    # 取最新的 stack_size 个状态
                    full_list = state_list[-stack_size:]
                
                # 检查每个状态的有效性
                valid_states = []
                for state in full_list:
                    if hasattr(state, '__len__') and len(state) > 0:
                        valid_states.append(np.array(state).flatten())
                    else:
                        valid_states.append(np.zeros(8))  # 默认8维状态
                
                if len(valid_states) != stack_size:
                    # 如果还是不够，用零填充
                    while len(valid_states) < stack_size:
                        valid_states.append(np.zeros(8))
                
                return np.concatenate(valid_states)
                
            except Exception as e:
                print(f"⚠️ get_stacked_state 异常: {e}")
                return np.zeros(8 * stack_size)

# 🔧 修复奖励函数模块导入
try:
    from reward_functions import compute_reward, DEFAULT_ENERGY_WEIGHT
except ImportError:
    try:
        from .reward_functions import compute_reward, DEFAULT_ENERGY_WEIGHT
    except ImportError:
        print("⚠️ 无法导入reward_functions模块（奖励计算不可用）")

        def compute_reward(*args, **kwargs):
            raise RuntimeError("reward_functions 模块导入失败，奖励计算不可用")

        DEFAULT_ENERGY_WEIGHT = 0.1

# ── 动作空间辅助: 离散方向编号统一 ─────────────────────────────
# 4 方向保持历史编号 (0上 1下 2左 3右)，与旧版逐位一致；
# n>4 方向为顺时针编号: 0上 1右上 2右 3右下 4下 5左下 6左 7左上
_LEGACY_MOVES_4 = [(0, -1), (0, 1), (-1, 0), (1, 0)]
_LEGACY_HEADINGS_4 = {0: -math.pi / 2.0, 1: math.pi / 2.0, 2: math.pi, 3: 0.0}


def _action_vectors(action_size):
    """离散动作 → 网格位移向量 (Worm2D)。8 方向对角不缩放 (网格语义，整数格点)。"""
    if action_size == 4:
        return list(_LEGACY_MOVES_4)
    return [
        (int(round(math.cos(a))), int(round(math.sin(a))))
        for i in range(action_size)
        for a in [i * 2.0 * math.pi / action_size - math.pi / 2.0]
    ]


def _action_headings(action_size):
    """离散动作 → 绝对朝向角 (CCB/ADB)。4 方向保持历史映射。"""
    if action_size == 4:
        return dict(_LEGACY_HEADINGS_4)
    return {i: i * 2.0 * math.pi / action_size - math.pi / 2.0 for i in range(action_size)}

def train_dqn_batch(worm, gamma):
    """DQN 批训练数学（Worm2D / CCB / ADB 共用）。

    采样判别用 batch_update（ExperienceReplay 也有 sample 方法，用 sample 会误判）；
    目标网络按 worm.step_count % worm.target_update_freq == 0 硬更新。
    """
    if not PYTORCH_AVAILABLE or worm.experience_replay is None:
        return
    if not getattr(worm, 'use_neural_training', True):
        return

    try:
        batch_size = min(worm.batch_size, len(worm.experience_replay))
        if batch_size < 8:  # 最小批次大小
            return

        # 采样经验：以 batch_update 区分优先回放与普通队列
        if hasattr(worm.experience_replay, 'batch_update'):
            # PrioritizedReplayBuffer
            try:
                tree_indices, experiences_data, is_weights = worm.experience_replay.sample(batch_size)
                if tree_indices is None or experiences_data is None:
                    return
            except Exception as sample_error:
                print(f"⚠️ 优先经验回放采样失败: {sample_error}")
                return
        else:
            # 普通 deque - 随机采样
            experiences_data = random.sample(list(worm.experience_replay), batch_size)
            is_weights = np.ones(batch_size)  # 等权重
            tree_indices = None

        # 提取批次数据
        states = np.vstack([e[0] for e in experiences_data])
        actions = np.array([e[1] for e in experiences_data])
        rewards = np.array([e[2] for e in experiences_data])
        next_states = np.vstack([e[3] for e in experiences_data])
        dones = np.array([e[4] for e in experiences_data])

        # 转换为张量
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.BoolTensor(dones)
        is_weights_tensor = torch.FloatTensor(is_weights)

        # 计算当前Q值
        current_q_values = worm.neural_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))

        # 计算目标Q值 (Double DQN: 在线网选动作、目标网估值)
        with torch.no_grad():
            if hasattr(worm, 'target_network') and worm.target_network is not None:
                next_actions = worm.neural_network(next_states_tensor).argmax(1)
                next_q_values_target = worm.target_network(next_states_tensor)
                next_max_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1))
            else:
                next_max_q_values = worm.neural_network(next_states_tensor).max(1)[0].unsqueeze(1)

        next_max_q_values[dones_tensor.unsqueeze(1)] = 0.0
        target_q_values = rewards_tensor.unsqueeze(1) + gamma * next_max_q_values

        # 计算损失
        td_errors = torch.abs(target_q_values - current_q_values).detach()
        loss = torch.mean(is_weights_tensor.unsqueeze(1) * torch.nn.functional.mse_loss(
            current_q_values, target_q_values, reduction='none'))

        # 反向传播
        worm.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(worm.neural_network.parameters(), max_norm=1.0)
        worm.optimizer.step()

        # 更新优先级
        if tree_indices is not None and hasattr(worm.experience_replay, 'batch_update'):
            try:
                td_errors_numpy = td_errors.squeeze().cpu().numpy()
                if td_errors_numpy.ndim > 1:
                    td_errors_numpy = td_errors_numpy.flatten()
                td_errors_numpy = td_errors_numpy.astype(np.float32)
                worm.experience_replay.batch_update(tree_indices, td_errors_numpy)
            except Exception as update_error:
                print(f"⚠️ 优先级更新失败: {update_error}")

        # 更新目标网络
        if hasattr(worm, 'target_network') and hasattr(worm, 'target_update_freq'):
            if worm.step_count % worm.target_update_freq == 0:
                worm.target_network.load_state_dict(worm.neural_network.state_dict())

    except Exception as e:
        print(f"⚠️ 神经网络批训练失败: {e}")


# 🔧 修复DQN导入
try:
    if PYTORCH_AVAILABLE:
        from neural_networks import DQNAgent, setup_neural_network, create_dqn_agent
        print("✓ DQN模块导入成功")
        DQN_IMPORT_SUCCESS = True
except ImportError:
    try:
        if PYTORCH_AVAILABLE:
            from .neural_networks import DQNAgent, setup_neural_network, create_dqn_agent
            print("✓ DQN模块导入成功")
            DQN_IMPORT_SUCCESS = True
    except ImportError:
        print("⚠️ 无法导入DQN相关模块")
        PYTORCH_AVAILABLE = False
        DQN_IMPORT_SUCCESS = False
        # 创建占位类
        class DQNAgent:
            def __init__(self, *args, **kwargs): 
                self.q_network = None
            def act(self, state): return 0
            def remember(self, *args): pass
            def replay(self): pass
        
        def setup_neural_network(*args, **kwargs): 
            return None, None
        
        def create_dqn_agent(*args, **kwargs): 
            return None


# 🔧 添加缺失的导入
try:
    import torch  # 需要为神经网络功能导入torch
except ImportError:
    torch = None
    print("⚠️ PyTorch未安装，将仅使用Q-table方法")

class BodyModel(ABC):
    """身体模型统一接口。

    后续不同身体表征只需要实现这组方法，训练和环境代码就可以通过
    观测、动作、物理推进、几何和指标访问模型状态。
    """

    @abstractmethod
    def reset(self, start_pos=None, **kwargs):
        """重置身体状态。"""

    @abstractmethod
    def get_observation(self, env=None, **kwargs):
        """返回策略可用的观测数据。"""

    @abstractmethod
    def apply_action(self, action, **kwargs):
        """接收并缓存控制动作。"""

    @abstractmethod
    def step_physics(self, env=None, dt=1.0, **kwargs):
        """推进身体物理状态。"""

    @abstractmethod
    def get_geometry(self):
        """返回可视化和碰撞检测所需的几何数据。"""

    @abstractmethod
    def get_metrics(self, env=None):
        """返回训练统计和调试指标。"""


class Worm2D(BodyModel):
    def __init__(self, start_pos, width, height, body_params={}, noise_params={}):  # <-- 修改点 1
        self.x, self.y = start_pos
        self.width = width
        self.height = height

        # --- 🔧 关键修复：使用从UI传入的身体和噪声参数 ---
        print(f"🔧 线虫初始化 - 接收到身体参数: {body_params}")
        print(f"🔧 线虫初始化 - 接收到噪声参数: {noise_params}")
        
        # 身体形态参数 (从UI滑块获取)
        self.num_segments = body_params.get("num_segments", 6)  # <-- 修改点：与8.22保持一致为6个节段
        segment_length = body_params.get("segment_length", 5.0)
        self.head_radius = body_params.get("head_radius", 4.0)
        self.body_width = body_params.get("body_width", 2.5)
        self.max_turn_angle = body_params.get("max_turn_angle", 30)
        self.forward_speed = body_params.get("forward_speed", 2.0)
        self.backward_speed = body_params.get("backward_speed", 1.0)
        self.turning_speed = body_params.get("turning_speed", 1.5)
        
        # 计算节段距离
        self.segment_distance = segment_length / max(1, self.num_segments - 1) if self.num_segments > 1 else segment_length
        self.body_length = segment_length
        
        # 布朗噪声参数 (从UI滑块获取)
        self.position_noise = noise_params.get("position_noise", 0.1)
        self.angle_noise = noise_params.get("angle_noise", 2.0)
        self.thermal_noise = noise_params.get("thermal_noise", 0.5)
        self.action_noise = noise_params.get("action_noise", 0.05)
        self.noise_correlation = noise_params.get("noise_correlation", 0.3)
        
        # 噪声状态变量 (用于时间相关噪声)
        self.prev_position_noise = [0.0, 0.0]
        self.prev_angle_noise = 0.0
        
        print(f"🔧 应用身体参数 - 节段数: {self.num_segments}, 节段长度: {segment_length}")
        print(f"🔧 应用身体参数 - 头径: {self.head_radius}, 体宽: {self.body_width}")
        print(f"🔧 应用运动参数 - 转向角: {self.max_turn_angle}°, 前进速度: {self.forward_speed}")
        print(f"🔧 应用噪声参数 - 位置噪声: {self.position_noise}, 角度噪声: {self.angle_noise}")
        print(f"🔧 应用噪声参数 - 温度噪声: {self.thermal_noise}, 相关性: {self.noise_correlation}")
        
        # 🔧 【第一步修正版】完全重新初始化身体段和温度感知
        self.body_segments = []
        self.body_temperatures = []

        x, y = start_pos
        print(f"🔧 初始化{self.num_segments}个身体段，起始位置: ({x}, {y})")

        # 创建身体段，沿x轴向左排列
        for i in range(self.num_segments):
            segment_x = max(0, min(width-1, x - i * self.segment_distance))  # 使用实际的segment_distance
            segment_y = y
            self.body_segments.append([segment_x, segment_y])
            self.body_temperatures.append(0.0)
            print(f"   身体段 {i}: 位置({segment_x:.1f}, {segment_y:.1f})")

        print(f"🔧 身体温度感知系统初始化：{len(self.body_temperatures)}个身体段")
        print(f"🔧 完整身体位置: {self.body_segments}")
        
        self.history = [self.body_segments.copy()]  # 保存完整身体段列表，而不是单个点
        print(f"🔧 历史记录初始化：保存完整身体段 {len(self.body_segments)} 个")
        
        self.total_reward = 0
        self.action_size = int(body_params.get("action_size", 4))  # 离散方向数: 4/8/16 (默认4保持兼容)
        self.q_table = [[[0.0] * self.action_size for _ in range(width)] for _ in range(height)]
        self.use_neural = False
        self.neural_network = None
        self.optimizer = None
        self.state_size = 8  # 单帧维度 (默认旧 8 维; state_v2 时改为 15)
        self.use_state_v2 = False  # 默认关闭，前端可开启
        self._pending_action = None
        self.last_action = None
        self.last_physics_result = {
            'moved': False,
            'action': None,
            'position': (self.x, self.y)
        }
        self.experience_replay = None
        self.batch_size = 64 # ⭐️ 建议增加 batch_size 以获得更稳定的梯度
        self.train_interval = 4  # 🔧 训练间隔调整为4
        self.step_count = 0
        self.recent_temperatures = []
        self.memory_size = 8
        self.visited_positions = {}
        self.position_memory_size = 30  # 改为30，匹配8.22版本
        self.max_energy = 100.0
        self.energy = self.max_energy
        self.energy_decay_rate = 0.5
        self.low_energy_threshold = 50.0
        self.reward_variant = "original"      # original / energy（reward_functions 模块统一计算）
        self.reward_energy_weight = DEFAULT_ENERGY_WEIGHT
        self.body_flexibility = 0.8
        self.cuticle_stiffness = 0.7
        self.max_bend_angle = 50
        self.hydrostatic_pressure = 0.3
        self.pressure_recovery_rate = 0.2
        self.muscle_wave_phase = 0.0
        self.dorsal_muscle_state = 0.0
        self.ventral_muscle_state = 0.0
        self.muscle_wave_amplitude = 0.5
        self.muscle_wave_frequency = 0.3
        self.muscle_coordination = 0.8
        self.base_muscle_wave_frequency = 0.3
        self.frequency_adaptation_rate = 0.1
        self.movement_frequency_boost = 0.5
        self.energy_frequency_factor = 0.8
        self.muscle_fatigue_level = 0.0
        self.fatigue_accumulation_rate = 0.015
        self.fatigue_recovery_rate = 0.008
        self.max_fatigue_penalty = 0.6
        self.fatigue_threshold = 0.3
        self.temperature_muscle_sensitivity = 0.8
        self.optimal_muscle_temperature = 75.0
        self.muscle_temp_adaptation_rate = 0.1
        self.current_temp = 25.0
        self.segment_elasticity = 0.85
        self.max_segment_stretch = 1.5
        self.min_segment_compression = 0.7
        self.angular_constraint = 45.0
        self.elastic_recovery_rate = 0.12
        self.segment_tensions = [0.0] * self.num_segments
        self.state_buffer = deque(maxlen=4)
        self._prev_head_pos = (self.x, self.y)  # 追踪头部速度

        # 🎯 【恢复奖励处理机制】保持优化的同时恢复必要的奖励处理
        self.reward_buffer = deque(maxlen=3000)  # 奖励缓冲区，用于统计和归一化
        self.reward_normalization_enabled = True  # 启用Z-score归一化
        self.reward_clipping_enabled = True  # 启用奖励裁剪
        self.reward_clip_range = (-5.0, 5.0)  # 裁剪范围
        self.min_buffer_size = 200  # 归一化启动的最小样本数
        
        print(f"🎯 混合优化的线虫智能体已初始化（简化计算+智能奖励处理）")
        print(f"   奖励缓冲区大小: {self.reward_buffer.maxlen}")
        print(f"   Z-score归一化: {'启用' if self.reward_normalization_enabled else '禁用'}")
        print(f"   奖励裁剪: {'启用' if self.reward_clipping_enabled else '禁用'}")
        print(f"   裁剪范围: {self.reward_clip_range}")

        # 🔧 【修复】添加缺失的步数计数器
        self.current_step = 0  # 当前步数计数器，用于调试信息控制
        
        # 🔧 【修复】添加缺失的感知系统属性
        self.perception_range = 6  # 感知范围
        self.hotspot_memory = {}   # 热点记忆字典
        self.memory_decay_steps = 200  # 记忆衰减步数
        self.perception_memory = deque(maxlen=10)  # 感知记忆缓冲区
        
        # 🔧 【新增】决策统计，用于监控学习和本能的平衡
        self.decision_stats = {
            'learning': 0,      # Q学习/神经网络决策次数
            'memory': 0,        # 记忆导向决策次数
            'perception': 0,    # 感知导向决策次数
            'gradient': 0,      # 梯度导向决策次数
            'total': 0          # 总决策次数
        }
        
        print(f"   感知范围: {self.perception_range}, 记忆衰减: {self.memory_decay_steps}步")

    def decide_move(self, env, epsilon=0.2, alpha=0.5, gamma=0.9):
        """
        决策并移动：实现撞墙反弹机制 - 添加超时保护
        """
        import time
        start_time = time.time()
        
        try:
            # 🔧 正确的步数管理
            self.current_step += 1
            
            # 🔧 添加更频繁的超时检查
            def check_timeout(stage_name, max_time=3.0):
                elapsed = time.time() - start_time
                if elapsed > max_time:
                    print(f"⚠️ decide_move {stage_name} 超时: {elapsed:.3f}s")
                    return True
                return False
            
            # 🔧 调试点1：感知环境 - 添加超时保护
            perception_start = time.time()
            perception_data = self.perceive_environment(env)
            perception_time = time.time() - perception_start
            
            if perception_time > 1.0:
                print(f"⚠️ 感知环境耗时: {perception_time:.3f}s")
            
            if check_timeout("感知环境"):
                return False
            
            # 🔧 调试点2：更新身体温度 - 添加超时保护
            temp_start = time.time()
            self.update_body_temperatures(env)
            temp_time = time.time() - temp_start
            
            if temp_time > 0.5:
                print(f"⚠️ 温度更新耗时: {temp_time:.3f}s")
            
            if check_timeout("身体温度更新"):
                return False
            
            # 🔧 调试点3：状态处理 - 添加超时保护
            state_start = time.time()
            current_pos = (self.x, self.y)
            # 使用统一 state_v2 或旧版 8 维状态
            if getattr(self, 'use_state_v2', False) and self.use_neural:
                current_state = self.get_state_v2(env)
            else:
                current_state = env.get_state_vector(current_pos, worm=self)

            # 检查状态有效性
            if current_state is None or len(current_state) == 0:
                default_dim = 10 if getattr(self, 'use_state_v2', False) else 8
                print(f"⚠️ 获取到无效状态，使用默认状态")
                current_state = np.zeros(default_dim, dtype=np.float32)

            self.state_buffer.append(current_state.astype(np.float32))
            state_time = time.time() - start_time

            if state_time > 0.2:
                print(f"⚠️ 状态处理耗时: {state_time:.3f}s")
            
            if check_timeout("状态处理"):
                return False
            
            # 🔧 调试点4：获取堆叠状态 - 添加超时保护
            stacked_start = time.time()
            try:
                stacked_state = get_stacked_state(self.state_buffer)
                if stacked_state is None or len(stacked_state) == 0:
                    default_dim = 40 if getattr(self, 'use_state_v2', False) else 32
                    print(f"⚠️ 堆叠状态无效，使用默认状态")
                    stacked_state = np.zeros(default_dim, dtype=np.float32)
            except Exception as e:
                print(f"❌ 堆叠状态处理失败: {e}")
                stacked_state = np.zeros(32)
            
            stacked_time = time.time() - stacked_start
            
            if stacked_time > 0.2:
                print(f"⚠️ 状态堆叠耗时: {stacked_time:.3f}s")
            
            if check_timeout("状态堆叠"):
                return False
            
            # 🔧 调试点5：动作选择 - 添加超时保护
            action_start = time.time()
            
            # 保存旧状态用于Q-learning
            old_x, old_y = int(self.x), int(self.y)
            
            # 🔧 限制epsilon调整的计算复杂度
            adjusted_epsilon = max(epsilon * 0.5, epsilon * (1.0 - self.current_step / 10000.0))
            
            # 动作选择 - 简化版本，使用32维的stacked_state
            action = self._select_action(stacked_state, adjusted_epsilon, old_x, old_y)
            
            action_time = time.time() - action_start
            if action_time > 0.5:
                print(f"⚠️ 动作选择耗时: {action_time:.3f}s")
            
            if check_timeout("动作选择"):
                return False
            
            # 🔧 调试点6：执行移动 - 添加超时保护
            move_start = time.time()
            
            # 保存移动前的位置和身体段
            old_position = (self.x, self.y)
            old_body_segments = [seg.copy() for seg in self.body_segments]
            
            # 执行移动
            move_successful = self.move(action, env)
            
            move_time = time.time() - move_start
            if move_time > 0.3:
                print(f"⚠️ 移动执行耗时: {move_time:.3f}s")
            
            if check_timeout("移动执行"):
                return False
            
            # 🔧 调试点7：奖励计算 - 添加超时保护
            reward_start = time.time()
            
            # 🔧 修复：获取新状态并构造堆叠状态用于神经网络
            new_pos = (self.x, self.y)
            if getattr(self, 'use_state_v2', False) and self.use_neural:
                new_state_frame = self.get_state_v2(env)
            else:
                new_state_frame = env.get_state_vector(new_pos, worm=self)
            if new_state_frame is None:
                default_dim = 10 if getattr(self, 'use_state_v2', False) else 8
                new_state_frame = np.zeros(default_dim, dtype=np.float32)

            # 为神经网络构造新堆叠状态
            temp_buffer = self.state_buffer.copy()
            temp_buffer.append(new_state_frame)
            new_stacked_state = get_stacked_state(temp_buffer)
            
            # 🔧 统一奖励函数模块（原版 = Worm2D 原始阶梯，逐位保留）
            try:
                reward = compute_reward(
                    env, self,
                    variant=getattr(self, 'reward_variant', 'original'),
                    old_energy=self.energy,
                    energy_weight=getattr(self, 'reward_energy_weight', DEFAULT_ENERGY_WEIGHT),
                    target=(self.x, self.y),
                )
            except Exception as reward_error:
                print(f"⚠️ 奖励计算失败: {reward_error}")
                reward = 0.0  # 使用默认奖励
            
            reward_time = time.time() - reward_start
            if reward_time > 0.5:
                print(f"⚠️ 奖励计算耗时: {reward_time:.3f}s")
            
            if check_timeout("奖励计算"):
                return False
            
            # 🔧 调试点8：学习更新 - 添加超时保护
            learning_start = time.time()
            
            # Q-learning 更新
            if not self.use_neural or not hasattr(self, 'neural_network') or self.neural_network is None:
                try:
                    # 确保坐标在有效范围内
                    if 0 <= old_y < len(self.q_table) and 0 <= old_x < len(self.q_table[0]):
                        old_q = self.q_table[old_y][old_x][action]
                        new_y, new_x = int(self.y), int(self.x)
                        if 0 <= new_y < len(self.q_table) and 0 <= new_x < len(self.q_table[0]):
                            max_next_q = max(self.q_table[new_y][new_x])
                            self.q_table[old_y][old_x][action] = old_q + alpha * (reward + gamma * max_next_q - old_q)
                except Exception as q_error:
                    print(f"⚠️ Q-learning更新失败: {q_error}")
            
            # 神经网络更新
            if self.use_neural and hasattr(self, 'experience_replay') and self.experience_replay is not None:
                try:
                    # 🔧 修复：使用当前作用域中的32维堆叠状态存储经验
                    current_stacked_state = stacked_state  # 当前的32维堆叠状态
                    next_stacked_state = new_stacked_state  # 新的32维堆叠状态
                    experience = (current_stacked_state.copy(), action, reward, next_stacked_state.copy(), False)
                    if hasattr(self.experience_replay, 'add'):
                        # PrioritizedReplayBuffer 使用 add 方法
                        self.experience_replay.add(experience)
                    elif hasattr(self.experience_replay, 'append'):
                        # 普通 deque 使用 append 方法
                        self.experience_replay.append(experience)
                    
                    # 限制训练频率，避免过度计算
                    if self.current_step % 10 == 0 and len(self.experience_replay) >= 32:
                        self._train_neural_network_batch(gamma)
                except Exception as nn_error:
                    print(f"⚠️ 神经网络更新失败: {nn_error}")
            
            learning_time = time.time() - learning_start
            if learning_time > 0.5:
                print(f"⚠️ 学习更新耗时: {learning_time:.3f}s")
            
            if check_timeout("学习更新"):
                return False
            
            # 🔧 调试点9：记录和清理 - 添加超时保护
            cleanup_start = time.time()
            
            # 更新总奖励
            self.total_reward += reward
            
            # 历史记录在_update_body_physics中统一更新，避免重复
            # 不在这里更新history，保持与8.22版本一致的数据格式
            
            # 限制历史记录长度，避免内存问题
            if hasattr(self, 'history') and len(self.history) > 1000:
                self.history = self.history[-500:]  # 保留最新的500步
            
            cleanup_time = time.time() - cleanup_start
            if cleanup_time > 0.2:
                print(f"⚠️ 记录清理耗时: {cleanup_time:.3f}s")
            
            # 🔧 最终超时检查
            total_time = time.time() - start_time
            if total_time > 5.0:
                print(f"⚠️ decide_move 总耗时过长: {total_time:.3f}s")
            
            return move_successful
            
        except Exception as e:
            print(f"❌ decide_move 异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    # 🔧 修复：将 perceive_environment 正确定义为类方法
    def perceive_environment(self, env):
        """
        感知周围环境 - 修复无限循环问题
        """
        import time
        start_time = time.time()
        
        try:
            head_x, head_y = self.body_segments[0]
            head_x, head_y = int(head_x), int(head_y)
            
            # 🔧 修复1：移除有问题的步数递增
            # self.current_step += 1  # ❌ 删除这行，步数应该在decide_move中管理
            
            perception_data = {
                'center_temp': env.get_temperature(head_x, head_y),
                'nearby_temps': [],
                'max_nearby_temp': -float('inf'),
                'best_direction': None,
                'perception_range': self.perception_range,
                'memory_hotspots': [],
                'memory_suggested_direction': None,
                'temperature_gradients': {},
                'gradient_direction': None,
                'gradient_confidence': 0.0,
                'perception_quality': 'normal'
            }
            
            center_temp = perception_data['center_temp']
            directions = {
                0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)
            }
            
            total_valid_gradients = 0
            gradient_sum = 0.0
            
            # 🔧 修复2：梯度计算添加更严格的控制
            gradient_timeout = 0.5  # 梯度计算最大时间
            gradient_start = time.time()
            
            for direction_idx, (dx, dy) in directions.items():
                # 🔧 更严格的超时检查
                if time.time() - gradient_start > gradient_timeout:
                    print(f"⚠️ 梯度计算超时，已处理方向: {direction_idx}")
                    break
                    
                gradient_samples = []
                max_steps = min(self.perception_range, 3)  # 🔧 进一步限制为3步
                
                for step in range(1, max_steps + 1):
                    check_x = head_x + dx * step
                    check_y = head_y + dy * step
                    
                    if 0 <= check_x < self.width and 0 <= check_y < self.height:
                        temp = env.get_temperature(check_x, check_y)
                        distance_weight = 1.0 / (step * 0.5 + 0.5)
                        weighted_gradient = (temp - center_temp) * distance_weight / step
                        gradient_samples.append(weighted_gradient)
                    else:
                        break
                
                if gradient_samples:
                    weights = [1.0 / (i + 1) for i in range(len(gradient_samples))]
                    total_weight = sum(weights)
                    
                    if total_weight > 0:
                        avg_gradient = sum(g * w for g, w in zip(gradient_samples, weights)) / total_weight
                        perception_data['temperature_gradients'][direction_idx] = avg_gradient
                        
                        if abs(avg_gradient) > 0.1:
                            total_valid_gradients += 1
                            gradient_sum += abs(avg_gradient)
            
            # 🔧 修复3：环境扫描添加更严格的控制
            scan_timeout = 1.0  # 扫描最大时间
            scan_start = time.time()
            
            # 🔧 限制扫描范围
            limited_range = min(self.perception_range, 4)  # 最大4格
            scan_count = 0
            max_scan_points = limited_range * limited_range * 4  # 预计算最大点数
            
            for dx in range(-limited_range, limited_range + 1):
                # 🔧 外层循环超时检查
                if time.time() - scan_start > scan_timeout:
                    print(f"⚠️ 环境扫描外层超时，已扫描: {scan_count} 点")
                    break
                    
                for dy in range(-limited_range, limited_range + 1):
                    if dx == 0 and dy == 0:
                        continue
                    
                    scan_count += 1
                    
                    # 🔧 更频繁的检查
                    if scan_count > max_scan_points:
                        # print(f"⚠️ 扫描点数达到限制: {max_scan_points}")
                        break
                    
                    # 🔧 每10个点检查一次时间
                    if scan_count % 10 == 0 and time.time() - scan_start > scan_timeout:
                        print(f"⚠️ 环境扫描内层超时，已扫描: {scan_count} 点")
                        break
                    
                    scan_x = head_x + dx
                scan_y = head_y + dy
                
                if 0 <= scan_x < self.width and 0 <= scan_y < self.height:
                    temp = env.get_temperature(scan_x, scan_y)
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance <= limited_range:
                        perception_data['nearby_temps'].append({
                            'position': (scan_x, scan_y),
                            'temperature': temp,
                            'distance': distance
                        })
                        
                        if temp > 75.0:
                            self.hotspot_memory[(scan_x, scan_y)] = {
                                'temp': temp,
                                'last_seen': getattr(self, 'current_step', 0)  # 🔧 安全获取步数
                            }
                        
                        if temp > perception_data['max_nearby_temp']:
                            perception_data['max_nearby_temp'] = temp
                            if abs(dx) > abs(dy):
                                perception_data['best_direction'] = 3 if dx > 0 else 2
                            else:
                                perception_data['best_direction'] = 1 if dy > 0 else 0
                
                # 🔧 外层循环也检查内层的退出条件
                if scan_count > max_scan_points or time.time() - scan_start > scan_timeout:
                    break
            
            # 🔧 修复4：记忆系统简化
            # 简化记忆清理，避免复杂计算
            if hasattr(self, 'hotspot_memory') and self.hotspot_memory:
                # 只保留最近的记忆项，避免无限循环
                if len(self.hotspot_memory) > 50:  # 限制记忆项数量
                    # 简单删除最老的一半
                    items = list(self.hotspot_memory.items())
                    items.sort(key=lambda x: x[1].get('last_seen', 0))
                    for pos, _ in items[:25]:  # 删除最老的25个
                        del self.hotspot_memory[pos]
                
                # 简化的记忆导向决策
                best_memory_temp = -float('inf')
                best_memory_pos = None
                
                for pos, info in list(self.hotspot_memory.items())[:20]:  # 只检查前20个
                    if info['temp'] > best_memory_temp:
                        best_memory_temp = info['temp']
                        best_memory_pos = pos
                
                if best_memory_pos and best_memory_temp > perception_data['max_nearby_temp']:
                    dx = best_memory_pos[0] - head_x
                    dy = best_memory_pos[1] - head_y
                    
                    if abs(dx) > abs(dy):
                        perception_data['memory_suggested_direction'] = 3 if dx > 0 else 2
                    else:
                        perception_data['memory_suggested_direction'] = 1 if dy > 0 else 0
                    
                    perception_data['memory_hotspots'].append({
                        'position': best_memory_pos,
                        'temperature': best_memory_temp
                    })
            
            # 🔧 修复5：梯度方向选择简化
            if perception_data['temperature_gradients']:
                best_gradient = -float('inf')
                best_gradient_direction = None
                
                for direction_idx, gradient in perception_data['temperature_gradients'].items():
                    if gradient > 0.3 and gradient > best_gradient:
                        best_gradient = gradient
                        best_gradient_direction = direction_idx
                
                perception_data['gradient_direction'] = best_gradient_direction
                
                if total_valid_gradients > 0:
                    avg_gradient_strength = gradient_sum / total_valid_gradients
                    perception_data['gradient_confidence'] = min(1.0, avg_gradient_strength * 2.0)
            
            # 🔧 安全地添加到感知记忆
            if hasattr(self, 'perception_memory'):
                self.perception_memory.append(perception_data.copy())
            
            # 🔧 调试信息已删除
            
            return perception_data
            
        except Exception as e:
            print(f"❌ perceive_environment 异常: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回安全的默认数据
            return {
                'center_temp': 25.0,
                'nearby_temps': [],
                'max_nearby_temp': -float('inf'),
                'best_direction': None,
                'perception_range': 3,  # 🔧 更安全的默认值
                'memory_hotspots': [],
                'memory_suggested_direction': None,
                'temperature_gradients': {},
                'gradient_direction': None,
                'gradient_confidence': 0.0,
                'perception_quality': 'poor'
            }

    # 🔧 【关键修复】添加缺失的方法
    def _adjust_exploration(self, epsilon):
        """调整探索率"""
        adjusted_epsilon = epsilon
        
        # 低能量时更保守
        if self.energy < self.low_energy_threshold:
            adjusted_epsilon = epsilon * 0.7  # 适度减少探索
        
        if len(self.recent_temperatures) >= 3:
            recent_temp = self.recent_temperatures[-1]
            earlier_temp = self.recent_temperatures[-3]
            temp_trend = recent_temp - earlier_temp
            
            if temp_trend > 1.0:
                adjusted_epsilon = adjusted_epsilon * 0.8
            elif temp_trend < -1.0:
                adjusted_epsilon = adjusted_epsilon * 1.2
        
        return adjusted_epsilon
    
    def update_body_temperatures(self, env):
        """
        🔧 【第二步】更新身体各段的温度感知
        """
        # 确保 body_temperatures 列表与 body_segments 长度一致
        while len(self.body_temperatures) < len(self.body_segments):
            self.body_temperatures.append(0.0)
        
        # 更新每个身体段的温度
        for i, (x, y) in enumerate(self.body_segments):
            # 获取该身体段位置的环境温度
            segment_temp = env.get_temperature(int(x), int(y))
            
            # 更新身体段温度（处理边界情况）
            if segment_temp != -float('inf'):
                self.body_temperatures[i] = segment_temp
            else:
                # 如果位置无效，保持上一次的温度
                if i < len(self.body_temperatures):
                    pass  # 保持原温度
                else:
                    self.body_temperatures.append(0.0)

    def get_state_v2(self, env=None):
        """返回统一 10 维状态向量：原 8 维 + 能量率 + 平均曲率。

        所有身体模型共用同一结构，4 帧堆叠 → 40 维 NN 输入。
        """
        state = []

        # 1-8. 复用环境 8 维状态向量(四方向梯度 + 温度 + 对齐 + 趋势 + 距离)
        if env is not None:
            base = env.get_state_vector((self.x, self.y), worm=self)
        else:
            base = np.zeros(8, dtype=np.float32)
        state.extend(base.astype(np.float32).tolist())

        # 9. 能量率
        state.append(float(self.energy / max(self.max_energy, 1.0)))

        # 10. 平均曲率(归一化)
        curv_limit = max(getattr(self, 'angular_constraint', 45.0), 1.0)
        mean_curv = 0.0
        points = [np.array([float(x), float(y)]) for x, y in self.body_segments]
        if len(points) >= 3:
            turn_angles = []
            for i in range(1, len(points) - 1):
                pv = points[i] - points[i-1]
                nv = points[i+1] - points[i]
                pn = float(np.linalg.norm(pv))
                nn = float(np.linalg.norm(nv))
                if pn > 1e-9 and nn > 1e-9:
                    cos_a = float(np.dot(pv, nv) / (pn * nn))
                    turn_angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cos_a)))))
            mean_curv = np.mean(turn_angles) if turn_angles else 0.0
        state.append(float(np.clip(mean_curv / curv_limit, 0.0, 1.0)))

        return np.array(state[:10], dtype=np.float32)

    def _select_action(self, current_state, adjusted_epsilon, old_x, old_y):
        """选择动作"""
        if self.use_neural and self.neural_network is not None and PYTORCH_AVAILABLE and torch is not None:
            if random.random() < adjusted_epsilon:
                action = random.randrange(self.action_size)
            else:
                try:
                    with torch.no_grad():
                        frame_dim = getattr(self, 'state_size', 8)
                        expected_dim = frame_dim * 4  # 堆叠 4 帧
                        if len(current_state) != expected_dim:
                            print(f"⚠️ 状态维度错误: {len(current_state)}, 期望{expected_dim}维，使用随机动作")
                            action = random.randrange(self.action_size)
                        else:
                            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
                            q_values = self.neural_network(state_tensor)
                            action = q_values.argmax().item()
                except Exception as e:
                    print(f"⚠️ 神经网络推理失败，使用随机动作: {e}")
                    action = random.randrange(self.action_size)
        else:
            if random.random() < adjusted_epsilon:
                action = random.randrange(self.action_size)
            else:
                # 使用旧状态坐标来查询Q表
                q_values = self.q_table[old_y][old_x]
                max_q = max(q_values)
                actions = [i for i, q in enumerate(q_values) if q == max_q]
                action = random.choice(actions)
        
        return action

    def _update_learning(self, current_stacked_state, action, reward, new_head_x, new_head_y, env, old_x, old_y, alpha, gamma):
        """
        修复版学习更新 - 解决 PrioritizedReplayBuffer 错误
        """
        if alpha == 0.0:
            return

        # 原有的奖励处理逻辑保持不变
        raw_reward = reward
        is_terminal_reward = (raw_reward == -10.0)
        
        if not is_terminal_reward:
            self.reward_buffer.append(raw_reward)
        
        if is_terminal_reward:
            processed_reward = -1.0
        else:
            if self.reward_clipping_enabled:
                processed_reward = np.clip(raw_reward, self.reward_clip_range[0], self.reward_clip_range[1])
            elif (self.reward_normalization_enabled and len(self.reward_buffer) > self.min_buffer_size):
                recent_buffer = list(self.reward_buffer)[-1000:]
                mean_reward = np.mean(recent_buffer)
                std_reward = np.std(recent_buffer) + 1e-8
                processed_reward = (raw_reward - mean_reward) / std_reward
                processed_reward = np.clip(processed_reward, -3.0, 3.0)
            else:
                processed_reward = raw_reward

        # 构造 next_stacked_state
        if getattr(self, 'use_state_v2', False) and self.use_neural:
            next_frame = self.get_state_v2(env)
        else:
            next_frame = env.get_state_vector((new_head_x, new_head_y), worm=self)
        temp_buffer = self.state_buffer.copy()
        temp_buffer.append(next_frame)
        next_stacked_state = get_stacked_state(temp_buffer)

        done = is_terminal_reward

        # 🔧 修复神经网络更新部分
        if self.use_neural and self.neural_network is not None and self.experience_replay is not None and getattr(self, 'use_neural_training', True):
            try:
                experience = (current_stacked_state, action, processed_reward, next_stacked_state, done)
                
                # 🔧 修复：检查经验回放缓冲区类型并使用正确的方法
                if hasattr(self.experience_replay, 'add'):
                    # PrioritizedReplayBuffer 使用 add 方法
                    self.experience_replay.add(experience)
                elif hasattr(self.experience_replay, 'append'):
                    # 普通 deque 使用 append 方法
                    self.experience_replay.append(experience)
                else:
                    # 兜底处理：创建简单的 deque
                    print("⚠️ 经验回放缓冲区方法不明确，使用默认方法")
                    if not hasattr(self, '_simple_experience_buffer'):
                        from collections import deque
                        self._simple_experience_buffer = deque(maxlen=10000)
                    self._simple_experience_buffer.append(experience)
                    self.experience_replay = self._simple_experience_buffer
            
                # 检查缓冲区大小并进行训练
                buffer_size = 0
                if hasattr(self.experience_replay, '__len__'):
                    buffer_size = len(self.experience_replay)
                elif hasattr(self.experience_replay, 'size'):
                    buffer_size = self.experience_replay.size()
                
                if buffer_size > self.batch_size:
                    self.step_count += 1
                    if self.step_count % self.train_interval == 0:
                        self._train_neural_network_batch(gamma)
                        
            except Exception as nn_error:
                print(f"⚠️ 神经网络更新失败: {nn_error}")
                # 回退到Q-learning
                if not self.use_neural or getattr(self, 'use_neural_training', True):
                    try:
                        current_q = self.q_table[old_y][old_x][action]
                        max_next_q = 0 if done else max(self.q_table[new_head_y][new_head_x])
                        new_q = current_q * (1 - alpha) + alpha * (processed_reward + gamma * max_next_q)
                        self.q_table[old_y][old_x][action] = new_q
                    except Exception as q_error:
                        print(f"⚠️ Q-learning 更新也失败: {q_error}")
        else:
            # Q-learning 更新
            if not self.use_neural or getattr(self, 'use_neural_training', True):
                try:
                    current_q = self.q_table[old_y][old_x][action]
                    max_next_q = 0 if done else max(self.q_table[new_head_y][new_head_x])
                    new_q = current_q * (1 - alpha) + alpha * (processed_reward + gamma * max_next_q)
                    self.q_table[old_y][old_x][action] = new_q
                except Exception as q_error:
                    print(f"⚠️ Q-learning 更新失败: {q_error}")

    def move(self, action, env):
        """执行移动动作"""
        import time
        start_time = time.time()
        self._prev_head_pos = (self.x, self.y)  # 记录移动前位置用于计算速度

        try:
            # 基础移动向量 (方向编号与 _action_vectors 统一; 8 方向对角不缩放)
            dx, dy = _action_vectors(self.action_size)[action]
            
            # 计算新的头部位置
            new_head_x = self.x + dx
            new_head_y = self.y + dy
            
            # 边界检查和撞墙反弹逻辑
            bounce_penalty = -5.0
            is_wall_collision = False
            
            if not (0 <= new_head_x < self.width and 0 <= new_head_y < self.height):
                # 撞墙了！执行反弹
                is_wall_collision = True
                self.total_reward += bounce_penalty
                
                # 计算反弹后的位置
                new_head_x = self.x - 2*dx
                new_head_y = self.y - 2*dy
                
                # 确保反弹后的位置也不会出界
                new_head_x = max(0, min(self.width - 1, new_head_x))
                new_head_y = max(0, min(self.height - 1, new_head_y))
            
            # 更新身体物理状态
            self._update_body_physics(new_head_x, new_head_y)
            
            # 更新位置
            self.x, self.y = new_head_x, new_head_y
            
            # 更新温度记录
            current_temp = env.get_temperature(int(new_head_x), int(new_head_y))
            self.recent_temperatures.append(current_temp)
            if len(self.recent_temperatures) > self.memory_size:
                self.recent_temperatures.pop(0)
            
            return True
            
        except Exception as e:
            print(f"❌ move方法异常: {e}")
            return False

    def _update_body_physics(self, new_head_x, new_head_y):
        """更新身体物理状态 - 简化版本"""
        # 🔧 简化的身体物理更新，避免复杂计算
        # 步骤1：头部移动到新位置
        self.body_segments[0] = [new_head_x, new_head_y]
        
        # 步骤2：让其他节段跟随（简化的跟随逻辑）
        for i in range(1, len(self.body_segments)):
            # 目标节段和前一个节段
            target_segment = self.body_segments[i-1]
            current_segment = self.body_segments[i]
            
            # 简单的跟随逻辑
            dx = target_segment[0] - current_segment[0]
            dy = target_segment[1] - current_segment[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > self.segment_distance * 1.2:  # 如果距离过大，移动
                if distance > 0.1:
                    unit_dx = dx / distance
                    unit_dy = dy / distance
                    
                    new_x = target_segment[0] - unit_dx * self.segment_distance
                    new_y = target_segment[1] - unit_dy * self.segment_distance
                    
                    self.body_segments[i] = [new_x, new_y]
        
        # 记录历史
        self.history.append(self.body_segments.copy())

    def _train_neural_network_batch(self, gamma):
        """修复版神经网络批训练（训练数学见模块级 train_dqn_batch）。"""
        train_dqn_batch(self, gamma)

    def reset(self, start_pos=None, **kwargs):
        """重置线虫状态"""
        try:
            if start_pos is None:
                start_pos = (self.x, self.y)
            self.x, self.y = start_pos
            self.total_reward = 0
            self.current_step = 0
            self.energy = self.max_energy
            self.muscle_fatigue_level = 0.0
            self.muscle_wave_phase = 0.0
            self._pending_action = None
            self.last_action = None
            self.last_physics_result = {
                'moved': False,
                'action': None,
                'position': (self.x, self.y)
            }
            
            # 重置身体段
            self.body_segments = []
            self.body_temperatures = []
            
            x, y = start_pos
            
            # 重新创建身体段
            for i in range(self.num_segments):
                segment_x = max(0, min(self.width-1, x - i * self.segment_distance))
                segment_y = y
                self.body_segments.append([segment_x, segment_y])
                self.body_temperatures.append(0.0)
            
            # 重置历史和缓冲区
            self.history = [self.body_segments.copy()]
            self.recent_temperatures = []
            self.state_buffer.clear()
            self._prev_head_pos = (self.x, self.y)  # 重置速度追踪
            
            # 重置记忆系统
            if hasattr(self, 'hotspot_memory'):
                self.hotspot_memory.clear()
            if hasattr(self, 'perception_memory'):
                self.perception_memory.clear()
            
            # 重置决策统计
            self.decision_stats = {
                'learning': 0, 'memory': 0, 'perception': 0, 'gradient': 0, 'total': 0
            }
            
            print(f"🔧 线虫重置完成 - 位置: {start_pos}, 身体段数: {len(self.body_segments)}")
            return self.get_observation(env=kwargs.get('env'))
            
        except Exception as e:
            print(f"❌ 线虫重置失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_observation(self, env=None, **kwargs):
        """返回统一观测数据，兼容当前 8 维环境状态。"""
        state_vector = None
        perception = None

        if env is not None:
            try:
                state_vector = env.get_state_vector((self.x, self.y), worm=self)
            except Exception as e:
                print(f"⚠️ 统一观测状态向量获取失败: {e}")
                state_vector = np.zeros(8, dtype=np.float32)

            if kwargs.get('include_perception', False):
                try:
                    perception = self.perceive_environment(env)
                except Exception as e:
                    print(f"⚠️ 统一观测感知数据获取失败: {e}")
                    perception = None

        return {
            'model': 'worm2d',
            'position': (self.x, self.y),
            'head_position': (self.x, self.y),
            'state_vector': state_vector,
            'body_temperatures': list(self.body_temperatures),
            'recent_temperatures': list(self.recent_temperatures),
            'energy': self.energy,
            'step': self.current_step,
            'pending_action': self._pending_action,
            'perception': perception
        }

    def apply_action(self, action, **kwargs):
        """缓存离散动作；物理推进由 step_physics 执行。"""
        try:
            action_index = int(action)
        except (TypeError, ValueError):
            raise ValueError(f"动作必须能转换为整数，实际为: {action!r}")

        if action_index < 0 or action_index >= self.action_size:
            raise ValueError(f"动作超出范围: {action_index}，有效范围 0 到 {self.action_size - 1}")

        self._pending_action = action_index
        return {
            'accepted': True,
            'action': action_index,
            'action_space': self.action_size
        }

    def step_physics(self, env=None, dt=1.0, **kwargs):
        """推进一次当前离散身体模型的物理状态。"""
        action = kwargs.get('action', None)
        if action is not None:
            self.apply_action(action)

        if self._pending_action is None:
            self.last_physics_result = {
                'moved': False,
                'action': None,
                'position': (self.x, self.y),
                'reason': 'no_pending_action'
            }
            return self.last_physics_result

        action_index = self._pending_action
        self._pending_action = None
        self.last_action = action_index

        if env is not None:
            moved = self.move(action_index, env)
        else:
            dx, dy = _action_vectors(self.action_size)[action_index]
            new_head_x = max(0, min(self.width - 1, self.x + dx))
            new_head_y = max(0, min(self.height - 1, self.y + dy))
            self._update_body_physics(new_head_x, new_head_y)
            self.x, self.y = new_head_x, new_head_y
            moved = True

        self.last_physics_result = {
            'moved': bool(moved),
            'action': action_index,
            'position': (self.x, self.y),
            'dt': dt
        }
        return self.last_physics_result

    def get_geometry(self):
        """返回当前分段身体几何。"""
        segments = [(float(x), float(y)) for x, y in self.body_segments]
        if segments:
            xs = [point[0] for point in segments]
            ys = [point[1] for point in segments]
            bounds = {
                'min_x': min(xs),
                'max_x': max(xs),
                'min_y': min(ys),
                'max_y': max(ys)
            }
            head = segments[0]
            tail = segments[-1]
        else:
            bounds = {'min_x': self.x, 'max_x': self.x, 'min_y': self.y, 'max_y': self.y}
            head = (float(self.x), float(self.y))
            tail = head

        return {
            'model': 'worm2d',
            'type': 'segmented_polyline',
            'head': head,
            'tail': tail,
            'segments': segments,
            'segment_count': len(segments),
            'segment_distance': float(self.segment_distance),
            'body_length': float(self.body_length),
            'head_radius': float(self.head_radius),
            'body_width': float(self.body_width),
            'bounds': bounds
        }

    def get_metrics(self, env=None):
        """返回训练和身体状态指标。"""
        valid_temps = [
            temp for temp in self.body_temperatures
            if temp != -float('inf') and np.isfinite(temp)
        ]
        points = [
            np.array([float(x), float(y)], dtype=float)
            for x, y in self.body_segments
        ]
        segment_lengths = []
        turn_angles = []
        if len(points) >= 2:
            segment_lengths = [
                float(np.linalg.norm(points[i + 1] - points[i]))
                for i in range(len(points) - 1)
            ]
        if len(points) >= 3:
            for i in range(1, len(points) - 1):
                prev_vector = points[i] - points[i - 1]
                next_vector = points[i + 1] - points[i]
                prev_norm = float(np.linalg.norm(prev_vector))
                next_norm = float(np.linalg.norm(next_vector))
                if prev_norm <= 1e-9 or next_norm <= 1e-9:
                    continue
                cosine = float(np.dot(prev_vector, next_vector) / (prev_norm * next_norm))
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                turn_angles.append(float(angle))

        target_segment_length = float(self.segment_distance)
        target_body_length = float(self.body_length)
        actual_body_length = float(sum(segment_lengths)) if segment_lengths else 0.0
        length_errors = [
            abs(length - target_segment_length)
            for length in segment_lengths
        ]
        min_segment_length = target_segment_length * float(getattr(self, 'min_segment_compression', 0.7))
        max_segment_length = target_segment_length * float(getattr(self, 'max_segment_stretch', 1.5))
        length_violation_count = sum(
            1 for length in segment_lengths
            if length < min_segment_length or length > max_segment_length
        )
        curvature_limit = float(getattr(self, 'angular_constraint', self.max_bend_angle))
        curvature_violation_count = sum(1 for angle in turn_angles if angle > curvature_limit)
        constraint_check_count = len(segment_lengths) + len(turn_angles)
        constraint_violation_count = length_violation_count + curvature_violation_count

        metrics = {
            'model': 'worm2d',
            'position': (self.x, self.y),
            'total_reward': float(self.total_reward),
            'current_step': int(self.current_step),
            'energy': float(self.energy),
            'energy_ratio': float(self.energy / self.max_energy) if self.max_energy else 0.0,
            'body_segments': len(self.body_segments),
            'history_length': len(self.history),
            'last_action': self.last_action,
            'pending_action': self._pending_action,
            'average_body_temperature': float(np.mean(valid_temps)) if valid_temps else None,
            'use_neural': bool(self.use_neural),
            'muscle_fatigue': float(getattr(self, 'muscle_fatigue_level', 0.0)),
            'target_body_length': target_body_length,
            'actual_body_length': actual_body_length,
            'body_length_error': float(actual_body_length - target_body_length),
            'body_length_error_abs': float(abs(actual_body_length - target_body_length)),
            'target_segment_length': target_segment_length,
            'average_segment_length': float(np.mean(segment_lengths)) if segment_lengths else 0.0,
            'max_segment_length_error': float(max(length_errors)) if length_errors else 0.0,
            'mean_segment_length_error': float(np.mean(length_errors)) if length_errors else 0.0,
            'length_violation_count': int(length_violation_count),
            'length_violation_rate': float(length_violation_count / len(segment_lengths)) if segment_lengths else 0.0,
            'curvature_mean_deg': float(np.mean(turn_angles)) if turn_angles else 0.0,
            'curvature_max_deg': float(max(turn_angles)) if turn_angles else 0.0,
            'curvature_limit_deg': curvature_limit,
            'curvature_violation_count': int(curvature_violation_count),
            'curvature_violation_rate': float(curvature_violation_count / len(turn_angles)) if turn_angles else 0.0,
            'constraint_violation_count': int(constraint_violation_count),
            'constraint_violation_rate': (
                float(constraint_violation_count / constraint_check_count)
                if constraint_check_count else 0.0
            )
        }

        if env is not None:
            try:
                metrics['head_temperature'] = float(env.get_temperature(int(self.x), int(self.y)))
            except Exception:
                metrics['head_temperature'] = None

            if hasattr(env, 'best_point') and env.best_point is not None:
                try:
                    best_x, best_y = env.best_point
                    metrics['distance_to_best'] = float(np.sqrt((self.x - best_x) ** 2 + (self.y - best_y) ** 2))
                except Exception:
                    metrics['distance_to_best'] = None

        return metrics

    def setup_neural_network(self, state_size=None, action_size=None, learning_rate=0.001):
        """设置神经网络"""
        try:
            if not PYTORCH_AVAILABLE or not DQN_IMPORT_SUCCESS:
                print("⚠️ 神经网络不可用，使用Q-table方法")
                self.use_neural = False
                return False
            
            # 单帧维度 (state_v2: 15, 旧版: 8); NN 输入 = 帧×4
            if state_size is None:
                state_size = 10 if getattr(self, 'use_state_v2', False) else 8
            self.state_size = state_size  # 单帧维度
            if action_size:
                self.action_size = action_size
            nn_input_size = state_size * 4  # 堆叠 4 帧

            print(f"🔧 初始化持久神经网络组件 (输入 {nn_input_size} 维)...")

            # 创建神经网络和优化器
            try:
                try:
                    from .neural_networks import setup_neural_network
                except ImportError:
                    from neural_networks import setup_neural_network
                self.neural_network, self.optimizer = setup_neural_network(
                    nn_input_size, self.action_size, learning_rate
                )
            except ImportError:
                print("⚠️ 无法导入setup_neural_network函数")
                self.neural_network, self.optimizer = None, None
            
            if self.neural_network is not None:
                self.use_neural = True
                
                # 初始化经验回放缓冲区
                try:
                    try:
                        from .neural_networks import ExperienceReplay
                    except ImportError:
                        from neural_networks import ExperienceReplay
                    self.experience_replay = ExperienceReplay(capacity=10000)
                except ImportError:
                    # 简单的经验回放实现
                    from collections import deque
                    self.experience_replay = deque(maxlen=10000)
                
                print(f"✓ 神经网络组件已准备就绪 - 使用 标准 DQN 架构")
                print(f"✓ 学习将跨轮次持续积累")
                return True
            else:
                print("⚠️ 神经网络创建失败，回退到Q-table方法")
                self.use_neural = False
                return False
                
        except Exception as e:
            print(f"❌ 神经网络设置失败: {e}")
            self.use_neural = False
            return False

    def get_state_summary(self):
        """获取状态摘要"""
        return {
            'position': (self.x, self.y),
            'energy': self.energy,
            'total_reward': self.total_reward,
            'current_step': self.current_step,
            'body_segments': len(self.body_segments),
            'use_neural': self.use_neural,
            'muscle_fatigue': getattr(self, 'muscle_fatigue_level', 0.0)
        }


class Worm2DModelAdapter(BodyModel):
    """把现有 Worm2D 对象包装成身体模型接口。"""

    def __init__(self, worm):
        object.__setattr__(self, "worm", worm)
        if not isinstance(worm, Worm2D):
            raise TypeError("Worm2DModelAdapter 只能包装 Worm2D 实例")

    @classmethod
    def create(cls, start_pos, width, height, body_params=None, noise_params=None):
        worm = Worm2D(
            start_pos=start_pos,
            width=width,
            height=height,
            body_params=body_params or {},
            noise_params=noise_params or {}
        )
        return cls(worm)

    def __getattr__(self, name):
        return getattr(self.worm, name)

    def __setattr__(self, name, value):
        if name == "worm":
            object.__setattr__(self, name, value)
        else:
            setattr(self.worm, name, value)

    def reset(self, start_pos=None, **kwargs):
        return self.worm.reset(start_pos=start_pos, **kwargs)

    def get_observation(self, env=None, **kwargs):
        return self.worm.get_observation(env=env, **kwargs)

    def apply_action(self, action, **kwargs):
        return self.worm.apply_action(action, **kwargs)

    def step_physics(self, env=None, dt=1.0, **kwargs):
        return self.worm.step_physics(env=env, dt=dt, **kwargs)

    def get_geometry(self):
        return self.worm.get_geometry()

    def get_metrics(self, env=None):
        return self.worm.get_metrics(env=env)


class ContinuousCenterlineBody(BodyModel):
    """固定弧长采样的连续中心线身体模型。"""

    model_name = "continuous_centerline"

    def __init__(self, start_pos, width, height, body_params=None, noise_params=None):
        body_params = body_params or {}
        noise_params = noise_params or {}
        self.width = int(width)
        self.height = int(height)
        self.num_segments = int(body_params.get("sample_count", body_params.get("num_segments", 9)))
        self.num_segments = max(3, self.num_segments)
        self.body_length = float(body_params.get("body_length", body_params.get("segment_length", 12.0)))
        self.segment_distance = self.body_length / max(1, self.num_segments - 1)
        self.head_radius = float(body_params.get("head_radius", 3.0))
        self.body_width = float(body_params.get("body_width", 2.0))
        self.forward_speed = float(body_params.get("forward_speed", 1.2))
        self.max_turn_angle = float(body_params.get("max_turn_angle", 35.0))
        self.angular_constraint = float(body_params.get("curvature_limit_deg", body_params.get("angular_constraint", 45.0)))
        self.length_stiffness = float(body_params.get("length_stiffness", 0.85))
        self.curvature_stiffness = float(body_params.get("curvature_stiffness", 0.35))
        self.damping = float(body_params.get("damping", 0.72))
        self.max_energy = float(body_params.get("max_energy", 100.0))
        self.energy = self.max_energy
        self.energy_decay_rate = float(body_params.get("energy_decay_rate", 0.12))
        self.low_energy_threshold = 0.5 * self.max_energy
        self.reward_variant = "original"      # original / energy（reward_functions 模块统一计算）
        self.reward_energy_weight = DEFAULT_ENERGY_WEIGHT
        self.muscle_fatigue_level = 0.0
        self.fatigue_accumulation_rate = float(body_params.get("fatigue_accumulation_rate", 0.006))
        self.fatigue_recovery_rate = float(body_params.get("fatigue_recovery_rate", 0.004))
        self.fatigue_threshold = 0.35
        self.max_fatigue_penalty = 0.5
        self.position_noise = float(noise_params.get("position_noise", 0.0))
        self.action_size = int(body_params.get("action_size", 4))  # 离散方向数: 4/8/16 (默认4保持兼容)
        self.use_neural = False
        self.use_actor_critic = False
        self.actor_critic_agent = None
        self.ac_state_dim = 12  # state_v2 12 维 (10 + cos/sin 朝向)
        self.ac_action_dim = 2  # AC 动作维: (heading, step); ADB 覆写为 3
        self.ac_action_bounds = [(-math.pi, math.pi), (0.05, 5.0)]  # 每维输出/裁剪边界
        # DQN 组件 (由 utils.setup_neural_network 装配)
        self.state_size = 12   # 单帧维度 = get_state_v2 输出 (DQN 输入 = 12×4)
        self.neural_network = None
        self.target_network = None
        self.optimizer = None
        self.experience_replay = None
        self.scheduler = None
        self.batch_size = 64
        self.train_interval = 4
        self.target_update_freq = 100
        self.step_count = 0        # DQN 训练批次计数 (每 100 批硬更新目标网络)
        self.dqn_step_counter = 0  # DQN 决策步计数 (step_physics 也递增 current_step，不能共用)
        self.ac_hidden_size = int(body_params.get("ac_hidden_size", 128))
        self.ac_actor_lr = float(body_params.get("ac_actor_lr", 1e-4))
        self.ac_critic_lr = float(body_params.get("ac_critic_lr", 1e-3))
        self.ac_gamma = float(body_params.get("ac_gamma", 0.95))
        self.ac_batch_size = int(body_params.get("ac_batch_size", 64))
        self.ac_noise_scale = float(body_params.get("ac_noise_scale", 0.6))
        self.q_table = [[[0.0] * self.action_size for _ in range(self.width)] for _ in range(self.height)]
        self.state_buffer = deque(maxlen=4)
        self.recent_temperatures = []
        self.visited_positions = {}
        self.current_step = 0
        self.total_reward = 0.0
        self.heading = float(body_params.get("initial_heading", 0.0))
        self.velocity = np.zeros(2, dtype=float)
        self._pending_action = None
        self.last_action = None
        self.last_physics_result = {}
        self.body_temperatures = [0.0] * self.num_segments
        self.segment_tensions = [0.0] * self.num_segments
        self.muscle_wave_phase = 0.0
        self.dorsal_muscle_state = 0.0
        self.ventral_muscle_state = 0.0
        self.base_muscle_wave_frequency = 0.0
        self.muscle_wave_frequency = 0.0
        self.reset(start_pos=start_pos)

    @classmethod
    def create(cls, start_pos, width, height, body_params=None, noise_params=None):
        return cls(start_pos=start_pos, width=width, height=height, body_params=body_params, noise_params=noise_params)

    def _clip_point(self, point):
        clipped = np.array(point, dtype=float)
        clipped[0] = float(np.clip(clipped[0], 0.0, max(0, self.width - 1)))
        clipped[1] = float(np.clip(clipped[1], 0.0, max(0, self.height - 1)))
        return clipped

    def _tangent(self):
        return np.array([math.cos(self.heading), math.sin(self.heading)], dtype=float)

    def _sync_public_state(self):
        self.centerline = np.array([self._clip_point(point) for point in self.centerline], dtype=float)
        self.body_segments = [[float(point[0]), float(point[1])] for point in self.centerline]
        self.body_segment = [self.body_segments[0], self.body_segments[-1]]
        self.x = float(self.centerline[0][0])
        self.y = float(self.centerline[0][1])
        if len(self.body_temperatures) != len(self.body_segments):
            self.body_temperatures = [0.0] * len(self.body_segments)

    def _sync_centerline_from_public_segments(self):
        if not hasattr(self, "body_segments") or len(self.body_segments) != self.num_segments:
            return
        self.centerline = np.array([[float(x), float(y)] for x, y in self.body_segments], dtype=float)
        self.x = float(self.centerline[0][0])
        self.y = float(self.centerline[0][1])

    def _initialize_centerline(self, start_pos):
        head = np.array([float(start_pos[0]), float(start_pos[1])], dtype=float)
        tangent = self._tangent()
        self.centerline = np.array([
            self._clip_point(head - tangent * self.segment_distance * i)
            for i in range(self.num_segments)
        ], dtype=float)
        self._enforce_constraints(iterations=3)
        self._sync_public_state()
        self.history = [self.body_segments.copy()]

    def _enforce_constraints(self, iterations=2):
        for _ in range(iterations):
            self.centerline[0] = self._clip_point(self.centerline[0])
            for i in range(1, self.num_segments):
                prev_point = self.centerline[i - 1]
                point = self.centerline[i]
                direction = point - prev_point
                distance = float(np.linalg.norm(direction))
                if distance <= 1e-9:
                    direction = -self._tangent()
                    distance = 1.0
                target = prev_point + direction / distance * self.segment_distance
                self.centerline[i] = self._clip_point(
                    point * (1.0 - self.length_stiffness) + target * self.length_stiffness
                )

            for i in range(1, self.num_segments - 1):
                prev_vector = self.centerline[i] - self.centerline[i - 1]
                next_vector = self.centerline[i + 1] - self.centerline[i]
                prev_norm = float(np.linalg.norm(prev_vector))
                next_norm = float(np.linalg.norm(next_vector))
                if prev_norm <= 1e-9 or next_norm <= 1e-9:
                    continue
                cosine = float(np.dot(prev_vector, next_vector) / (prev_norm * next_norm))
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                if angle > self.angular_constraint:
                    smoothed = 0.5 * (self.centerline[i - 1] + self.centerline[i + 1])
                    self.centerline[i] = self._clip_point(
                        self.centerline[i] * (1.0 - self.curvature_stiffness)
                        + smoothed * self.curvature_stiffness
                    )

    def reset(self, start_pos=None, **kwargs):
        if start_pos is None:
            start_pos = (getattr(self, "x", 0.0), getattr(self, "y", 0.0))
        self.energy = self.max_energy
        self.muscle_fatigue_level = 0.0
        self.current_step = 0
        self.total_reward = 0.0
        self.velocity = np.zeros(2, dtype=float)
        self._pending_action = None
        self.last_action = None
        self.last_physics_result = {'moved': False, 'action': None, 'position': start_pos}
        self.recent_temperatures.clear()
        self.visited_positions.clear()
        self.state_buffer.clear()
        self._initialize_centerline(start_pos)
        # 重置 Actor-Critic 噪声
        if self.use_actor_critic and self.actor_critic_agent is not None:
            self.actor_critic_agent.noise.reset()
        return self.get_observation(env=kwargs.get("env"))

    def get_observation(self, env=None, **kwargs):
        state_vector = None
        if env is not None:
            try:
                state_vector = env.get_state_vector((self.x, self.y), worm=self)
            except Exception:
                state_vector = np.zeros(8, dtype=np.float32)
        return {
            'model': self.model_name,
            'position': (self.x, self.y),
            'head_position': (self.x, self.y),
            'heading': float(self.heading),
            'velocity': (float(self.velocity[0]), float(self.velocity[1])),
            'state_vector': state_vector,
            'centerline': self.body_segments.copy(),
            'energy': float(self.energy),
            'muscle_fatigue': float(self.muscle_fatigue_level),
            'pending_action': self._pending_action,
        }

    def apply_action(self, action, **kwargs):
        self._pending_action = action
        return {'accepted': True, 'action': action}

    def _parse_action(self, action):
        if action is None:
            return self.heading, 0.0
        if isinstance(action, dict):
            heading = float(action.get("heading", self.heading + float(action.get("heading_delta", 0.0))))
            step = float(action.get("step", action.get("step_length", self.forward_speed)))
            return heading, step
        if isinstance(action, (tuple, list, np.ndarray)) and len(action) >= 2:
            return float(action[0]), float(action[1])
        action_index = int(action)
        headings = _action_headings(getattr(self, 'action_size', 4))
        return headings.get(action_index, self.heading), self.forward_speed

    def step_physics(self, env=None, dt=1.0, **kwargs):
        self._sync_centerline_from_public_segments()
        action = kwargs.get("action", self._pending_action)
        heading, step_distance = self._parse_action(action)
        old_head = self.centerline[0].copy()
        turn_amount = abs((heading - self.heading + math.pi) % (2.0 * math.pi) - math.pi)
        self.heading = heading
        desired_velocity = np.array([math.cos(self.heading), math.sin(self.heading)], dtype=float) * step_distance
        if self.position_noise:
            desired_velocity += np.random.normal(0.0, self.position_noise, size=2)
        self.velocity = self.velocity * self.damping + desired_velocity * (1.0 - self.damping)
        new_head = self._clip_point(old_head + self.velocity * float(dt))
        old_centerline = self.centerline.copy()
        self.centerline[0] = new_head
        for i in range(1, self.num_segments):
            self.centerline[i] = old_centerline[i - 1]
        self._enforce_constraints(iterations=3)
        self._sync_public_state()
        movement = float(np.linalg.norm(new_head - old_head))
        self.energy = max(0.0, self.energy - self.energy_decay_rate * (movement + 0.25 * turn_amount))
        if movement > 0:
            self.muscle_fatigue_level = min(1.0, self.muscle_fatigue_level + self.fatigue_accumulation_rate * movement)
        else:
            self.muscle_fatigue_level = max(0.0, self.muscle_fatigue_level - self.fatigue_recovery_rate)
        self.current_step += 1
        self.last_action = action
        self._pending_action = None
        self.history.append(self.body_segments.copy())
        if env is not None:
            self.body_temperatures = [
                float(env.get_temperature(int(point[0]), int(point[1])))
                for point in self.body_segments
            ]
            self.recent_temperatures.append(self.body_temperatures[0])
        moved = movement > 1e-9
        self.last_physics_result = {
            'moved': moved,
            'action': action,
            'position': (self.x, self.y),
            'movement': movement,
        }
        return self.last_physics_result

    def decide_move(self, env, epsilon=0.2, alpha=0.5, gamma=0.9):
        """决策并移动。优先级：Actor-Critic → DQN → Q-learning。"""
        # Actor-Critic 路径
        if self.use_actor_critic and self.actor_critic_agent is not None:
            return self.decide_move_actor_critic(env)

        # DQN / Dueling DQN 路径
        if self.use_neural and self.neural_network is not None and PYTORCH_AVAILABLE:
            return self._decide_move_neural(env, epsilon, gamma)

        # Q-learning 路径（原有逻辑）
        old_x, old_y = int(round(self.x)), int(round(self.y))
        old_energy = self.energy
        if random.random() < epsilon:
            action = random.randint(0, self.action_size - 1)
        else:
            action = int(np.argmax(self.q_table[old_y][old_x]))
        result = self.step_physics(env=env, action=action)
        new_x, new_y = int(round(self.x)), int(round(self.y))
        # 统一奖励函数模块（温度阶梯 + 单步能量；Q 路径无 stuck/constraint 惩罚）
        reward = compute_reward(
            env, self,
            variant=self.reward_variant,
            old_energy=old_energy,
            energy_weight=self.reward_energy_weight,
            target=(new_x, new_y),
        )
        if 0 <= old_y < self.height and 0 <= old_x < self.width:
            old_q = self.q_table[old_y][old_x][action]
            max_next_q = max(self.q_table[new_y][new_x])
            self.q_table[old_y][old_x][action] = old_q + alpha * (reward + gamma * max_next_q - old_q)
        self.total_reward += reward
        return bool(result.get('moved', False))

    def _decide_move_neural(self, env, epsilon, gamma):
        """DQN/Dueling DQN 决策：state_v2(12维)×4帧 → ε-greedy → 物理步 → 经验回放 → 批训练。

        - 惰性初始化 state_buffer：reset_worm_for_new_round 预热的是 8 维旧帧，
          首步检测帧维与 self.state_size 不一致时清空重灌 12 维帧，避免静默截断。
        - 经验 5 元组 done 恒 False（Q/DQN 路径暂无能量耗尽终止，与 Worm2D 现状一致）。
        - 训练：每 10 步且 buffer ≥ batch_size 时批训练；step_count 每训练一批 +1，
          每 target_update_freq(100) 批硬更新目标网络。
        """
        # 1. 惰性初始化状态缓冲，保证帧维度一致
        frame = self.get_state_v2(env)
        if (len(self.state_buffer) == 0
                or len(self.state_buffer[-1]) != self.state_size):
            self.state_buffer.clear()
            for _ in range(self.state_buffer.maxlen):
                self.state_buffer.append(frame)
        self.state_buffer.append(frame)
        stacked_state = get_stacked_state(self.state_buffer)

        # 2. 动作选择 (ε-greedy)
        if random.random() < epsilon:
            action = random.randrange(self.action_size)
        else:
            try:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(stacked_state).unsqueeze(0)
                    q_values = self.neural_network(state_tensor)
                    action = int(q_values.argmax().item())
            except Exception as e:
                print(f"⚠️ 神经网络推理失败，使用随机动作: {e}")
                action = random.randrange(self.action_size)

        # 3. 执行物理步并计算奖励
        old_energy = self.energy
        result = self.step_physics(env=env, action=action)
        reward = compute_reward(
            env, self,
            variant=getattr(self, 'reward_variant', 'original'),
            old_energy=old_energy,
            energy_weight=getattr(self, 'reward_energy_weight', DEFAULT_ENERGY_WEIGHT),
            target=(self.x, self.y),
        )

        # 4. 存储经验 (5 元组, done 恒 False)
        next_frame = self.get_state_v2(env)
        temp_buffer = self.state_buffer.copy()
        temp_buffer.append(next_frame)
        next_stacked = get_stacked_state(temp_buffer)
        experience = (stacked_state.copy(), action, reward, next_stacked.copy(), False)
        try:
            if hasattr(self.experience_replay, 'add'):
                # PrioritizedReplayBuffer 使用 add 方法
                self.experience_replay.add(experience)
            elif hasattr(self.experience_replay, 'append'):
                # 普通 deque 使用 append 方法
                self.experience_replay.append(experience)
        except Exception as e:
            print(f"⚠️ 经验存储失败: {e}")

        # 5. 触发批训练：每 10 个决策步且 buffer ≥ batch_size
        self.dqn_step_counter += 1
        buffer_size = len(self.experience_replay) if hasattr(self.experience_replay, '__len__') else 0
        if buffer_size >= self.batch_size and self.dqn_step_counter % 10 == 0:
            self.step_count += 1
            train_dqn_batch(self, gamma)

        # 6. 统计
        self.total_reward += reward
        return bool(result.get('moved', False))

    # ── Actor-Critic 相关方法 ─────────────────────────

    def get_state_v2(self, env=None):
        """返回 CCB/ADB 12 维状态向量：原 8 维 + 能量率 + 平均曲率 + 朝向(cos,sin)。

        朝向维度是能量-转向耦合的必要信息（转向能耗依赖当前朝向），
        用 cos/sin 表示以避免 ±π 跳变。Worm2D 保持 10 维（无 heading 概念）。
        """
        state = []

        # 1-8. 复用环境 8 维状态向量(四方向梯度 + 温度 + 对齐 + 趋势 + 距离)
        if env is not None:
            base = env.get_state_vector((self.x, self.y), worm=self)
        else:
            base = np.zeros(8, dtype=np.float32)
        state.extend(base.astype(np.float32).tolist())

        # 9. 能量率
        state.append(float(self.energy / max(self.max_energy, 1.0)))

        # 10. 平均曲率(归一化)
        shape = self._shape_metrics() if hasattr(self, '_shape_metrics') else {}
        curv_limit = max(self.angular_constraint, 1.0)
        mean_curv = shape.get('curvature_mean_deg', 0.0)
        state.append(float(np.clip(mean_curv / curv_limit, 0.0, 1.0)))

        # 11-12. 朝向 (cos/sin)
        state.append(float(math.cos(self.heading)))
        state.append(float(math.sin(self.heading)))

        return np.array(state[:12], dtype=np.float32)

    def setup_actor_critic(self):
        """初始化 DDPG Agent。"""
        if DDPGAgent is None or not AC_TORCH_AVAILABLE:
            print("⚠️ Actor-Critic 不可用 (需要 PyTorch)")
            self.use_actor_critic = False
            return False

        try:
            import torch  # noqa: F401
        except ImportError:
            self.use_actor_critic = False
            return False

        self.actor_critic_agent = DDPGAgent(
            state_dim=self.ac_state_dim,
            action_dim=getattr(self, 'ac_action_dim', 2),
            action_bounds=getattr(self, 'ac_action_bounds', None),
            hidden_size=self.ac_hidden_size,
            actor_lr=self.ac_actor_lr,
            critic_lr=self.ac_critic_lr,
            gamma=self.ac_gamma,
            batch_size=self.ac_batch_size,
            noise_scale=self.ac_noise_scale,
        )
        self.use_actor_critic = True
        self.use_neural = False  # AC 和 Q-learning 互斥
        print("✓ Actor-Critic (DDPG) Agent 初始化完成")
        return True

    def _build_actor_action(self, action):
        """把 Actor 连续输出转为物理动作 dict。CCB: (heading, step)。ADB 覆写为 (波幅, 频率, 曲率偏置)。"""
        return {"heading": float(action[0]), "step": float(action[1])}

    def decide_move_actor_critic(self, env):
        """
        Actor-Critic 决策：使用 Actor 输出连续动作 (维数 = ac_action_dim)，
        经 _build_actor_action 转为物理动作，收集经验并在 buffer 足够时训练。
        """
        agent = self.actor_critic_agent
        if agent is None:
            return False

        # 1. 获取当前状态
        state = self.get_state_v2(env)

        # 2. 选择动作
        action = agent.act(state, add_noise=agent.train_mode)
        action_dict = self._build_actor_action(action)

        # 3. 记录旧状态用于经验回放
        old_energy = self.energy

        # 4. 执行物理步
        result = self.step_physics(env=env, action=action_dict)

        # 5. 计算奖励（统一奖励函数：温度阶梯 + 单步能量 + 约束/原地惩罚）
        new_head = (self.x, self.y)
        movement = result.get('movement', 0.0)
        shape = self._shape_metrics() if hasattr(self, '_shape_metrics') else {}
        constraint_penalty = shape.get('constraint_violation_rate', 0.0) * 0.5
        reward = compute_reward(
            env, self,
            variant=self.reward_variant,
            old_energy=old_energy,
            energy_weight=self.reward_energy_weight,
            stuck_penalty=0.5,
            constraint_penalty=constraint_penalty,
            target=new_head,
            movement=movement,
        )

        # 6. 存储经验
        next_state = self.get_state_v2(env)
        done = self.energy <= 0.0
        agent.remember(state, np.asarray(action, dtype=np.float32), reward, next_state, done)

        # 7. 训练
        agent.train()

        # 8. 更新统计
        self.total_reward += reward
        self.current_step += 1
        self.recent_temperatures.append(float(env.get_temperature(int(self.x), int(self.y))))

        return bool(result.get('moved', False))

    def get_geometry(self):
        points = [(float(x), float(y)) for x, y in self.body_segments]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return {
            'model': self.model_name,
            'type': 'continuous_centerline',
            'head': points[0],
            'tail': points[-1],
            'segments': points,
            'centerline': points,
            'sample_count': len(points),
            'segment_count': len(points),
            'segment_distance': float(self.segment_distance),
            'body_length': float(self.body_length),
            'bounds': {
                'min_x': min(xs),
                'max_x': max(xs),
                'min_y': min(ys),
                'max_y': max(ys),
            },
        }

    def _shape_metrics(self):
        points = [np.array(point, dtype=float) for point in self.body_segments]
        segment_lengths = [
            float(np.linalg.norm(points[i + 1] - points[i]))
            for i in range(len(points) - 1)
        ]
        turn_angles = []
        for i in range(1, len(points) - 1):
            prev_vector = points[i] - points[i - 1]
            next_vector = points[i + 1] - points[i]
            prev_norm = float(np.linalg.norm(prev_vector))
            next_norm = float(np.linalg.norm(next_vector))
            if prev_norm <= 1e-9 or next_norm <= 1e-9:
                continue
            cosine = float(np.dot(prev_vector, next_vector) / (prev_norm * next_norm))
            turn_angles.append(float(math.degrees(math.acos(max(-1.0, min(1.0, cosine))))))
        actual_length = float(sum(segment_lengths))
        length_errors = [abs(length - self.segment_distance) for length in segment_lengths]
        curvature_violations = sum(1 for angle in turn_angles if angle > self.angular_constraint)
        return {
            'actual_body_length': actual_length,
            'body_length_error': float(actual_length - self.body_length),
            'body_length_error_abs': float(abs(actual_length - self.body_length)),
            'target_body_length': float(self.body_length),
            'target_segment_length': float(self.segment_distance),
            'average_segment_length': float(np.mean(segment_lengths)) if segment_lengths else 0.0,
            'mean_segment_length_error': float(np.mean(length_errors)) if length_errors else 0.0,
            'max_segment_length_error': float(max(length_errors)) if length_errors else 0.0,
            'curvature_mean_deg': float(np.mean(turn_angles)) if turn_angles else 0.0,
            'curvature_max_deg': float(max(turn_angles)) if turn_angles else 0.0,
            'curvature_limit_deg': float(self.angular_constraint),
            'curvature_violation_count': int(curvature_violations),
            'curvature_violation_rate': float(curvature_violations / len(turn_angles)) if turn_angles else 0.0,
            'constraint_violation_count': int(curvature_violations),
            'constraint_violation_rate': float(curvature_violations / len(turn_angles)) if turn_angles else 0.0,
        }

    def get_metrics(self, env=None):
        metrics = {
            'model': self.model_name,
            'position': (float(self.x), float(self.y)),
            'heading': float(self.heading),
            'total_reward': float(self.total_reward),
            'current_step': int(self.current_step),
            'energy': float(self.energy),
            'energy_ratio': float(self.energy / self.max_energy) if self.max_energy else 0.0,
            'muscle_fatigue': float(self.muscle_fatigue_level),
            'body_segments': len(self.body_segments),
            'history_length': len(self.history),
            'last_action': self.last_action,
            'pending_action': self._pending_action,
            'use_actor_critic': bool(getattr(self, 'use_actor_critic', False)),
            'training_method': 'actor_critic' if getattr(self, 'use_actor_critic', False) else 'q_learning',
        }
        metrics.update(self._shape_metrics())
        if env is not None:
            try:
                metrics['head_temperature'] = float(env.get_temperature(int(self.x), int(self.y)))
            except Exception:
                metrics['head_temperature'] = None
            if getattr(env, 'best_point', None) is not None:
                best_x, best_y = env.best_point
                metrics['distance_to_best'] = float(np.sqrt((self.x - best_x) ** 2 + (self.y - best_y) ** 2))
        return metrics


class ActiveDeformationBody(ContinuousCenterlineBody):
    """RFT 力基波驱动身体: 波 + 曲率偏置为输入, 运动由阻力力/力矩平衡涌现。

    物理依据 (Taylor 1951 游泳板; RFT; C. elegans 实测 cN/cT ≈ 1.4):
    - 形状 (身体局部系): y(s) = A·sin(k·s − phase) + (b/2)·s², 行波向尾传播
    - 每点速度: v = V + Ω×r + w (净平动 + 旋转 + 行波横向速度)
    - 各向异性阻力: F_i = −Δs·D_i·v_i, D_i = drag_coeff·(t tᵀ + drag_ratio·n nᵀ)
    - 自推进: ΣF=0 且 Σ r×F=0 → 3×3 线性方程解 (Vx, Vy, Ω); 全体点平移+旋转
    - 转向: 曲率偏置 b 产生净力矩 → 身体旋转, b 符号决定左右
    - 能量 = 机械耗散功 Σ vᵀD v Δs dt (替换旧的拍脑袋系数)
    - 仅支持 Actor-Critic 连续动作 (波幅, 频率, 曲率偏置); 无 heading 动作
    - 决策步 ≠ 物理步: 每决策步内做 sub_steps 个子步积分 (相位平滑、积分更准)
    """

    model_name = "active_deformation"

    def __init__(self, start_pos, width, height, body_params=None, noise_params=None):
        body_params = body_params or {}
        self.wave_amplitude = float(body_params.get("wave_amplitude", 1.0))
        self.wave_frequency = float(body_params.get("wave_frequency", 0.25))
        self.wave_phase = float(body_params.get("wave_phase", 0.0))
        self.wave_speed = float(body_params.get("wave_speed", 1.0))
        self.wave_length = float(body_params.get("wave_length", body_params.get("body_length", body_params.get("segment_length", 12.0))))
        self.steer_bias = float(body_params.get("steer_bias", 0.0))   # 曲率偏置 (AC 动作第三维)
        self.wave_envelope = bool(body_params.get("wave_envelope", True))  # 头尾波幅包络 (生物真实 + 削弱端点伪影)
        self.drag_ratio = 1.4   # 固定: C. elegans 实测法向/切向阻力比 (非可调)
        self.drag_coeff = float(body_params.get("drag_coeff", 0.35))  # 阻力绝对量级 (标定速度尺度)
        self.sub_steps = max(1, int(body_params.get("sub_steps", 10)))  # 子步积分
        self.curriculum_freeze_steps = int(body_params.get("curriculum_freeze_steps", 0))  # 课程: 前 N 决策步冻结 b=0
        self.decision_count = 0
        # RFT 状态: 质心与身体轴朝向 (必须在 super().__init__ 前, 其 reset 会调用 _initialize_centerline)
        self.com = np.array([float(start_pos[0]), float(start_pos[1])], dtype=float)
        self.body_theta = float(body_params.get("initial_heading", 0.0))
        super().__init__(start_pos=start_pos, width=width, height=height, body_params=body_params, noise_params=noise_params)
        # AC 动作 = (波幅, 频率, 曲率偏置); 速度与转向由物理涌现, 不存在于动作空间
        self.ac_action_dim = 3
        self.ac_action_bounds = [(0.05, 2.0), (0.02, 0.8), (-0.08, 0.08)]
        self.ac_state_dim = 14   # 8 环境 + 能量率 + 曲率 + 身体轴朝向 cos/sin + 波相位 cos/sin
        self.state_size = 14
        self.base_muscle_wave_frequency = self.wave_frequency
        self.muscle_wave_frequency = self.wave_frequency
        self.muscle_wave_amplitude = self.wave_amplitude

    # ── 形状与坐标系 ─────────────────────────────

    def _rebuild_shape_points(self):
        """由 (com, body_theta, phase, A, b) 重建世界系身体点。

        约定: s 从 −L/2 (头) 到 +L/2 (尾); 身体轴向尾, 头朝向 = body_theta;
        世界点 = com + R(θ)·(−s, y)。感知位置 = 质心 com。
        """
        k = 2.0 * math.pi / max(self.wave_length, 1e-6)
        A, b = self.wave_amplitude, self.steer_bias
        # 中点采样: n 个点放在 n 个等分段的中心, 避免 λ=body_length 时首尾端点相位重合
        # (端点采样会对同一相位重复计数, 产生与点数无关的横向漂移伪影)
        s = np.array([-self.body_length / 2.0 + (i + 0.5) * self.body_length / self.num_segments
                      for i in range(self.num_segments)], dtype=float)
        # 波幅包络: 头尾渐变为零 (真实线虫弯曲幅值头尾为零, 且削弱端点力矩伪影)
        env = np.sin(math.pi * (s + self.body_length / 2.0) / self.body_length) if self.wave_envelope else np.ones(self.num_segments)
        y = A * env * np.sin(k * s - self.wave_phase) + 0.5 * b * s * s
        cth, sth = math.cos(self.body_theta), math.sin(self.body_theta)
        pts = np.stack([cth * (-s) - sth * y, sth * (-s) + cth * y], axis=1) + self.com
        self.centerline = pts
        self.body_segments = [[float(p[0]), float(p[1])] for p in pts]
        self.x, self.y = float(self.com[0]), float(self.com[1])

    def _initialize_centerline(self, start_pos):
        if not hasattr(self, 'com'):
            self.com = np.array([float(start_pos[0]), float(start_pos[1])], dtype=float)
        if not hasattr(self, 'body_theta'):
            self.body_theta = float(getattr(self, 'heading', 0.0))
        # 不重置 wave_phase: 尊重 body_params 传入值 (轮次重置在 _sync_centerline_from_public_segments 中处理)
        self._rebuild_shape_points()
        self.history = [self.body_segments.copy()]

    def _sync_centerline_from_public_segments(self):
        """reset_worm_for_new_round 直接写入直线 body_segments 后调用此处:
        把直线解释为新一轮初始轴 (质心=头部位置, 朝向=头−尾方向), 重建 RFT 形状。"""
        if not hasattr(self, "body_segments") or len(self.body_segments) != self.num_segments:
            return
        head = np.array([float(self.body_segments[0][0]), float(self.body_segments[0][1])])
        tail = np.array([float(self.body_segments[-1][0]), float(self.body_segments[-1][1])])
        self.com = head.copy()
        direction = head - tail  # 头朝向 = 从尾指向头
        if np.linalg.norm(direction) > 1e-9:
            self.body_theta = math.atan2(direction[1], direction[0])
        else:
            self.body_theta = float(getattr(self, 'heading', 0.0))
        self.wave_phase = 0.0
        self._rebuild_shape_points()

    def _sync_public_state(self):
        """ADB: 感知位置 = 质心; heading 与身体轴朝向同步。"""
        self.centerline = np.array([self._clip_point(p) for p in self.centerline], dtype=float)
        self.body_segments = [[float(p[0]), float(p[1])] for p in self.centerline]
        self.body_segment = [self.body_segments[0], self.body_segments[-1]]
        self.x = float(self.com[0])
        self.y = float(self.com[1])
        self.heading = self.body_theta
        if len(self.body_temperatures) != len(self.body_segments):
            self.body_temperatures = [0.0] * len(self.body_segments)

    # ── 动作与决策 ─────────────────────────────

    def apply_action(self, action, **kwargs):
        if isinstance(action, dict):
            if "wave_amplitude" in action:
                self.wave_amplitude = float(action["wave_amplitude"])
            if "wave_frequency" in action:
                self.wave_frequency = float(action["wave_frequency"])
            if "steer_bias" in action:
                self.steer_bias = float(action["steer_bias"])
            if "wave_phase" in action:
                self.wave_phase = float(action["wave_phase"])
            if "wave_speed" in action:
                self.wave_speed = float(action["wave_speed"])
        elif isinstance(action, (tuple, list, np.ndarray)) and len(action) == 3:
            self.wave_amplitude = float(action[0])
            self.wave_frequency = float(action[1])
            self.steer_bias = float(action[2])
        elif isinstance(action, (tuple, list, np.ndarray)) and len(action) >= 4:
            self.wave_amplitude = float(action[0])
            self.wave_frequency = float(action[1])
            self.wave_phase = float(action[2])
            self.wave_speed = float(action[3])
        return super().apply_action(action, **kwargs)

    def _build_actor_action(self, action):
        """ADB: Actor 输出 (波幅, 频率, 曲率偏置) → 物理动作 dict。
        课程阶段一: 前 curriculum_freeze_steps 个决策步强制 b=0 (只学速度-能耗)。"""
        self.decision_count += 1
        bias = float(action[2])
        if self.curriculum_freeze_steps > 0 and self.decision_count <= self.curriculum_freeze_steps:
            bias = 0.0
        return {
            "wave_amplitude": float(action[0]),
            "wave_frequency": float(action[1]),
            "steer_bias": bias,
        }

    def decide_move(self, env, epsilon=0.2, alpha=0.5, gamma=0.9):
        """ADB 仅支持 Actor-Critic 连续波参数控制 (无 heading/离散方向动作)。"""
        if not self.use_actor_critic or self.actor_critic_agent is None:
            raise RuntimeError(
                "ADB (RFT 波驱动) 仅支持 Actor-Critic 路径: 请先调用 setup_actor_critic()。"
                "Q-learning/DQN 方向动作对波驱动模型无意义。"
            )
        return self.decide_move_actor_critic(env)

    def get_state_v2(self, env=None):
        """ADB 14 维状态: 环境 8 维(质心) + 能量率 + 平均曲率 + 身体轴朝向 cos/sin + 波相位 cos/sin。"""
        state = []
        if env is not None:
            base = env.get_state_vector((self.x, self.y), worm=self)
        else:
            base = np.zeros(8, dtype=np.float32)
        state.extend(base.astype(np.float32).tolist())
        state.append(float(self.energy / max(self.max_energy, 1.0)))
        shape = self._shape_metrics() if hasattr(self, '_shape_metrics') else {}
        curv_limit = max(self.angular_constraint, 1.0)
        state.append(float(np.clip(shape.get('curvature_mean_deg', 0.0) / curv_limit, 0.0, 1.0)))
        state.append(float(math.cos(self.body_theta)))
        state.append(float(math.sin(self.body_theta)))
        state.append(float(math.cos(self.wave_phase)))
        state.append(float(math.sin(self.wave_phase)))
        return np.array(state[:14], dtype=np.float32)

    # ── RFT 物理 ─────────────────────────────

    def _rft_substep(self, dt):
        """一个 RFT 物理子步: 行波推进 + 3×3 力/力矩平衡解 (Vx, Vy, Ω) + 全体平移旋转。"""
        self.wave_phase += 2.0 * math.pi * self.wave_frequency * dt * self.wave_speed
        k = 2.0 * math.pi / max(self.wave_length, 1e-6)
        A, f, b = self.wave_amplitude, self.wave_frequency, self.steer_bias
        n = self.num_segments
        s = np.array([-self.body_length / 2.0 + (i + 0.5) * self.body_length / n
                      for i in range(n)], dtype=float)  # 中点采样 (同 _rebuild_shape_points)
        cth, sth = math.cos(self.body_theta), math.sin(self.body_theta)

        # 形状与波速 (局部系, 含头尾幅值包络)
        env = np.sin(math.pi * (s + self.body_length / 2.0) / self.body_length) if self.wave_envelope else np.ones(n)
        envp = (math.pi / self.body_length) * np.cos(math.pi * (s + self.body_length / 2.0) / self.body_length) if self.wave_envelope else np.zeros(n)
        y = A * env * np.sin(k * s - self.wave_phase) + 0.5 * b * s * s
        yp = A * (env * k * np.cos(k * s - self.wave_phase) + envp * np.sin(k * s - self.wave_phase)) + b * s  # dy/ds
        ydot = -A * env * (2.0 * math.pi * f) * np.cos(k * s - self.wave_phase)  # 行波横向速度

        # 局部单位切向 (沿身体轴向尾, 局部系方向 (-1, yp)) → 世界系
        inv = 1.0 / np.sqrt(1.0 + yp * yp)
        tx = (-cth - yp * sth) * inv
        ty = (-sth + yp * cth) * inv
        nx, ny = -ty, tx

        # 相对质心的位置与行波世界速度
        rx = cth * (-s) - sth * y
        ry = sth * (-s) + cth * y
        wx = -sth * ydot
        wy = cth * ydot

        # 组装 3×3: ΣF=0, Σ r×F=0; 未知 [Vx, Vy, Ω]; M·x = rhs
        cT, cN = 1.0, self.drag_ratio
        M = np.zeros((3, 3), dtype=float)
        rhs = np.zeros(3, dtype=float)
        for i in range(n):
            dxx = (cT * tx[i] * tx[i] + cN * nx[i] * nx[i]) * self.drag_coeff
            dxy = (cT * tx[i] * ty[i] + cN * nx[i] * ny[i]) * self.drag_coeff
            dyy = (cT * ty[i] * ty[i] + cN * ny[i] * ny[i]) * self.drag_coeff
            rotx, roty = -ry[i], rx[i]                      # Ω 的速度贡献方向
            drotx = dxx * rotx + dxy * roty                 # D·rot
            droty = dxy * rotx + dyy * roty
            dwx = dxx * wx[i] + dxy * wy[i]                 # D·w
            dwy = dxy * wx[i] + dyy * wy[i]
            M[0, 0] += dxx; M[0, 1] += dxy; M[0, 2] += drotx
            M[1, 0] += dxy; M[1, 1] += dyy; M[1, 2] += droty
            rhs[0] -= dwx; rhs[1] -= dwy
            # 力矩行: Σ r×D·V + Σ r×D·Ωrot = −Σ r×D·w
            M[2, 0] += rx[i] * dxy - ry[i] * dxx
            M[2, 1] += rx[i] * dyy - ry[i] * dxy
            M[2, 2] += rx[i] * droty - ry[i] * drotx
            rhs[2] -= rx[i] * dwy - ry[i] * dwx
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol = np.zeros(3)
        vx, vy, omega = float(sol[0]), float(sol[1]), float(sol[2])

        # 更新刚体运动并重建形状
        self.com += np.array([vx, vy]) * dt
        self.body_theta += omega * dt
        self._rebuild_shape_points()

        # 边界: 整体平移回界内 (保持形状, 不逐点 clip)
        lo_x, lo_y = 0.0, 0.0
        hi_x, hi_y = float(max(0, self.width - 1)), float(max(0, self.height - 1))
        shift_x = min(0.0, hi_x - self.centerline[:, 0].max()) + max(0.0, lo_x - self.centerline[:, 0].min())
        shift_y = min(0.0, hi_y - self.centerline[:, 1].max()) + max(0.0, lo_y - self.centerline[:, 1].min())
        if shift_x != 0.0 or shift_y != 0.0:
            self.com += np.array([shift_x, shift_y])
            self._rebuild_shape_points()

        # 机械耗散功: Σ vᵀ D v Δs dt (恒 ≥ 0)
        ds = self.segment_distance
        diss = 0.0
        for i in range(n):
            vix = vx + omega * (-ry[i]) + wx[i]
            viy = vy + omega * rx[i] + wy[i]
            fpx = (cT * tx[i] * (tx[i] * vix + ty[i] * viy)
                   + cN * nx[i] * (nx[i] * vix + ny[i] * viy)) * self.drag_coeff
            fpy = (cT * ty[i] * (tx[i] * vix + ty[i] * viy)
                   + cN * ny[i] * (nx[i] * vix + ny[i] * viy)) * self.drag_coeff
            diss += (vix * fpx + viy * fpy) * ds * dt
        return {'movement': float(np.hypot(vx, vy) * dt), 'dissipation': float(max(0.0, diss))}

    def step_physics(self, env=None, dt=1.0, **kwargs):
        """ADB 物理步: 决策步内做 sub_steps 个子步 RFT 积分。能量扣减 = 机械耗散功。"""
        action = kwargs.get("action", self._pending_action)
        if isinstance(action, dict):
            self.apply_action(action)
        n_sub = max(1, int(self.sub_steps))
        dt_sub = float(dt) / n_sub
        total_movement = 0.0
        total_dissipation = 0.0
        for _ in range(n_sub):
            sub = self._rft_substep(dt=dt_sub)
            total_movement += sub['movement']
            total_dissipation += sub['dissipation']
            self.history.append(self.body_segments.copy())  # 子步帧 → 动画看到平滑行波
        self.energy = max(0.0, self.energy - total_dissipation)
        self.muscle_fatigue_level = min(
            1.0,
            self.muscle_fatigue_level
            + self.fatigue_accumulation_rate * abs(self.wave_amplitude * self.wave_frequency) * float(dt),
        )
        self.muscle_wave_phase = self.wave_phase
        self.muscle_wave_frequency = self.wave_frequency
        self.muscle_wave_amplitude = self.wave_amplitude
        self.dorsal_muscle_state = math.sin(self.wave_phase)
        self.ventral_muscle_state = -self.dorsal_muscle_state
        if env is not None:
            self.body_temperatures = [
                float(env.get_temperature(int(p[0]), int(p[1]))) for p in self.body_segments
            ]
        self.current_step += 1
        self._sync_public_state()
        result = {
            'moved': total_movement > 1e-9,
            'action': action,
            'position': (self.x, self.y),
            'movement': total_movement,
            'dissipation': total_dissipation,
            'wave_phase': float(self.wave_phase),
        }
        self.last_physics_result = result
        return result

    def get_metrics(self, env=None):
        metrics = super().get_metrics(env=env)
        metrics.update({
            'wave_amplitude': float(self.wave_amplitude),
            'wave_frequency': float(self.wave_frequency),
            'wave_phase': float(self.wave_phase),
            'wave_speed': float(self.wave_speed),
            'wave_length': float(self.wave_length),
            'steer_bias': float(self.steer_bias),
            'drag_ratio': float(self.drag_ratio),
            'body_theta': float(self.body_theta),
        })
        return metrics


def create_body_model(model_type="worm2d", **kwargs):
    """身体模型工厂函数。当前支持基于 Worm2D 的兼容实现。"""
    normalized_type = str(model_type).lower()
    if normalized_type in ("worm2d", "segmented_worm2d", "legacy_worm2d"):
        return Worm2DModelAdapter.create(**kwargs)
    if normalized_type in ("continuous_centerline", "centerline", "continuous"):
        return ContinuousCenterlineBody.create(**kwargs)
    if normalized_type in ("active_deformation", "active_wave", "wave_body"):
        return ActiveDeformationBody.create(**kwargs)
    raise ValueError(f"未知身体模型类型: {model_type}")


__all__ = [
    "BodyModel",
    "Worm2D",
    "Worm2DModelAdapter",
    "ContinuousCenterlineBody",
    "ActiveDeformationBody",
    "create_body_model",
]
