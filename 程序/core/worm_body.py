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
        self.q_table = [[[0.0, 0.0, 0.0, 0.0] for _ in range(width)] for _ in range(height)]
        self.use_neural = False
        self.neural_network = None
        self.optimizer = None
        self.state_size = 8
        self.action_size = 4
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
            current_state = env.get_state_vector(current_pos, worm=self)
            
            # 检查状态有效性
            if current_state is None or len(current_state) == 0:
                print(f"⚠️ 获取到无效状态，使用默认状态")
                current_state = np.zeros(8)
            
            self.state_buffer.append(current_state)
            state_time = time.time() - state_start
            
            if state_time > 0.2:
                print(f"⚠️ 状态处理耗时: {state_time:.3f}s")
            
            if check_timeout("状态处理"):
                return False
            
            # 🔧 调试点4：获取堆叠状态 - 添加超时保护
            stacked_start = time.time()
            try:
                stacked_state = get_stacked_state(self.state_buffer)
                if stacked_state is None or len(stacked_state) == 0:
                    print(f"⚠️ 堆叠状态无效，使用默认状态")
                    stacked_state = np.zeros(32)  # 8 * 4 = 32
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
            
            # 🔧 修复：获取新状态并构造32维堆叠状态用于神经网络
            new_pos = (self.x, self.y)
            new_state_8d = env.get_state_vector(new_pos, worm=self)
            if new_state_8d is None:
                new_state_8d = np.zeros(8)
            
            # 为神经网络构造32维的新堆叠状态
            temp_buffer = self.state_buffer.copy()
            temp_buffer.append(new_state_8d)
            new_stacked_state = get_stacked_state(temp_buffer)
            
            # 🔧 使用8.22版本的简单奖励计算
            try:
                reward = self._calculate_reward(env, self.x, self.y)
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

    def _select_action(self, current_state, adjusted_epsilon, old_x, old_y):
        """选择动作"""
        if self.use_neural and self.neural_network is not None and PYTORCH_AVAILABLE and torch is not None:
            if random.random() < adjusted_epsilon:
                action = random.choice([0,1,2,3])
            else:
                try:
                    with torch.no_grad():
                        # 🔧 强制检查状态维度，确保是32维状态
                        if len(current_state) != 32:
                            print(f"⚠️ 状态维度错误: {len(current_state)}, 期望32维，使用随机动作")
                            action = random.choice([0,1,2,3])
                        else:
                            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
                            q_values = self.neural_network(state_tensor)
                            action = q_values.argmax().item()
                except Exception as e:
                    print(f"⚠️ 神经网络推理失败，使用随机动作: {e}")
                    action = random.choice([0,1,2,3])
        else:
            if random.random() < adjusted_epsilon:
                action = random.choice([0,1,2,3])
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

    def _calculate_reward(self, env, new_head_x, new_head_y):
        """
        简化的奖励计算 - 回归7.26版本的简洁设计
        """
        new_temp = env.get_temperature(new_head_x, new_head_y)

        # 出界或无效位置
        if new_temp == -float('inf'):
            return -10.0

        # 基础温度奖励系统 - 调整为120度最佳温度
        if new_temp >= 120:
            temp_reward = 5.0
        elif new_temp >= 115:
            temp_reward = 4.5
        elif new_temp >= 110:
            temp_reward = 4.0
        elif new_temp >= 100:
            temp_reward = 3.0
        elif new_temp >= 90:
            temp_reward = 2.0
        elif new_temp >= 80:
            temp_reward = 1.0
        elif new_temp >= 70:
            temp_reward = 0.5
        elif new_temp >= 60:
            temp_reward = 0.3
        elif new_temp >= 50:
            temp_reward = 0.1
        elif new_temp >= 40:
            temp_reward = 0.0
        elif new_temp >= 30:
            temp_reward = -0.3
        elif new_temp >= 20:
            temp_reward = -0.5
        elif new_temp >= 10:
            temp_reward = -0.8
        else:
            temp_reward = -1.0

        # 温度梯度奖励
        if len(self.recent_temperatures) >= 2:
            recent_avg = sum(self.recent_temperatures[-2:]) / 2
            if new_temp > recent_avg + 1.0:
                temp_reward += 1.0
            elif new_temp > recent_avg + 0.5:
                temp_reward += 0.3
            elif new_temp < recent_avg - 1.0:
                temp_reward -= 0.8
            elif new_temp < recent_avg - 0.5:
                temp_reward -= 0.2

        temp_reward += 0.1
        return temp_reward

    def move(self, action, env):
        """执行移动动作"""
        import time
        start_time = time.time()
        
        try:
            # 基础移动向量
            moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 上下左右
            dx, dy = moves[action]
            
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
        """修复版神经网络批训练"""
        if not PYTORCH_AVAILABLE or self.experience_replay is None:
            return
        if not getattr(self, 'use_neural_training', True):
            return
        
        try:
            # 🔧 修复：兼容不同类型的经验回放缓冲区
            batch_size = min(self.batch_size, len(self.experience_replay))
            if batch_size < 8:  # 最小批次大小
                return
            
            # 采样经验
            if hasattr(self.experience_replay, 'sample'):
                # PrioritizedReplayBuffer
                try:
                    tree_indices, experiences_data, is_weights = self.experience_replay.sample(batch_size)
                    if tree_indices is None or experiences_data is None:
                        return
                except Exception as sample_error:
                    print(f"⚠️ 优先经验回放采样失败: {sample_error}")
                    return
            else:
                # 普通 deque - 随机采样
                import random
                experiences_data = random.sample(list(self.experience_replay), batch_size)
                is_weights = np.ones(batch_size)  # 等权重
                tree_indices = None
        
            # 提取批次数据
            states = np.vstack([e[0] for e in experiences_data])
            actions = np.array([e[1] for e in experiences_data])
            rewards = np.array([e[2] for e in experiences_data])
            next_states = np.vstack([e[3] for e in experiences_data])
            dones = np.array([e[4] for e in experiences_data])
            
            # 转换为张量
            import torch
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards)
            next_states_tensor = torch.FloatTensor(next_states)
            dones_tensor = torch.BoolTensor(dones)
            is_weights_tensor = torch.FloatTensor(is_weights)
            
            # 计算当前Q值
            current_q_values = self.neural_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))
            
            # 计算目标Q值
            with torch.no_grad():
                if hasattr(self, 'target_network') and self.target_network is not None:
                    next_actions = self.neural_network(next_states_tensor).argmax(1)
                    next_q_values_target = self.target_network(next_states_tensor)
                    next_max_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1))
                else:
                    next_max_q_values = self.neural_network(next_states_tensor).max(1)[0].unsqueeze(1)
            
            next_max_q_values[dones_tensor.unsqueeze(1)] = 0.0
            target_q_values = rewards_tensor.unsqueeze(1) + gamma * next_max_q_values
        
            # 计算损失
            td_errors = torch.abs(target_q_values - current_q_values).detach()
            loss = torch.mean(is_weights_tensor.unsqueeze(1) * torch.nn.functional.mse_loss(current_q_values, target_q_values, reduction='none'))
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.neural_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 更新优先级
            if tree_indices is not None and hasattr(self.experience_replay, 'batch_update'):
                try:
                    td_errors_numpy = td_errors.squeeze().cpu().numpy()
                    if td_errors_numpy.ndim > 1:
                        td_errors_numpy = td_errors_numpy.flatten()
                    td_errors_numpy = td_errors_numpy.astype(np.float32)
                    self.experience_replay.batch_update(tree_indices, td_errors_numpy)
                except Exception as update_error:
                    print(f"⚠️ 优先级更新失败: {update_error}")
        
            # 更新目标网络
            if hasattr(self, 'target_network') and hasattr(self, 'target_update_freq'):
                if self.step_count % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.neural_network.state_dict())
        
        except Exception as e:
            print(f"⚠️ 神经网络批训练失败: {e}")

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
            moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dx, dy = moves[action_index]
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
        metrics = {
            'model': 'worm2d',
            'position': (self.x, self.y),
            'total_reward': float(self.total_reward),
            'current_step': int(self.current_step),
            'energy': float(self.energy),
            'body_segments': len(self.body_segments),
            'history_length': len(self.history),
            'last_action': self.last_action,
            'pending_action': self._pending_action,
            'average_body_temperature': float(np.mean(valid_temps)) if valid_temps else None,
            'use_neural': bool(self.use_neural),
            'muscle_fatigue': float(getattr(self, 'muscle_fatigue_level', 0.0))
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
            
            # 使用传入的参数或默认值
            self.state_size = state_size or 32  # 8 * 4 = 32 (堆叠状态)
            self.action_size = action_size or 4
            
            print(f"🔧 初始化持久神经网络组件...")
            
            # 创建神经网络和优化器
            try:
                try:
                    from .neural_networks import setup_neural_network
                except ImportError:
                    from neural_networks import setup_neural_network
                self.neural_network, self.optimizer = setup_neural_network(
                    self.state_size, self.action_size, learning_rate
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

    def calculate_reward_optimized(self, old_position, new_position, env, perception_data, 
                                 old_body_segments, move_successful):
        """
        优化的奖励计算方法
        """
        try:
            if not move_successful:
                return -10.0  # 移动失败的惩罚
            
            # 获取新位置温度
            new_temp = env.get_temperature(int(new_position[0]), int(new_position[1]))
            
            # 使用简化的奖励计算
            reward = self._calculate_reward(env, int(new_position[0]), int(new_position[1]))
            
            # 添加感知奖励
            if perception_data:
                # 如果朝着更高温度方向移动，给予奖励
                if 'best_direction' in perception_data and perception_data['best_direction'] is not None:
                    reward += 0.5
                
                # 梯度导向奖励
                if 'gradient_confidence' in perception_data and perception_data['gradient_confidence'] > 0.5:
                    reward += 0.3
            
            # 能量奖励
            if self.energy > self.low_energy_threshold:
                reward += 0.1
            else:
                reward -= 0.2
            
            return reward
            
        except Exception as e:
            print(f"⚠️ 奖励计算异常: {e}")
            return 0.0

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


def create_body_model(model_type="worm2d", **kwargs):
    """身体模型工厂函数。当前支持基于 Worm2D 的兼容实现。"""
    normalized_type = str(model_type).lower()
    if normalized_type in ("worm2d", "segmented_worm2d", "legacy_worm2d"):
        return Worm2DModelAdapter.create(**kwargs)
    raise ValueError(f"未知身体模型类型: {model_type}")


__all__ = [
    "BodyModel",
    "Worm2D",
    "Worm2DModelAdapter",
    "create_body_model",
]
