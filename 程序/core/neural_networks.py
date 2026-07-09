"""
神经网络模块 - 包含深度学习相关的类和功能
"""
from collections import deque
import numpy as np
import random
import os

# 🔧 修复PyTorch导入
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
    print("✓ PyTorch导入成功")
except ImportError as e:
    print(f"❌ PyTorch导入失败: {e}")
    PYTORCH_AVAILABLE = False
    # 创建占位类避免ImportError
    class nn:
        class Module:
            def __init__(self): pass
        class Linear:
            def __init__(self, *args, **kwargs): pass
        class ReLU:
            def __init__(self, *args, **kwargs): pass
    
    class torch:
        @staticmethod
        def FloatTensor(*args): return None
        @staticmethod
        def no_grad(): return None
        class optim:
            class Adam:
                def __init__(self, *args, **kwargs): pass

# 🔧 修复utils导入
if PYTORCH_AVAILABLE:
    try:
        from .utils import get_stacked_state
    except ImportError:
        try:
            from utils import get_stacked_state
        except ImportError:
            print("⚠️ 无法导入get_stacked_state，使用内置版本")
            def get_stacked_state(state_history, stack_size=4):
                """简单的状态堆叠函数"""
                if not state_history:
                    return np.zeros(8 * stack_size)
                
                if len(state_history) < stack_size:
                    last_state = state_history[-1]
                    padding = [last_state for _ in range(stack_size - len(state_history))]
                    return np.concatenate(list(state_history) + padding)
                else:
                    return np.concatenate(list(state_history)[-stack_size:])


# ⭐️ 修复1：只保留一个 SumTree 的定义
class SumTree:
    """
    SumTree 数据结构，用于高效地按优先级抽样。
    树的每个叶子节点存储一个经验的优先级，每个父节点是其子节点的优先级之和。
    """
    def __init__(self, capacity):
        import numpy
        self.numpy = numpy
        self.capacity = capacity
        self.tree = self.numpy.zeros(2 * capacity - 1)
        self.data = self.numpy.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        """返回所有优先级的总和（即树的根节点）"""
        return self.tree[0] # ⭐️ 修复2：返回了正确的值


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区 (Prioritized Experience Replay Buffer)。
    这个类封装了SumTree，并提供了与强化学习交互的接口。
    """
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        import numpy
        self.numpy = numpy
        self.tree = SumTree(capacity)
        # ⭐️ 关键修复：abs_err_upper 作为实例变量
        self.abs_err_upper = 1.0

    def add(self, experience):
        max_p = self.numpy.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, experience)

    def sample(self, n):
        batch_indices = self.numpy.empty((n,), dtype=self.numpy.int32)
        batch_data = self.numpy.empty((n,), dtype=object)
        is_weights = self.numpy.empty((n, 1), dtype=self.numpy.float32)

        # 健壮性检查
        if self.tree.size == 0 or self.tree.total_priority == 0:
            return None, None, None

        priority_segment = self.tree.total_priority / n
        self.beta = self.numpy.min([1.0, self.beta + self.beta_increment_per_sampling])

        min_prob = self.numpy.min(self.tree.tree[-self.tree.capacity:]) / (self.tree.total_priority + 1e-7)
        if self.tree.size > 0 and min_prob > 0:
            max_weight = (min_prob * self.tree.size) ** (-self.beta)
        else:
            max_weight = 1.0

        for i in range(n):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            v = self.numpy.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            sampling_prob = p / (self.tree.total_priority + 1e-7)
            if self.tree.size > 0 and sampling_prob > 0:
                is_weights[i, 0] = (sampling_prob * self.tree.size) ** (-self.beta)
            else:
                is_weights[i, 0] = 1.0
            batch_indices[i] = idx
            batch_data[i] = data

        if max_weight > 0:
            is_weights /= max_weight
        else:
            is_weights = self.numpy.ones((n, 1), dtype=self.numpy.float32)

        return batch_indices, batch_data, is_weights

    def batch_update(self, tree_indices, abs_errors):
        """更新优先级 - 最终安全版本"""
        # 强制类型检查和转换
        import numpy as np_local  # 局部导入确保安全
        
        # 确保 abs_errors 是 numpy 数组
        if not isinstance(abs_errors, np_local.ndarray):
            try:
                abs_errors = np_local.array(abs_errors, dtype=np_local.float32)
            except Exception as e:
                print(f"❌ 无法转换 abs_errors 为 numpy 数组: {e}")
                print(f"   abs_errors 类型: {type(abs_errors)}")
                print(f"   abs_errors 内容: {abs_errors}")
                return
        
        # 确保数据类型正确
        abs_errors = abs_errors.astype(np_local.float32)
        
        # 添加 epsilon
        abs_errors = abs_errors + self.epsilon
        
        # 使用局部 numpy 进行安全操作
        clipped_errors = np_local.minimum(abs_errors, self.abs_err_upper)
        ps = np_local.power(clipped_errors, self.alpha)
        
        # 更新优先级
        for ti, p in zip(tree_indices, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return self.tree.size


class SimpleNeuralNetwork(nn.Module):
    """简单的前馈神经网络"""
    def __init__(self, input_size=32, hidden_size=64, output_size=4):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ExperienceReplay:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))
        
    def __len__(self):
        return len(self.memory)


class DuelingDQN(nn.Module):
    """Dueling DQN网络 - 第二步：分离价值流和优势流"""
    def __init__(self, input_size=32, hidden_size=32, output_size=4):  # 改为32，匹配8.22版本
        super(DuelingDQN, self).__init__()
        
        # 共享特征层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # 价值流和优势流
        self.value_head = nn.Linear(hidden_size, 1)
        self.advantage_head = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    """深度Q网络智能体"""
    def __init__(self, state_size=32, action_size=4, lr=0.001, gamma=0.95, 
                 epsilon=0.98, epsilon_min=0.1, epsilon_decay=0.008,
                 memory_size=10000, batch_size=64):  # 改为64，匹配8.22版本
        if not PYTORCH_AVAILABLE:
            print("⚠️ PyTorch不可用，DQN智能体将无法工作")
            return
            
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ExperienceReplay(memory_size)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        
        # 创建主网络和目标网络，使用默认隐藏层大小32
        self.q_network = DuelingDQN(state_size, 32, action_size)
        self.target_network = DuelingDQN(state_size, 32, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 初始化目标网络
        self.update_target_network()
        
    def update_target_network(self):
        """更新目标网络"""
        if PYTORCH_AVAILABLE:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        """选择动作"""
        if not PYTORCH_AVAILABLE:
            return random.randint(0, self.action_size - 1)
            
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def replay(self):
        """训练网络"""
        if not PYTORCH_AVAILABLE or len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def setup_neural_network(state_size=32, action_size=4, lr=0.001):
    """设置神经网络"""
    if not PYTORCH_AVAILABLE:
        print("⚠️ PyTorch不可用，返回空值")
        return None, None
    
    try:
        # 创建网络和优化器
        network = DuelingDQN(state_size, 256, action_size)
        optimizer = optim.Adam(network.parameters(), lr=lr)
        
        print(f"✓ 神经网络设置成功 - 输入: {state_size}, 输出: {action_size}")
        return network, optimizer
        
    except Exception as e:
        print(f"❌ 神经网络设置失败: {e}")
        return None, None


def create_dqn_agent(state_size=32, action_size=4, **kwargs):
    """创建DQN智能体的便捷函数"""
    if not PYTORCH_AVAILABLE:
        print("⚠️ PyTorch不可用，无法创建DQN智能体")
        return None
    
    return DQNAgent(state_size, action_size, **kwargs)


# 🔧 确保模块导出所有必要的类和函数
__all__ = [
    'PYTORCH_AVAILABLE',
    'SumTree', 
    'PrioritizedReplayBuffer',
    'SimpleNeuralNetwork',
    'ExperienceReplay', 
    'DuelingDQN',
    'DQNAgent',
    'setup_neural_network',
    'create_dqn_agent',
    'get_stacked_state'
]

# 🔧 添加模块初始化完成的标志
if PYTORCH_AVAILABLE:
    print("✓ DQN模块完全加载成功")
else:
    print("⚠️ DQN模块以兼容模式加载（无PyTorch支持）")