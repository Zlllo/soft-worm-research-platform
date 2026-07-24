"""
Actor-Critic (DDPG) 模块 — 用于连续身体模型的连续动作控制。

提供:
- Actor:  状态 → (heading, step_length) 连续动作
- Critic: 状态+动作 → Q值
- 经验回放缓冲区
- Ornstein-Uhlenbeck 探索噪声
"""

import numpy as np
from collections import deque
import random
import math

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False


# ── 网络定义 (仅当 PyTorch 可用时) ────────────────────────

if TORCH_AVAILABLE:

    class Actor(nn.Module):
        """策略网络: state → (heading, step_length)"""

        def __init__(self, state_dim, hidden_size=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2),
                nn.Tanh(),
            )
            # 可学习的缩放因子
            self.heading_scale = nn.Parameter(torch.tensor(math.pi))
            self.step_scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, state):
            raw = self.net(state)
            heading = raw[:, 0] * self.heading_scale          # [-π, π]
            step = (raw[:, 1] + 1.0) / 2.0 * self.step_scale   # [0, step_scale]
            return torch.stack([heading, step], dim=1)


    class Critic(nn.Module):
        """Q网络: state + action → Q值"""

        def __init__(self, state_dim, action_dim=2, hidden_size=128):
            super().__init__()
            self.state_net = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
            )
            self.action_net = nn.Sequential(
                nn.Linear(action_dim, hidden_size),
                nn.ReLU(),
            )
            self.combined = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, state, action):
            s = self.state_net(state)
            a = self.action_net(action)
            return self.combined(torch.cat([s, a], dim=1))

else:
    # PyTorch 不可用时提供占位
    Actor = None
    Critic = None


# ── 经验回放 ──────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ── OU 噪声 ───────────────────────────────────────────────

class OUNoise:
    """Ornstein-Uhlenbeck 过程: 产生时间相关的探索噪声"""

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.3):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(action_dim)

    def reset(self):
        self.state = np.zeros(self.action_dim)

    def sample(self, scale=1.0):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state * scale


# ── DDPG Agent ────────────────────────────────────────────

class DDPGAgent:
    """DDPG 智能体，管理策略学习和动作选择。"""

    def __init__(
        self,
        state_dim,
        action_dim=2,
        hidden_size=128,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.95,
        tau=0.005,
        buffer_capacity=50000,
        batch_size=64,
        train_interval=1,
        noise_theta=0.15,
        noise_sigma=0.3,
        noise_scale=0.6,
        noise_decay=0.9995,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("Actor-Critic 需要 PyTorch")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self.step_count = 0

        # 网络
        self.actor = Actor(state_dim, hidden_size)
        self.actor_target = Actor(state_dim, hidden_size)
        self.critic = Critic(state_dim, action_dim, hidden_size)
        self.critic_target = Critic(state_dim, action_dim, hidden_size)

        # 初始化目标网络权重
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 经验回放和噪声
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.noise = OUNoise(action_dim, theta=noise_theta, sigma=noise_sigma)

        # 模式
        self.train_mode = True

    # ── 动作选择 ──────────────────────────────────────

    def act(self, state, add_noise=True):
        """给定状态，返回 (heading, step_length) 动作。"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).numpy()
        self.actor.train()

        if add_noise and self.train_mode:
            noise = self.noise.sample(self.noise_scale)
            action += noise
            # 衰减噪声
            self.noise_scale = max(0.05, self.noise_scale * self.noise_decay)

        # 裁剪
        heading = float(np.clip(action[0], -math.pi, math.pi))
        step = float(np.clip(action[1], 0.05, 5.0))
        return heading, step

    # ── 经验存储 ──────────────────────────────────────

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    # ── 训练 ──────────────────────────────────────────

    def train(self):
        """执行一次 DDPG 更新。"""
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        self.step_count += 1
        if self.step_count % self.train_interval != 0:
            return None, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # ── 更新 Critic ──
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ── 更新 Actor ──
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ── 软更新目标网络 ──
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(self.tau * s.data + (1.0 - self.tau) * t.data)

    # ── 模式切换 ──────────────────────────────────────

    def eval(self):
        self.train_mode = False

    def train_mode_on(self):
        self.train_mode = True

    # ── 保存/加载 ─────────────────────────────────────

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, d):
        self.actor.load_state_dict(d['actor'])
        self.actor_target.load_state_dict(d['actor_target'])
        self.critic.load_state_dict(d['critic'])
        self.critic_target.load_state_dict(d['critic_target'])
        self.actor_optimizer.load_state_dict(d['actor_optimizer'])
        self.critic_optimizer.load_state_dict(d['critic_optimizer'])
