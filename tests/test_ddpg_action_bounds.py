"""训练和执行必须使用同一合法连续动作空间。"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '程序'))
from core.actor_critic import DDPGAgent


@pytest.mark.parametrize('bounds', [
    [(-math.pi, math.pi), (0.05, 5.0)],
    [(0.05, 2.0), (0.02, 0.8), (-0.08, 0.08)],
])
def test_training_and_execution_use_bounded_actions(bounds):
    torch.manual_seed(7)
    agent = DDPGAgent(4, action_dim=len(bounds), action_bounds=bounds,
                      hidden_size=8, batch_size=4)
    # 旧实现的 scale 可以把策略推到物理动作范围之外。
    for actor in (agent.actor, agent.actor_target):
        if hasattr(actor, 'action_scales'):
            actor.action_scales.data.fill_(3.0)
        actor.net[-2].weight.data.zero_()
        actor.net[-2].bias.data.fill_(2.0)

    lo = torch.tensor([b[0] for b in bounds])
    hi = torch.tensor([b[1] for b in bounds])
    states = torch.zeros(4, 4)
    predicted = agent.actor(states)
    assert torch.all(predicted >= lo)
    assert torch.all(predicted <= hi)
    np.testing.assert_allclose(agent.act(np.zeros(4), add_noise=False),
                               predicted[0].detach().numpy(), rtol=1e-6, atol=1e-7)
    predicted.sum().backward()
    assert torch.any(agent.actor.net[-2].bias.grad.abs() > 0)

    seen = []
    def check_actions(module, inputs):
        actions = inputs[1].detach()
        assert torch.all(actions >= lo)
        assert torch.all(actions <= hi)
        seen.append(actions.clone())

    handles = [agent.critic.register_forward_pre_hook(check_actions),
               agent.critic_target.register_forward_pre_hook(check_actions)]
    for _ in range(4):
        action = agent.act(np.zeros(4), add_noise=False)
        agent.remember(np.zeros(4), action, 1.0, np.ones(4), False)
    critic_loss, actor_loss = agent.train()
    for handle in handles:
        handle.remove()
    assert len(seen) == 3  # 回放动作、目标策略动作、Actor 优化动作
    assert np.isfinite([critic_loss, actor_loss]).all()
    for actor in (agent.actor, agent.actor_target):
        actions = actor(states)
        assert torch.all(actions >= lo)
        assert torch.all(actions <= hi)


def test_checkpoint_roundtrip_and_legacy_rejection():
    agent = DDPGAgent(4, hidden_size=8)
    saved = agent.state_dict()
    restored = DDPGAgent(4, hidden_size=8)
    restored.load_state_dict(saved)
    np.testing.assert_allclose(agent.act(np.ones(4), add_noise=False),
                               restored.act(np.ones(4), add_noise=False))
    saved['actor']['action_scales'] = torch.ones(2)
    with pytest.raises(ValueError, match='请重新训练'):
        restored.load_state_dict(saved)
