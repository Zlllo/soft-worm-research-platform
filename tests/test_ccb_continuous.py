"""连续观测、几何约束及训练/评估契约的行为检验。"""
import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '程序'))
with contextlib.redirect_stdout(io.StringIO()):
    from core.environment import Environment2D
    from core.reward_functions import apply_reward_config, compute_reward
    from core.utils import create_temperature_environment, reset_worm_for_new_round
    from core.worm_body import ContinuousCenterlineBody


def body(**params):
    defaults = dict(body_length=12, sample_count=9, ac_hidden_size=16)
    defaults.update(params)
    return ContinuousCenterlineBody((5, 5), 40, 40, defaults, {})


def single_center():
    with contextlib.redirect_stdout(io.StringIO()):
        array, best = create_temperature_environment(40, 40, None, 'single_center')
    return Environment2D(array, best)


def test_interpolation_matches_affine_field_and_changes_with_subpixel_motion():
    y, x = np.mgrid[:8, :10]
    env = Environment2D(20 + 3*x + 2*y)
    for px, py in [(1.1, 2.4), (1.4, 2.4), (8.99, 6.1), (9, 7)]:
        assert env.get_temperature_continuous(px, py) == pytest.approx(20 + 3*px + 2*py)
    assert env.get_temperature(1.1, 2.4) == env.get_temperature(1.4, 2.4)
    assert env.get_state_vector((1.1, 2.4), continuous=True)[4] != env.get_state_vector((1.4, 2.4), continuous=True)[4]
    assert abs(env.get_temperature_continuous(2 + 1e-7, 3.3) - env.get_temperature_continuous(2 - 1e-7, 3.3)) < 1e-5
    assert env.get_temperature_continuous(-0.01, 2) == -float('inf')
    assert env.get_temperature_continuous(float('nan'), 2) == -float('inf')


@pytest.mark.parametrize('samples', [5, 9, 20])
def test_reversals_and_tiny_steps_preserve_lengths_and_turn_limits(samples):
    b = body(sample_count=samples)
    rng = np.random.default_rng(23)
    for step in range(300):
        b.step_physics(action={'heading': rng.uniform(-math.pi, math.pi),
                               'step': 0.01 if step % 2 else rng.uniform(0, 5)})
        points = np.asarray(b.body_segments)
        vectors = np.diff(points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(lengths, 12/(samples-1), atol=1e-8, rtol=0)
        cosines = np.sum(vectors[1:] * vectors[:-1], axis=1) / (lengths[1:] * lengths[:-1])
        angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
        assert angles.max() <= 45 + 1e-6
        assert points.min() >= -1e-8 and points.max() <= 39 + 1e-8


def test_continuous_reward_detects_improvement_inside_old_temperature_band():
    env = Environment2D(np.tile(np.linspace(50, 59, 40), (40, 1)))
    b = body()
    old_temp = env.get_temperature_continuous(10, 10)
    stay = compute_reward(env, b, variant='continuous', old_temperature=old_temp,
                          target=(10, 10), movement=0.1)
    closer = compute_reward(env, b, variant='continuous', old_temperature=old_temp,
                            target=(10.1, 10), movement=0.1)
    assert closer > stay
    assert compute_reward(env, b, target=(10, 10)) == compute_reward(env, b, target=(10.1, 10))


def test_relative_turn_is_continuous_across_angle_seam_and_velocity_is_observed():
    b = body()
    b.heading = math.pi - 0.01
    action = b._build_actor_action((0.02, 0.5))
    difference = (action['heading'] - b.heading + math.pi) % (2*math.pi) - math.pi
    assert difference == pytest.approx(0.02)
    first = b.get_actor_state(single_center())
    b.velocity[:] = [0.1, -0.2]
    second = b.get_actor_state(single_center())
    assert first.shape == second.shape == (14,)
    np.testing.assert_allclose(first[:12], second[:12])
    assert not np.array_equal(first[12:], second[12:])
    assert b.get_state_v2(single_center()).shape == (12,)


def test_goal_is_terminal_and_evaluation_does_not_train():
    torch = pytest.importorskip('torch')
    b = body()
    env = single_center()
    apply_reward_config(b, {'method': 'Actor-Critic'})
    b.setup_actor_critic()
    b.reset((20, 20), env=env)
    agent = b.actor_critic_agent
    # 测试零位移到达终点，不依赖随机初始策略。
    agent.act = lambda *args, **kwargs: (0.0, 0.0)
    b.decide_move_actor_critic(env)
    assert b.goal_reached and b.episode_done
    assert b.current_step == 1
    assert agent.replay_buffer.buffer[-1][-1] is True
    count = len(agent.replay_buffer)
    weights = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    agent.eval()
    b.reset((12, 5), env=env)
    for _ in range(4):
        b.decide_move_actor_critic(env)
    assert not b.goal_reached
    assert len(b.rl_step_records) == 4
    assert len(agent.replay_buffer) == count
    assert b.current_step == 4
    for key, value in agent.actor.state_dict().items():
        assert torch.equal(weights[key], value)


def test_geometry_controller_can_reach_single_source_from_multiple_starts():
    env = single_center()
    b = body()
    for start in [(5, 5), (30, 5), (30, 30), (5, 30)]:
        reset_worm_for_new_round(b, env, start, 40, 40, 'single_center')
        for _ in range(150):
            dx, dy = 20 - b.x, 20 - b.y
            if math.hypot(dx, dy) < 0.75:
                break
            b.step_physics(action={'heading': math.atan2(dy, dx), 'step': min(1.2, math.hypot(dx, dy))})
        assert env.get_distance_to_best(b.x, b.y) < 0.75
        assert b.get_metrics()['body_length_error_abs'] < 1e-8


def test_frozen_evaluation_preserves_training_state_and_noise():
    torch = pytest.importorskip('torch')
    from core.evaluation import evaluate_continuous_policy
    b = body()
    env = single_center()
    b.setup_actor_critic()
    b.reward_variant = 'continuous'
    b.decide_move_actor_critic(env)
    agent = b.actor_critic_agent
    agent.noise.state[:] = [0.2, -0.1]
    position = (b.x, b.y)
    count, scale = len(agent.replay_buffer), agent.noise_scale
    noise = agent.noise.state.copy()
    weights = {key: value.clone() for key, value in agent.actor.state_dict().items()}
    result = evaluate_continuous_policy(b, env, max_steps=3, starts=[(5, 5), (30, 30)])
    assert result['episode_count'] == 2
    assert agent.train_mode is True
    assert (b.x, b.y) == position
    assert len(agent.replay_buffer) == count
    assert agent.noise_scale == scale
    np.testing.assert_allclose(agent.noise.state, noise)
    for key, value in agent.actor.state_dict().items():
        assert torch.equal(value, weights[key])


def test_engine_continues_after_zero_motion_and_exports_evaluation(monkeypatch, tmp_path):
    import json
    import types
    import simulation_engine
    from core.evaluation import evaluate_continuous_policy
    b = body()
    env = single_center()
    b.setup_actor_critic()
    b.reward_variant = 'continuous'
    b.actor_critic_agent.act = lambda *args, **kwargs: (0.0, 0.0)
    monkeypatch.setattr(simulation_engine, 'st', types.SimpleNamespace(session_state={'stop_requested': False}))
    monkeypatch.setattr(simulation_engine, 'create_training_body_model', lambda *a, **kw: b)
    monkeypatch.setattr(simulation_engine, 'create_temperature_environment', lambda *a: (env.temp_array, env.best_point))
    saved = []
    monkeypatch.setattr(simulation_engine, 'save_and_visualize_results', lambda *a, **kw: saved.append(a))
    config = types.SimpleNamespace(output_dir=str(tmp_path))
    params = dict(width=40, height=40, num_rounds=1, steps_per_round=4,
                  initial_epsilon=1, min_epsilon=.1, epsilon_decay=.01,
                  learning_rate=.1, discount_factor=.99, method='Actor-Critic',
                  body_model_type='continuous_centerline', reward_variant='continuous',
                  body_params={'ac_random_starts': False})
    list(simulation_engine.run_standard_simulation_engine(config, params, 'single_center', False))
    assert len(saved) == 1
    assert b.training_round_records[-1]['steps'] == 4
    assert len(b.rl_step_records) == 4


@pytest.mark.parametrize('failure_stage', ['reset', 'step', 'save'])
def test_engine_reports_failure_without_success(monkeypatch, tmp_path, failure_stage):
    import types
    import simulation_engine
    import core.utils
    b = body()
    env = single_center()
    b.setup_actor_critic()
    b.actor_critic_agent.act = lambda *args, **kwargs: (0.0, 0.0)
    monkeypatch.setattr(simulation_engine, 'st', types.SimpleNamespace(session_state={'stop_requested': False}))
    monkeypatch.setattr(simulation_engine, 'create_training_body_model', lambda *a, **kw: b)
    monkeypatch.setattr(simulation_engine, 'create_temperature_environment', lambda *a: (env.temp_array, env.best_point))
    monkeypatch.setattr(simulation_engine, 'save_and_visualize_results', lambda *a, **kw: None)

    def fail(*args, **kwargs):
        raise RuntimeError('expected test failure')

    if failure_stage == 'reset':
        monkeypatch.setattr(core.utils, 'reset_ccb_training_round', fail)
    elif failure_stage == 'step':
        monkeypatch.setattr(b, 'decide_move_actor_critic', fail)
    else:
        monkeypatch.setattr(simulation_engine, 'save_and_visualize_results', fail)
    params = dict(width=40, height=40, num_rounds=1, steps_per_round=2,
                  initial_epsilon=1, min_epsilon=.1, epsilon_decay=.01,
                  learning_rate=.1, discount_factor=.99, method='Actor-Critic',
                  body_model_type='continuous_centerline', reward_variant='continuous',
                  body_params={'ac_random_starts': False})
    events = list(simulation_engine.run_standard_simulation_engine(
        types.SimpleNamespace(output_dir=str(tmp_path)), params, 'single_center', False))
    assert events[-1][0] == -1
    assert 'expected test failure' in events[-1][2]
    assert not any(event[0] == 1000 for event in events)
