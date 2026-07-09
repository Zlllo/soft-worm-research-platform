import contextlib
import io
import random
import sys
from pathlib import Path

import numpy as np
import pytest


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


from core.environment import Environment2D
from core.utils import (
    create_temperature_environment,
    generate_dynamic_rotating_double_center,
    reset_worm_for_new_round,
    setup_neural_network,
)
from core.worm_body import Worm2D


def build_q_learning_worm(width=20, height=20, start_pos=(5, 5), num_segments=3):
    with contextlib.redirect_stdout(io.StringIO()):
        worm = Worm2D(
            start_pos=start_pos,
            width=width,
            height=height,
            body_params={"num_segments": num_segments, "segment_length": 3.0},
            noise_params={},
        )
    worm.use_neural = False
    return worm


def build_single_center_env(width=20, height=20):
    temp_array, best_point = create_temperature_environment(
        width, height, seed=None, field_type="single_center"
    )
    return Environment2D(temp_array, best_point=best_point)


def assert_point_in_bounds(point, width, height):
    x, y = point
    assert 0 <= x < width
    assert 0 <= y < height


def test_single_center_temperature_field_and_state_vector():
    width, height = 30, 24
    temp_array, best_point = create_temperature_environment(
        width, height, seed=None, field_type="single_center"
    )
    env = Environment2D(temp_array, best_point=best_point)

    assert temp_array.shape == (height, width)
    assert np.isfinite(temp_array).all()
    assert best_point == (15, 12)
    assert_point_in_bounds(best_point, width, height)
    assert env.get_temperature(*best_point) == pytest.approx(float(np.max(temp_array)))
    assert env.get_temperature(*best_point) > env.get_temperature(0, 0)

    state = env.get_state_vector(best_point)
    assert state.shape == (8,)
    assert state.dtype == np.float32
    assert np.isfinite(state).all()


def test_dynamic_temperature_field_changes_and_environment_update():
    width, height = 32, 32
    temp_t0, best_t0 = generate_dynamic_rotating_double_center(
        width, height, t=0, omega_deg=15
    )
    temp_t6, best_t6 = generate_dynamic_rotating_double_center(
        width, height, t=6, omega_deg=15
    )

    assert temp_t0.shape == (height, width)
    assert temp_t6.shape == (height, width)
    assert np.isfinite(temp_t0).all()
    assert np.isfinite(temp_t6).all()
    assert not np.allclose(temp_t0, temp_t6)
    assert best_t0 != best_t6
    assert_point_in_bounds(best_t0, width, height)
    assert_point_in_bounds(best_t6, width, height)

    env = Environment2D(temp_t0, best_point=best_t0)
    env.update_temperature_array(temp_t6, new_best_point=best_t6)
    assert env.width == width
    assert env.height == height
    assert env.best_point == best_t6
    assert np.isfinite(env.get_temperature(*best_t6))

    manual_field = np.zeros((height, width), dtype=np.float32)
    manual_field[3, 5] = 99.0
    env.update_temperature_array(manual_field)
    assert env.best_point == (5, 3)
    assert env.get_temperature(5, 3) == pytest.approx(99.0)


def test_worm_body_creation_respects_body_params():
    worm = build_q_learning_worm(width=20, height=20, start_pos=(7, 8), num_segments=4)

    assert (worm.x, worm.y) == (7, 8)
    assert worm.num_segments == 4
    assert len(worm.body_segments) == 4
    assert len(worm.body_temperatures) == 4
    assert len(worm.q_table) == 20
    assert len(worm.q_table[0]) == 20
    assert worm.history == [worm.body_segments]
    assert all(0 <= x < worm.width and 0 <= y < worm.height for x, y in worm.body_segments)


def test_single_decision_step_updates_motion_and_learning_state():
    random.seed(1)
    np.random.seed(1)
    env = build_single_center_env()
    worm = build_q_learning_worm(width=env.width, height=env.height)

    old_x, old_y = int(worm.x), int(worm.y)
    old_q_values = list(worm.q_table[old_y][old_x])
    old_history_len = len(worm.history)

    moved = worm.decide_move(env, epsilon=1.0, alpha=0.1, gamma=0.9)

    assert moved is True
    assert 0 <= worm.x < env.width
    assert 0 <= worm.y < env.height
    assert worm.body_segments[0] == [worm.x, worm.y]
    assert len(worm.history) > old_history_len
    assert len(worm.recent_temperatures) >= 1
    assert np.isfinite(worm.total_reward)
    assert worm.q_table[old_y][old_x] != old_q_values


def test_short_q_learning_training_two_rounds():
    random.seed(2)
    np.random.seed(2)
    env = build_single_center_env()
    worm = build_q_learning_worm(width=env.width, height=env.height)

    rewards = []
    history_lengths = []
    for _ in range(2):
        reset_worm_for_new_round(
            worm,
            env,
            start_pos=(5, 5),
            width=env.width,
            height=env.height,
            field_type="single_center",
        )
        assert len(worm.state_buffer) == 4

        for _ in range(6):
            assert worm.decide_move(env, epsilon=0.5, alpha=0.1, gamma=0.9)

        rewards.append(worm.total_reward)
        history_lengths.append(len(worm.history))

    assert len(rewards) == 2
    assert np.isfinite(rewards).all()
    assert all(length >= 2 for length in history_lengths)


def test_dqn_initialization_when_pytorch_is_available():
    from core.neural_networks import PYTORCH_AVAILABLE

    if not PYTORCH_AVAILABLE:
        pytest.skip("PyTorch is not installed in this environment")

    import torch

    env = build_single_center_env()
    worm = build_q_learning_worm(width=env.width, height=env.height)
    training_params = {
        "method": "DQN",
        "hidden_size": 16,
        "neural_lr": 0.001,
        "weight_decay": 0.0005,
    }

    assert setup_neural_network(worm, training_params) is True
    assert worm.use_neural is True
    assert worm.neural_network is not None
    assert worm.target_network is not None
    assert worm.experience_replay is not None

    with torch.no_grad():
        output = worm.neural_network(torch.zeros((1, 32), dtype=torch.float32))
    assert tuple(output.shape) == (1, 4)
