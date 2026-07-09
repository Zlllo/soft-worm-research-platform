"""
Core smoke tests for the C. elegans simulation.

These tests intentionally avoid Streamlit and Matplotlib so the physics and
learning core can be checked before the UI dependencies are installed.
"""

import os
import sys
import time

import numpy as np


__test__ = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def build_q_learning_worm(width=20, height=20):
    from core.worm_body import Worm2D

    worm = Worm2D(
        start_pos=(5, 5),
        width=width,
        height=height,
        body_params={"num_segments": 3, "segment_length": 3.0},
        noise_params={},
    )
    worm.use_neural = False
    return worm


def test_environment_basics():
    from core.environment import Environment2D
    from core.utils import create_temperature_environment

    temp_array, best_point = create_temperature_environment(20, 20, None, "single_center")
    env = Environment2D(temp_array, best_point=best_point)

    assert_true(env.width == 20 and env.height == 20, "环境尺寸不正确")
    assert_true(best_point == (10, 10), f"单热源最佳点异常: {best_point}")

    center_temp = env.get_temperature(10, 10)
    corner_temp = env.get_temperature(0, 0)
    assert_true(center_temp > corner_temp, "中心温度应高于角落温度")

    state = env.get_state_vector((10, 10))
    assert_true(isinstance(state, np.ndarray), "状态向量应为 numpy.ndarray")
    assert_true(state.shape == (8,), f"状态向量维度应为 8，实际为 {state.shape}")

    print("✓ 环境创建、温度读取、状态向量测试通过")
    return env


def test_worm_creation_and_single_step(env):
    worm = build_q_learning_worm(width=env.width, height=env.height)

    assert_true((worm.x, worm.y) == (5, 5), "线虫初始位置不正确")
    assert_true(len(worm.body_segments) == 3, "线虫身体段数不正确")
    assert_true(len(worm.body_temperatures) == 3, "身体温度数组长度不正确")
    assert_true(worm.use_neural is False, "冒烟测试应使用 Q-Learning 路径")

    perception = worm.perceive_environment(env)
    assert_true("center_temp" in perception, "感知数据缺少 center_temp")

    before_history_len = len(worm.history)
    start = time.time()
    moved = worm.decide_move(env, epsilon=0.5, alpha=0.1, gamma=0.9)
    elapsed = time.time() - start

    assert_true(moved, "线虫单步移动失败")
    assert_true(0 <= worm.x < env.width and 0 <= worm.y < env.height, "线虫移动后越界")
    assert_true(len(worm.history) > before_history_len, "移动后历史记录未更新")
    assert_true(elapsed < 2.0, f"单步移动耗时过长: {elapsed:.3f}s")

    print(f"✓ 线虫创建、感知、单步移动测试通过，耗时 {elapsed:.3f}s")
    return worm


def test_body_model_interface(env):
    from core.worm_body import BodyModel, Worm2DModelAdapter, create_body_model

    body_model = create_body_model(
        model_type="worm2d",
        start_pos=(5, 5),
        width=env.width,
        height=env.height,
        body_params={"num_segments": 3, "segment_length": 3.0},
        noise_params={},
    )

    assert_true(isinstance(body_model, BodyModel), "身体模型应实现 BodyModel 接口")

    observation = body_model.get_observation(env)
    assert_true(observation["position"] == (5, 5), "接口观测位置不正确")
    assert_true(observation["state_vector"].shape == (8,), "接口观测状态向量维度应为 8")

    action_result = body_model.apply_action(3)
    assert_true(action_result["accepted"] is True, "接口动作未被接受")
    physics_result = body_model.step_physics(env)
    assert_true(physics_result["moved"] is True, "接口物理推进失败")

    geometry = body_model.get_geometry()
    assert_true(geometry["type"] == "segmented_polyline", "接口几何类型不正确")
    assert_true(geometry["segment_count"] == 3, "接口几何节段数不正确")

    metrics = body_model.get_metrics(env)
    assert_true(metrics["body_segments"] == 3, "接口指标节段数不正确")
    assert_true(np.isfinite(metrics["head_temperature"]), "接口指标头部温度异常")

    adapter = Worm2DModelAdapter(body_model.worm)
    reset_observation = adapter.reset((5, 5), env=env)
    assert_true(reset_observation["position"] == (5, 5), "兼容包装 reset 后位置不正确")

    print("✓ 身体模型统一接口测试通过")


def test_short_q_learning_training(env):
    from core.utils import reset_worm_for_new_round

    worm = build_q_learning_worm(width=env.width, height=env.height)
    rewards = []
    histories = []

    for round_index in range(2):
        reset_worm_for_new_round(
            worm,
            env,
            start_pos=(5, 5),
            width=env.width,
            height=env.height,
            field_type="single_center",
        )

        for _ in range(10):
            moved = worm.decide_move(env, epsilon=0.4, alpha=0.1, gamma=0.9)
            assert_true(moved, f"第 {round_index + 1} 轮短训练移动失败")

        rewards.append(float(worm.total_reward))
        histories.append(worm.history.copy())

    assert_true(len(rewards) == 2, "短训练奖励记录数量不正确")
    assert_true(all(np.isfinite(reward) for reward in rewards), "短训练奖励包含非有限值")
    assert_true(all(len(history) >= 2 for history in histories), "短训练轨迹记录不足")

    print(f"✓ 2 轮 x 10 步 Q-Learning 短训练测试通过，奖励={rewards}")


def test_dqn_initialization_if_available(env):
    from core.neural_networks import PYTORCH_AVAILABLE
    from core.utils import setup_neural_network

    if not PYTORCH_AVAILABLE:
        print("跳过 DQN 初始化测试：当前环境未安装 PyTorch")
        return

    worm = build_q_learning_worm(width=env.width, height=env.height)
    training_params = {
        "method": "DQN",
        "hidden_size": 16,
        "neural_lr": 0.001,
        "weight_decay": 0.0005,
    }

    result = setup_neural_network(worm, training_params)

    assert_true(result is True, "DQN 初始化函数未返回 True")
    assert_true(worm.use_neural is True, "DQN 初始化后 use_neural 应为 True")
    assert_true(worm.neural_network is not None, "DQN 初始化后 neural_network 不应为空")
    assert_true(worm.experience_replay is not None, "DQN 初始化后 experience_replay 不应为空")

    print(f"✓ DQN 初始化测试通过，网络={type(worm.neural_network).__name__}")


def run_all_tests():
    print("开始核心冒烟测试...")
    env = test_environment_basics()
    test_worm_creation_and_single_step(env)
    test_body_model_interface(env)
    test_short_q_learning_training(env)
    test_dqn_initialization_if_available(env)
    print("全部核心冒烟测试通过")


if __name__ == "__main__":
    run_all_tests()
