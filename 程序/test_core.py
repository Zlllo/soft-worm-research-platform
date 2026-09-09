import contextlib
import importlib
import importlib.util
import io
import json
import math
import random
import sys
import types
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
    save_body_metrics,
    setup_neural_network,
)
from core.worm_body import ActiveDeformationBody, ContinuousCenterlineBody, Worm2D, create_body_model
from core.reward_functions import DEFAULT_ENERGY_WEIGHT, apply_reward_config, compute_reward


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


def test_worm_body_metrics_include_shape_constraints():
    env = build_single_center_env()
    worm = build_q_learning_worm(width=env.width, height=env.height, start_pos=(8, 8), num_segments=4)
    worm.body_segments = [[8, 8], [6, 8], [5, 9], [3, 9]]
    worm.energy = 75.0
    worm.muscle_fatigue_level = 0.25

    metrics = worm.get_metrics(env=env)

    assert metrics["energy"] == pytest.approx(75.0)
    assert metrics["energy_ratio"] == pytest.approx(0.75)
    assert metrics["muscle_fatigue"] == pytest.approx(0.25)
    assert metrics["actual_body_length"] > 0
    assert metrics["target_body_length"] > 0
    assert "body_length_error_abs" in metrics
    assert "curvature_mean_deg" in metrics
    assert "curvature_max_deg" in metrics
    assert "constraint_violation_rate" in metrics
    assert metrics["distance_to_best"] >= 0


def test_continuous_centerline_body_preserves_length_and_reports_metrics():
    env = build_single_center_env(width=30, height=30)
    body = create_body_model(
        model_type="continuous_centerline",
        start_pos=(12, 12),
        width=30,
        height=30,
        body_params={
            "sample_count": 7,
            "body_length": 12.0,
            "forward_speed": 2.0,
            "damping": 0.0,
            "curvature_limit_deg": 55.0,
        },
        noise_params={},
    )

    assert isinstance(body, ContinuousCenterlineBody)
    geometry = body.get_geometry()
    assert geometry["type"] == "continuous_centerline"
    assert geometry["sample_count"] == 7
    assert len(geometry["centerline"]) == 7

    body.apply_action({"heading": 0.0, "step": 2.0})
    result = body.step_physics(env=env)
    metrics = body.get_metrics(env=env)

    assert result["moved"] is True
    assert metrics["model"] == "continuous_centerline"
    assert metrics["actual_body_length"] == pytest.approx(12.0, abs=1.0)
    assert metrics["body_length_error_abs"] < 1.0
    assert metrics["curvature_max_deg"] <= metrics["curvature_limit_deg"] + 1e-6
    assert metrics["energy"] < body.max_energy
    assert len(body.history) == 2


def test_active_deformation_body_advances_wave_and_factory_alias():
    env = build_single_center_env(width=32, height=32)
    body = create_body_model(
        model_type="active_wave",
        start_pos=(14, 14),
        width=32,
        height=32,
        body_params={
            "sample_count": 9,
            "body_length": 14.0,
            "wave_amplitude": 1.4,
            "wave_frequency": 0.5,
            "wave_phase": 0.0,
            "wave_speed": 1.0,
            "damping": 0.0,
        },
        noise_params={},
    )

    assert isinstance(body, ActiveDeformationBody)
    initial_phase = body.wave_phase
    initial_geometry = body.get_geometry()["centerline"]

    body.apply_action({"heading": 0.0, "step": 1.0, "wave_amplitude": 1.2, "wave_frequency": 0.4})
    result = body.step_physics(env=env, dt=1.0)
    metrics = body.get_metrics(env=env)
    updated_geometry = body.get_geometry()["centerline"]

    assert result["moved"] is True
    assert body.wave_phase > initial_phase
    assert metrics["model"] == "active_deformation"
    assert metrics["wave_amplitude"] == pytest.approx(1.2)
    assert metrics["wave_frequency"] == pytest.approx(0.4)
    assert metrics["actual_body_length"] == pytest.approx(14.0, abs=1.5)
    assert metrics["energy"] < body.max_energy
    assert updated_geometry != initial_geometry


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


def test_save_body_metrics_writes_comparison_json(tmp_path):
    env = build_single_center_env()
    worm = build_q_learning_worm(width=env.width, height=env.height, start_pos=(5, 5), num_segments=4)
    worm.body_segments = [[5, 5], [4, 5], [3, 6], [2, 6]]
    worm.history = [worm.body_segments.copy()]
    worm.total_reward = 3.5

    config = types.SimpleNamespace(
        output_dir=str(tmp_path),
        experiment_name="pytest_body_metrics",
        field_type="single_center",
        timestamp="2026-07-09T00:00:00",
    )

    metrics_file = save_body_metrics(config, [worm.history], [worm.total_reward], worm, env)

    with open(metrics_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["experiment_name"] == "pytest_body_metrics"
    assert payload["model"] == "worm2d"
    assert payload["final_metrics"]["energy"] == pytest.approx(worm.energy)
    assert "body_length_error_abs" in payload["final_metrics"]
    assert "curvature_max_deg" in payload["final_metrics"]
    assert "muscle_fatigue" in payload["final_metrics"]
    assert payload["summary"]["round_count"] == 1
    assert payload["round_metrics"][0]["reward"] == pytest.approx(3.5)


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

    input_dim = worm.state_size * 4  # 帧维度 × 4 帧堆叠
    with torch.no_grad():
        output = worm.neural_network(torch.zeros((1, input_dim), dtype=torch.float32))
    assert tuple(output.shape) == (1, 4)


def _install_streamlit_and_matplotlib_stubs(monkeypatch):
    streamlit_stub = types.SimpleNamespace(session_state={})
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)

    if importlib.util.find_spec("matplotlib") is not None:
        return streamlit_stub

    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.rcParams = {}
    matplotlib_stub.use = lambda *args, **kwargs: None

    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    pyplot_stub.rcParams = matplotlib_stub.rcParams
    pyplot_stub.cm = types.SimpleNamespace(
        viridis=lambda value: value,
        get_cmap=lambda name: (lambda value: value),
    )

    def noop(*args, **kwargs):
        return types.SimpleNamespace()

    pyplot_stub.figure = noop
    pyplot_stub.subplots = lambda *args, **kwargs: (noop(), noop())
    pyplot_stub.close = noop
    pyplot_stub.savefig = noop
    pyplot_stub.tight_layout = noop
    pyplot_stub.colorbar = noop
    pyplot_stub.subplot = noop
    pyplot_stub.imshow = noop
    pyplot_stub.plot = noop
    pyplot_stub.scatter = noop
    pyplot_stub.title = noop
    pyplot_stub.xlabel = noop
    pyplot_stub.ylabel = noop
    pyplot_stub.legend = noop
    pyplot_stub.grid = noop
    pyplot_stub.hist = noop

    font_manager_stub = types.ModuleType("matplotlib.font_manager")
    font_manager_stub.fontManager = types.SimpleNamespace(ttflist=[])

    animation_stub = types.ModuleType("matplotlib.animation")
    animation_stub.FuncAnimation = lambda *args, **kwargs: types.SimpleNamespace(save=noop)
    animation_stub.FFMpegWriter = lambda *args, **kwargs: noop()

    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_stub)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot_stub)
    monkeypatch.setitem(sys.modules, "matplotlib.font_manager", font_manager_stub)
    monkeypatch.setitem(sys.modules, "matplotlib.animation", animation_stub)
    return streamlit_stub


def _build_engine_config(tmp_path, name):
    return types.SimpleNamespace(
        experiment_name=name,
        output_dir=str(tmp_path),
        log_file=str(tmp_path / "experiment.log"),
        results_image=str(tmp_path / "training_results.png"),
        animation_gif=str(tmp_path / "training_animation.gif"),
        animation_video=str(tmp_path / "training_animation.mp4"),
        q_table_file=str(tmp_path / "q_table.txt"),
    )


def _build_engine_training_params(width=16, height=16):
    return {
        "width": width,
        "height": height,
        "num_rounds": 1,
        "steps_per_round": 3,
        "initial_epsilon": 0.2,
        "min_epsilon": 0.1,
        "epsilon_decay": 0.01,
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "body_params": {"num_segments": 3, "segment_length": 3.0},
        "noise_params": {},
        "method": "Q-Learning",
    }


def _spy_body_model_factory(monkeypatch, simulation_engine):
    original_create_body_model = simulation_engine.create_body_model
    factory_calls = []

    def spy_create_body_model(*args, **kwargs):
        body_model = original_create_body_model(*args, **kwargs)
        factory_calls.append({"args": args, "kwargs": kwargs, "body_model": body_model})
        return body_model

    monkeypatch.setattr(simulation_engine, "create_body_model", spy_create_body_model)
    return factory_calls


def test_standard_simulation_engine_uses_body_model_factory(monkeypatch, tmp_path):
    _install_streamlit_and_matplotlib_stubs(monkeypatch)
    simulation_engine = importlib.import_module("simulation_engine")
    monkeypatch.setattr(
        simulation_engine,
        "st",
        types.SimpleNamespace(session_state={"stop_requested": False}),
    )

    from core.worm_body import Worm2DModelAdapter

    factory_calls = _spy_body_model_factory(monkeypatch, simulation_engine)
    saved_results = []

    def record_save_results(config, all_histories, all_rewards, worm, env, training_params, temp_array, best_point):
        saved_results.append(
            {
                "rounds": len(all_rewards),
                "histories": len(all_histories),
                "worm": worm,
                "env_size": (env.width, env.height),
                "best_point": best_point,
            }
        )

    monkeypatch.setattr(simulation_engine, "save_and_visualize_results", record_save_results)

    config = _build_engine_config(tmp_path, "pytest_standard_engine")
    training_params = _build_engine_training_params()

    with contextlib.redirect_stdout(io.StringIO()):
        events = list(
            simulation_engine.run_standard_simulation_engine(
                config=config,
                training_params=training_params,
                field_type="single_center",
                use_neural_network=False,
                enable_step_tracking=False,
            )
        )

    assert factory_calls, "标准训练入口没有调用身体模型工厂"
    factory_kwargs = factory_calls[0]["kwargs"]
    assert factory_kwargs["model_type"] == "worm2d"
    assert factory_kwargs["width"] == training_params["width"]
    assert factory_kwargs["height"] == training_params["height"]
    assert factory_kwargs["body_params"] == training_params["body_params"]
    assert isinstance(factory_calls[0]["body_model"], Worm2DModelAdapter)

    assert saved_results, "标准训练没有进入结果保存阶段"
    assert saved_results[0]["rounds"] == 1
    assert saved_results[0]["histories"] == 1
    assert saved_results[0]["env_size"] == (16, 16)
    assert isinstance(saved_results[0]["worm"], Worm2DModelAdapter)

    assert events[-1][0:2] == (1000, 1000)
    assert events[-1][3]["total_rounds"] == 1
    assert any(event[3].get("round") == 1 for event in events if isinstance(event[3], dict))


def test_standard_simulation_engine_runs_with_continuous_centerline_body(monkeypatch, tmp_path):
    _install_streamlit_and_matplotlib_stubs(monkeypatch)
    simulation_engine = importlib.import_module("simulation_engine")
    monkeypatch.setattr(
        simulation_engine,
        "st",
        types.SimpleNamespace(session_state={"stop_requested": False}),
    )

    factory_calls = _spy_body_model_factory(monkeypatch, simulation_engine)
    saved_results = []

    def record_save_results(config, all_histories, all_rewards, worm, env, training_params, temp_array, best_point):
        saved_results.append(
            {
                "rounds": len(all_rewards),
                "histories": len(all_histories),
                "worm": worm,
                "metrics": worm.get_metrics(env=env),
            }
        )

    monkeypatch.setattr(simulation_engine, "save_and_visualize_results", record_save_results)

    config = _build_engine_config(tmp_path, "pytest_continuous_engine")
    training_params = _build_engine_training_params(width=24, height=24)
    training_params.update(
        {
            "body_model_type": "continuous_centerline",
            "steps_per_round": 2,
            "body_params": {
                "sample_count": 7,
                "body_length": 10.0,
                "forward_speed": 1.0,
                "damping": 0.0,
            },
        }
    )

    with contextlib.redirect_stdout(io.StringIO()):
        events = list(
            simulation_engine.run_standard_simulation_engine(
                config=config,
                training_params=training_params,
                field_type="spotty_field",
                use_neural_network=False,
                enable_step_tracking=False,
            )
        )

    assert factory_calls[0]["kwargs"]["model_type"] == "continuous_centerline"
    assert isinstance(factory_calls[0]["body_model"], ContinuousCenterlineBody)
    assert saved_results[0]["rounds"] == 1
    assert saved_results[0]["histories"] == 1
    assert saved_results[0]["metrics"]["model"] == "continuous_centerline"
    assert saved_results[0]["metrics"]["body_length_error_abs"] < 2.0
    assert events[-1][0:2] == (1000, 1000)


def test_transfer_simulation_engine_uses_body_model_factory(monkeypatch, tmp_path):
    _install_streamlit_and_matplotlib_stubs(monkeypatch)
    simulation_engine = importlib.import_module("simulation_engine")
    monkeypatch.setattr(
        simulation_engine,
        "st",
        types.SimpleNamespace(session_state={"stop_requested": False}),
    )

    from core.worm_body import Worm2DModelAdapter

    factory_calls = _spy_body_model_factory(monkeypatch, simulation_engine)
    saved_results = []

    def record_save_results(config, all_histories, all_rewards, worm, env, training_params, temp_array, best_point):
        saved_results.append(
            {
                "rounds": len(all_rewards),
                "histories": len(all_histories),
                "worm": worm,
                "env_size": (env.width, env.height),
            }
        )

    monkeypatch.setattr(simulation_engine, "save_and_visualize_results", record_save_results)

    config = _build_engine_config(tmp_path, "pytest_transfer_engine")
    training_params = _build_engine_training_params(width=80, height=80)
    training_params["steps_per_round"] = 1

    with contextlib.redirect_stdout(io.StringIO()):
        events = list(
            simulation_engine.run_transfer_simulation_engine(
                config=config,
                training_params=training_params,
                source_field="single_center",
                target_field="dual_center",
                use_neural_network=False,
            )
        )

    assert len(factory_calls) == 2
    assert [call["kwargs"]["model_type"] for call in factory_calls] == ["worm2d", "worm2d"]
    assert all(call["kwargs"]["body_params"] == training_params["body_params"] for call in factory_calls)
    assert all(isinstance(call["body_model"], Worm2DModelAdapter) for call in factory_calls)

    assert saved_results, "迁移学习没有进入结果保存阶段"
    assert saved_results[0]["rounds"] == 20
    assert saved_results[0]["histories"] == 20
    assert isinstance(saved_results[0]["worm"], Worm2DModelAdapter)
    assert events[-1][0:2] == (1200, 1200)
    assert events[-1][3]["source_field"] == "single_center"
    assert events[-1][3]["target_field"] == "dual_center"


def test_curriculum_simulation_engine_uses_body_model_factory(monkeypatch, tmp_path):
    _install_streamlit_and_matplotlib_stubs(monkeypatch)
    simulation_engine = importlib.import_module("simulation_engine")

    class StopAfterFirstStageCheck:
        def __init__(self):
            self.calls = 0

        def get(self, key, default=None):
            if key != "stop_requested":
                return default
            self.calls += 1
            return self.calls > 1

    monkeypatch.setattr(
        simulation_engine,
        "st",
        types.SimpleNamespace(session_state=StopAfterFirstStageCheck()),
    )

    from core.worm_body import Worm2DModelAdapter

    factory_calls = _spy_body_model_factory(monkeypatch, simulation_engine)
    monkeypatch.setattr(simulation_engine, "save_and_visualize_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        simulation_engine,
        "setup_neural_network",
        lambda worm, training_params: setattr(worm, "use_neural", True) or True,
    )

    def fake_zero_shot_testing_engine(*args, **kwargs):
        yield 1, 1, "fake zero-shot test", {"phase": "zero_shot_testing"}
        return [], [], {"final_performance": 0.0}

    monkeypatch.setattr(
        simulation_engine,
        "zero_shot_testing_engine",
        fake_zero_shot_testing_engine,
    )

    config = _build_engine_config(tmp_path, "pytest_curriculum_engine")
    training_params = _build_engine_training_params(width=16, height=16)
    training_params.update({"hidden_size": 16, "neural_lr": 0.001})

    with contextlib.redirect_stdout(io.StringIO()):
        events = list(
            simulation_engine.run_curriculum_simulation_engine(
                config=config,
                training_params=training_params,
                test_stage_idx=0,
                env_size=16,
                enable_step_tracking=False,
            )
        )

    assert len(factory_calls) == 2
    assert [call["kwargs"]["model_type"] for call in factory_calls] == ["worm2d", "worm2d"]
    assert all(call["kwargs"]["body_params"] == training_params["body_params"] for call in factory_calls)
    assert all(isinstance(call["body_model"], Worm2DModelAdapter) for call in factory_calls)

    assert events[-1][0] == events[-1][1]
    assert events[-1][3]["total_training_stages"] >= 1


# ══════════════════════════════════════════════════════════════════════════════
# 奖励函数模块测试 (core/reward_functions.py)
# ══════════════════════════════════════════════════════════════════════════════

def _reference_original_reward(env, worm, target):
    """Worm2D 原始奖励的独立参考实现（独立重写，用于逐值对比）。"""
    tx, ty = int(round(target[0])), int(round(target[1]))
    new_temp = env.get_temperature(tx, ty)

    if new_temp == -float('inf'):
        return -10.0

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

    if len(worm.recent_temperatures) >= 2:
        recent_avg = sum(worm.recent_temperatures[-2:]) / 2
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


def test_reward_original_variant_matches_worm2d_reference():
    """模糊测试：original 变体与 Worm2D 原始奖励逐值相等（含梯度分支与出界）。"""
    env = build_single_center_env(width=20, height=20)
    worm = build_q_learning_worm(width=20, height=20)
    rng = random.Random(42)
    for _ in range(300):
        worm.recent_temperatures = [rng.uniform(0.0, 150.0) for _ in range(rng.randint(0, 6))]
        target = (rng.uniform(-5.0, 25.0), rng.uniform(-5.0, 25.0))
        expected = _reference_original_reward(env, worm, target)
        actual = compute_reward(env, worm, variant="original", target=target)
        assert actual == expected, f"target={target}, temps={worm.recent_temperatures}: {actual} != {expected}"


def test_reward_energy_variant_formula():
    """能量变体数学：reward = 阶梯 + (-w_e · max(0, oldE − E) / maxE)。"""
    env = build_single_center_env(width=20, height=20)
    worm = build_q_learning_worm(width=20, height=20)
    worm.recent_temperatures = []
    worm.max_energy = 100.0
    worm.energy = 70.0
    target = (5, 5)

    base = _reference_original_reward(env, worm, target)
    w = 0.25
    # 消耗 30 → 惩罚 w·30/100
    reward = compute_reward(env, worm, variant="energy", old_energy=100.0, energy_weight=w, target=target)
    assert reward == pytest.approx(base - w * (100.0 - 70.0) / 100.0)
    # 无消耗 → 无惩罚
    reward2 = compute_reward(env, worm, variant="energy", old_energy=70.0, energy_weight=w, target=target)
    assert reward2 == pytest.approx(base)
    # 能量反而增加 → max(0, ·) 截断，无惩罚
    reward3 = compute_reward(env, worm, variant="energy", old_energy=50.0, energy_weight=w, target=target)
    assert reward3 == pytest.approx(base)
    # original 变体完全忽略能量
    reward4 = compute_reward(env, worm, variant="original", old_energy=100.0, target=target)
    assert reward4 == pytest.approx(base)


def test_energy_variant_inert_for_worm2d():
    """Worm2D 能量从不衰减 → 含能量变体与原版逐值相等。"""
    env = build_single_center_env(width=20, height=20)
    worm = build_q_learning_worm(width=20, height=20)
    rng = random.Random(7)
    for _ in range(150):
        worm.recent_temperatures = [rng.uniform(0.0, 150.0) for _ in range(rng.randint(0, 6))]
        target = (rng.randint(0, 19), rng.randint(0, 19))
        base = compute_reward(env, worm, variant="original", target=target)
        variant = compute_reward(
            env, worm, variant="energy", old_energy=worm.energy,
            energy_weight=rng.uniform(0.0, 0.5), target=target,
        )
        assert variant == base


def test_reward_stuck_and_constraint_penalties():
    """AC 路径的 stuck / constraint 惩罚按参数生效。"""
    env = build_single_center_env(width=20, height=20)
    worm = build_q_learning_worm(width=20, height=20)
    worm.recent_temperatures = []
    target = (5, 5)
    base = compute_reward(env, worm, variant="original", target=target)

    # stuck: 位移 < 1e-9 时扣除
    stuck = compute_reward(env, worm, variant="original", target=target,
                           stuck_penalty=0.5, movement=0.0)
    assert stuck == pytest.approx(base - 0.5)
    # 正常移动不扣
    moving = compute_reward(env, worm, variant="original", target=target,
                            stuck_penalty=0.5, movement=2.0)
    assert moving == pytest.approx(base)
    # constraint: 违例惩罚直接扣除
    constrained = compute_reward(env, worm, variant="original", target=target,
                                 constraint_penalty=0.25)
    assert constrained == pytest.approx(base - 0.25)
    # 与能量惩罚叠加
    combined = compute_reward(env, worm, variant="energy", old_energy=100.0,
                              energy_weight=0.1, stuck_penalty=0.5, movement=0.0,
                              constraint_penalty=0.25, target=target)
    expected = base - 0.1 * (100.0 - worm.energy) / 100.0 - 0.5 - 0.25
    assert combined == pytest.approx(expected)


def test_apply_reward_config_sets_worm_attributes():
    """apply_reward_config 装配变体到各种身体模型，默认值正确。"""
    worm = build_q_learning_worm(width=20, height=20)
    cc = create_body_model(
        model_type="continuous_centerline", start_pos=(5, 5), width=20, height=20,
        body_params={"sample_count": 5, "body_length": 8.0}, noise_params={},
    )

    apply_reward_config(worm, {})
    assert worm.reward_variant == "original"
    assert worm.reward_energy_weight == DEFAULT_ENERGY_WEIGHT

    apply_reward_config(worm, {"reward_variant": "energy", "reward_energy_weight": 0.3})
    assert worm.reward_variant == "energy"
    assert worm.reward_energy_weight == 0.3

    apply_reward_config(cc, {"reward_variant": "energy"})
    assert cc.reward_variant == "energy"
    assert cc.reward_energy_weight == DEFAULT_ENERGY_WEIGHT


def test_ccb_q_learning_runs_with_module_reward():
    """CCB Q-learning 路径走统一奖励函数模块（含能量变体不报错、奖励有限）。"""
    random.seed(3)
    np.random.seed(3)
    env = build_single_center_env(width=20, height=20)
    body = create_body_model(
        model_type="continuous_centerline", start_pos=(5, 5), width=20, height=20,
        body_params={"sample_count": 5, "body_length": 8.0, "damping": 0.0}, noise_params={},
    )
    apply_reward_config(body, {"reward_variant": "energy", "reward_energy_weight": 0.1})
    assert body.reward_variant == "energy"

    moved = body.decide_move(env, epsilon=1.0, alpha=0.1, gamma=0.9)
    assert moved is True
    assert np.isfinite(body.total_reward)


# ══════════════════════════════════════════════════════════════════════════════
# 动作空间扩展测试 (8 方向; 4 方向保持历史行为)
# ══════════════════════════════════════════════════════════════════════════════

EIGHT_DIR_VECTORS = [
    (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
]  # 顺时针: 上 右上 右 右下 下 左下 左 左上


def build_q_learning_worm_8dir(width=20, height=20, start_pos=(5, 5), num_segments=3):
    with contextlib.redirect_stdout(io.StringIO()):
        worm = Worm2D(
            start_pos=start_pos,
            width=width,
            height=height,
            body_params={"num_segments": num_segments, "segment_length": 3.0, "action_size": 8},
            noise_params={},
        )
    worm.use_neural = False
    return worm


def test_eight_direction_worm2d_action_space():
    """8 方向：Q 表宽 8，apply_action 边界正确 (7 合法, 8/-1 越界)。"""
    worm = build_q_learning_worm_8dir()
    assert worm.action_size == 8
    assert all(len(row[0]) == 8 for row in worm.q_table)

    assert worm.apply_action(7)["accepted"] is True
    with pytest.raises(ValueError):
        worm.apply_action(8)
    with pytest.raises(ValueError):
        worm.apply_action(-1)


def test_eight_direction_moves_and_diagonal_landing():
    """8 方向移动：顺时针落点逐一正确，对角整数格点 (不缩放)。"""
    env = build_single_center_env(width=20, height=20)
    for action, (dx, dy) in enumerate(EIGHT_DIR_VECTORS):
        worm = build_q_learning_worm_8dir(start_pos=(10, 10))
        worm.move(action, env)
        assert (worm.x - 10, worm.y - 10) == (dx, dy), f"action={action} 落点错误"


def test_four_direction_legacy_order_unchanged():
    """4 方向回归：默认参数下动作编号与历史版本完全一致 (0上 1下 2左 3右)。"""
    env = build_single_center_env(width=20, height=20)
    legacy = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    worm = build_q_learning_worm()
    assert worm.action_size == 4
    assert all(len(row[0]) == 4 for row in worm.q_table)
    for action, (dx, dy) in enumerate(legacy):
        worm.x, worm.y = 10, 10
        worm.move(action, env)
        assert (worm.x - 10, worm.y - 10) == (dx, dy), f"action={action} 落点错误"


def test_eight_direction_ccb_headings_and_q_learning():
    """CCB 8 方向：朝向按 45° 顺时针编号，Q-learning 冒烟正常。"""
    env = build_single_center_env(width=20, height=20)
    body = create_body_model(
        model_type="continuous_centerline", start_pos=(5, 5), width=20, height=20,
        body_params={"sample_count": 5, "body_length": 8.0, "damping": 0.0, "action_size": 8},
        noise_params={},
    )
    assert body.action_size == 8
    assert all(len(row[0]) == 8 for row in body.q_table)

    expected = [i * 2.0 * np.pi / 8 - np.pi / 2.0 for i in range(8)]
    for action, angle in enumerate(expected):
        heading, _ = body._parse_action(action)
        assert heading == pytest.approx(angle, abs=1e-12)

    random.seed(4)
    np.random.seed(4)
    assert body.decide_move(env, epsilon=1.0, alpha=0.1, gamma=0.9) is True
    assert np.isfinite(body.total_reward)


def test_eight_direction_dqn_output_dim():
    """8 方向 + DQN：网络输出维度 8 (输入维度不变)。"""
    from core.neural_networks import PYTORCH_AVAILABLE

    if not PYTORCH_AVAILABLE:
        pytest.skip("PyTorch is not installed in this environment")

    import torch

    worm = build_q_learning_worm_8dir()
    training_params = {
        "method": "DQN",
        "hidden_size": 16,
        "neural_lr": 0.001,
        "weight_decay": 0.0005,
    }
    assert setup_neural_network(worm, training_params) is True
    assert worm.use_neural is True

    input_dim = worm.state_size * 4
    with torch.no_grad():
        output = worm.neural_network(torch.zeros((1, input_dim), dtype=torch.float32))
    assert tuple(output.shape) == (1, 8)


def test_save_q_table_dynamic_columns(tmp_path):
    """save_q_table：8 方向输出 8 列并带顺时针标签。"""
    env = build_single_center_env(width=20, height=20)
    worm = build_q_learning_worm_8dir()
    config = types.SimpleNamespace(
        experiment_name="pytest_qtable_8dir",
        q_table_file=str(tmp_path / "q_table.txt"),
    )

    from core.utils import save_q_table

    save_q_table(config, worm.q_table, env)

    with open(config.q_table_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header = lines[1].strip().split("\t")
    assert header[0] == "位置(x,y)"
    assert header[1] == "温度"
    assert header[2:10] == ["上Q", "右上Q", "右Q", "右下Q", "下Q", "左下Q", "左Q", "左上Q"]
    assert header[10] == "偏好动作"
    assert len(lines) == 3 + env.width * env.height


# ══════════════════════════════════════════════════════════════════════════════
# CCB/ADB DQN / Dueling DQN 通路测试
# ══════════════════════════════════════════════════════════════════════════════

def build_ccb_body(width=20, height=20, start_pos=(5, 5), **body_params):
    params = {"sample_count": 5, "body_length": 8.0, "damping": 0.0}
    params.update(body_params)
    with contextlib.redirect_stdout(io.StringIO()):
        body = create_body_model(
            model_type="continuous_centerline", start_pos=start_pos,
            width=width, height=height, body_params=params, noise_params={},
        )
    return body


def test_ccb_state_v2_has_heading_and_12_dims():
    """CCB state_v2 = 12 维，末两位为 cos/sin 朝向；ac_state_dim/state_size 同步 12。"""
    env = build_single_center_env(width=20, height=20)
    body = build_ccb_body()

    state = body.get_state_v2(env)
    assert state.shape == (12,)
    assert body.ac_state_dim == 12
    assert body.state_size == 12
    assert state[10] == pytest.approx(math.cos(body.heading), abs=1e-6)
    assert state[11] == pytest.approx(math.sin(body.heading), abs=1e-6)

    # 转向后 cos/sin 跟随变化
    body.heading = math.pi / 2.0
    state2 = body.get_state_v2(env)
    assert state2[10] == pytest.approx(0.0, abs=1e-6)
    assert state2[11] == pytest.approx(1.0, abs=1e-6)


def test_ccb_dqn_setup_dimensions():
    """CCB + DQN：输入 12×4=48 维，输出 = action_size。"""
    from core.neural_networks import PYTORCH_AVAILABLE

    if not PYTORCH_AVAILABLE:
        pytest.skip("PyTorch is not installed in this environment")

    import torch

    body = build_ccb_body()
    params = {"method": "DQN", "hidden_size": 16, "neural_lr": 0.001, "weight_decay": 0.0005}
    assert setup_neural_network(body, params) is True
    assert body.use_neural is True
    assert body.state_size == 12

    with torch.no_grad():
        output = body.neural_network(torch.zeros((1, 48), dtype=torch.float32))
    assert tuple(output.shape) == (1, 4)  # 默认 action_size=4

    body8 = build_ccb_body(action_size=8)
    setup_neural_network(body8, params)
    with torch.no_grad():
        output8 = body8.neural_network(torch.zeros((1, 48), dtype=torch.float32))
    assert tuple(output8.shape) == (1, 8)


def test_ccb_dqn_decide_move_smoke_and_training():
    """CCB DQN 分支：随机动作 75 步 → 经验入库、至少一次批训练、Q 表未被写。"""
    from core.neural_networks import PYTORCH_AVAILABLE

    if not PYTORCH_AVAILABLE:
        pytest.skip("PyTorch is not installed in this environment")

    random.seed(5)
    np.random.seed(5)
    env = build_single_center_env(width=20, height=20)
    body = build_ccb_body()
    setup_neural_network(
        body, {"method": "DQN", "hidden_size": 16, "neural_lr": 0.001, "weight_decay": 0.0005}
    )

    moved_count = 0
    for _ in range(75):
        if body.decide_move(env, epsilon=1.0, alpha=0.1, gamma=0.9):
            moved_count += 1
    assert moved_count >= 50  # 随机游走会撞墙（位移0），允许少数夹停步

    assert np.isfinite(body.total_reward)
    assert len(body.experience_replay) >= 64
    assert body.step_count >= 1  # 触发过批训练
    assert all(all(all(q == 0.0 for q in x_row) for x_row in y_row) for y_row in body.q_table)  # Q 表未被 DQN 路径写入


def test_ccb_dueling_variant_network_class():
    """CCB + Dueling DQN：主网络和目标网络都是 DuelingDQN。"""
    from core.neural_networks import DuelingDQN, PYTORCH_AVAILABLE

    if not PYTORCH_AVAILABLE:
        pytest.skip("PyTorch is not installed in this environment")

    body = build_ccb_body()
    setup_neural_network(
        body, {"method": "Dueling DQN", "hidden_size": 16, "neural_lr": 0.001, "weight_decay": 0.0005}
    )
    assert isinstance(body.neural_network, DuelingDQN)
    assert isinstance(body.target_network, DuelingDQN)


def test_standard_simulation_engine_ccb_dqn(monkeypatch, tmp_path):
    """引擎级：CCB + DQN 短训练完整走到结果保存阶段。"""
    from core.neural_networks import PYTORCH_AVAILABLE

    if not PYTORCH_AVAILABLE:
        pytest.skip("PyTorch is not installed in this environment")

    _install_streamlit_and_matplotlib_stubs(monkeypatch)
    simulation_engine = importlib.import_module("simulation_engine")
    monkeypatch.setattr(
        simulation_engine,
        "st",
        types.SimpleNamespace(session_state={"stop_requested": False}),
    )

    factory_calls = _spy_body_model_factory(monkeypatch, simulation_engine)
    saved_results = []
    monkeypatch.setattr(
        simulation_engine, "save_and_visualize_results",
        lambda *args, **kwargs: saved_results.append(args) or None,
    )

    config = _build_engine_config(tmp_path, "pytest_ccb_dqn_engine")
    training_params = _build_engine_training_params(width=16, height=16)
    training_params.update({
        "method": "DQN",
        "hidden_size": 16,
        "neural_lr": 0.001,
        "weight_decay": 0.0005,
        "body_model_type": "continuous_centerline",
        "body_params": {"sample_count": 5, "body_length": 8.0, "damping": 0.0},
        "steps_per_round": 2,
    })

    with contextlib.redirect_stdout(io.StringIO()):
        events = list(
            simulation_engine.run_standard_simulation_engine(
                config=config,
                training_params=training_params,
                field_type="single_center",
                use_neural_network=True,
                enable_step_tracking=False,
            )
        )

    assert factory_calls[0]["kwargs"]["model_type"] == "continuous_centerline"
    assert saved_results, "CCB + DQN 训练没有进入结果保存阶段"
    assert events[-1][0:2] == (1000, 1000)


# ── 2026-09-02: ADB RFT 力基波驱动 (3×3 力/力矩平衡 + 子步积分 + 课程训练) ──────


def build_adb_body(width=200, height=200, start_pos=(100, 100), **body_params):
    params = {"sample_count": 9, "body_length": 12.0, "wave_amplitude": 1.0,
              "wave_frequency": 0.25, "steer_bias": 0.0, "sub_steps": 10}
    params.update(body_params)
    with contextlib.redirect_stdout(io.StringIO()):
        body = create_body_model(
            model_type="active_deformation", start_pos=start_pos,
            width=width, height=height, body_params=params, noise_params={},
        )
    return body


def adb_speed(A, f, b=0.0, warmup=20, steps=40):
    """跑 warmup+steps 个宏步, 返回 (每步速率, 每步转角)。"""
    body = build_adb_body(wave_amplitude=A, wave_frequency=f, steer_bias=b)
    for _ in range(warmup):
        body.step_physics(env=None, dt=1.0)
    x0, y0 = body.com.copy()
    t0 = body.body_theta
    for _ in range(steps):
        body.step_physics(env=None, dt=1.0)
    disp = body.com - np.array([x0, y0])
    return float(np.hypot(disp[0], disp[1])) / steps, (body.body_theta - t0) / steps


def test_adb_rft_forward_motion_and_taylor_scaling():
    """RFT: 波向尾传 → 身体前向推进; 速度 ∝ A²·f (Taylor 1951); 无波不动。"""
    s1, _ = adb_speed(0.4, 0.25)
    s2, _ = adb_speed(0.8, 0.25)   # A 翻倍 → 速度 ≈ ×4 (平方律)
    s3, _ = adb_speed(0.8, 0.5)    # f 翻倍 → 速度 ≈ ×2
    s0, _ = adb_speed(0.0, 0.5)    # 无波 → 不动
    assert s0 < 1e-4
    assert s1 > 1e-4
    assert s2 == pytest.approx(4.0 * s1, rel=0.25)
    assert s3 == pytest.approx(2.0 * s2, rel=0.25)


def test_adb_rft_isotropic_drag_no_propulsion():
    """drag_ratio=1.0 (各向同性阻力) → 推进归零 (RFT 各向异性的必要条件)。

    测量窗口取整周期 (f=0.25 → 周期 4 宏步), 消除摆动残留。
    """
    body = build_adb_body()
    body.drag_ratio = 1.0
    for _ in range(20):
        body.step_physics(env=None, dt=1.0)
    x0, y0 = body.com.copy()
    for _ in range(120):  # 30 个整周期
        body.step_physics(env=None, dt=1.0)
    rate = float(np.hypot(*(body.com - np.array([x0, y0])))) / 120
    assert rate < 1e-3


def test_adb_rft_steer_bias_both_directions():
    """曲率偏置 b 的符号决定转向方向 (力矩平衡 → 身体旋转)。

    测量窗口取整周期, 消除身体轴摆动 (±~18°) 的残留。
    """
    _, dt_pos = adb_speed(1.0, 0.25, b=0.05, warmup=20, steps=40)   # 10 个整周期
    _, dt_neg = adb_speed(1.0, 0.25, b=-0.05, warmup=20, steps=40)
    assert dt_pos > 1e-5
    assert dt_neg < -1e-5


def test_adb_rft_mirror_covariance():
    """镜像协变: 相位 0 与相位 π 出发 1 宏步的侧向位移互为反号 (有限线虫相位锁定漂移, 自洽性)。"""

    def one_macro(phase0):
        b2 = build_adb_body()
        b2.wave_phase = phase0
        b2._rebuild_shape_points()
        x0, y0 = b2.com.copy()
        b2.step_physics(env=None, dt=1.0)
        return (b2.com - np.array([x0, y0]))[1]

    d0, d1 = one_macro(0.0), one_macro(math.pi)
    assert abs(d0) > 1e-3
    assert d1 == pytest.approx(-d0, abs=1e-3)


def test_adb_rft_substeps_and_energy():
    """子步积分: history 每子步一帧; 相位每宏步推进 2πf; 能耗 = 机械耗散功。"""
    body = build_adb_body()
    h0 = len(body.history)
    result = body.step_physics(env=None, dt=1.0)
    assert len(body.history) - h0 == body.sub_steps
    assert result["dissipation"] >= 0.0
    assert body.max_energy - body.energy == pytest.approx(result["dissipation"], abs=1e-9)

    body2 = build_adb_body(wave_frequency=0.3)
    ph0 = body2.wave_phase
    body2.step_physics(env=None, dt=1.0)
    assert body2.wave_phase - ph0 == pytest.approx(2 * math.pi * 0.3, abs=1e-9)


def test_adb_ac_actions_and_curriculum_freeze():
    """ADB AC 动作 3 维 (A, f, b) + 边界裁剪; 课程阶段一冻结 b=0。"""
    from core.neural_networks import PYTORCH_AVAILABLE

    if not PYTORCH_AVAILABLE:
        pytest.skip("PyTorch is not installed in this environment")

    env = build_single_center_env(width=20, height=20)
    body = build_adb_body(width=20, height=20, start_pos=(5, 5), curriculum_freeze_steps=3)
    assert body.ac_action_dim == 3
    with contextlib.redirect_stdout(io.StringIO()):
        assert body.setup_actor_critic() is True
    agent = body.actor_critic_agent
    assert agent.action_dim == 3

    action = agent.act(body.get_state_v2(env), add_noise=False)
    assert len(action) == 3
    assert 0.05 - 1e-6 <= action[0] <= 2.0 + 1e-6
    assert 0.02 - 1e-6 <= action[1] <= 0.8 + 1e-6
    assert -0.08 - 1e-6 <= action[2] <= 0.08 + 1e-6

    fake = (1.0, 0.25, 0.06)
    for i in range(4):
        d = body._build_actor_action(fake)
        expected = 0.0 if i < 3 else 0.06
        assert d["steer_bias"] == pytest.approx(expected, abs=1e-12)
        assert set(d.keys()) == {"wave_amplitude", "wave_frequency", "steer_bias"}


def test_adb_requires_ac_and_ac_path_smoke():
    """ADB 无离散方向路径: 未装配 AC 时 decide_move 明确报错; AC 路径冒烟。"""
    from core.neural_networks import PYTORCH_AVAILABLE

    if not PYTORCH_AVAILABLE:
        pytest.skip("PyTorch is not installed in this environment")

    env = build_single_center_env(width=40, height=40)
    body = build_adb_body(width=40, height=40, start_pos=(20, 20))
    with pytest.raises(RuntimeError, match="仅支持 Actor-Critic"):
        body.decide_move(env)

    with contextlib.redirect_stdout(io.StringIO()):
        body.setup_actor_critic()
    random.seed(7)
    np.random.seed(7)
    assert body.decide_move(env, epsilon=0.5, alpha=0.1, gamma=0.9) is True
    assert np.isfinite(body.total_reward)
    assert 0.05 - 1e-6 <= body.wave_amplitude <= 2.0 + 1e-6
    assert 0.02 - 1e-6 <= body.wave_frequency <= 0.8 + 1e-6


def test_adb_round_reset_rebuilds_rft_shape():
    """reset_worm_for_new_round: 直线解释为新一轮初始轴, 重建 RFT 形状。"""
    env = build_single_center_env(width=40, height=40)
    body = build_adb_body(width=40, height=40, start_pos=(20, 20))
    reset_worm_for_new_round(body, env, start_pos=(10, 10), width=40, height=40,
                             field_type="single_center")
    assert body.com[0] == pytest.approx(10.0, abs=1e-9)
    assert body.com[1] == pytest.approx(10.0, abs=1e-9)
    assert body.body_theta == pytest.approx(0.0, abs=1e-9)  # 水平排列: 头朝 +x
    assert body.wave_phase == 0.0
    assert len(body.body_segments) == body.num_segments


def test_standard_simulation_engine_adb_ac(monkeypatch, tmp_path):
    """引擎级：ADB + Actor-Critic 短训练完整走到结果保存阶段。"""
    from core.neural_networks import PYTORCH_AVAILABLE

    if not PYTORCH_AVAILABLE:
        pytest.skip("PyTorch is not installed in this environment")

    _install_streamlit_and_matplotlib_stubs(monkeypatch)
    simulation_engine = importlib.import_module("simulation_engine")
    monkeypatch.setattr(
        simulation_engine,
        "st",
        types.SimpleNamespace(session_state={"stop_requested": False}),
    )

    factory_calls = _spy_body_model_factory(monkeypatch, simulation_engine)
    saved_results = []
    monkeypatch.setattr(
        simulation_engine, "save_and_visualize_results",
        lambda *args, **kwargs: saved_results.append(args) or None,
    )

    config = _build_engine_config(tmp_path, "pytest_adb_ac_engine")
    training_params = _build_engine_training_params(width=16, height=16)
    training_params.update({
        "method": "Actor-Critic",
        "body_model_type": "active_deformation",
        "body_params": {"sample_count": 5, "body_length": 8.0, "sub_steps": 5},
        "steps_per_round": 2,
    })

    with contextlib.redirect_stdout(io.StringIO()):
        events = list(
            simulation_engine.run_standard_simulation_engine(
                config=config,
                training_params=training_params,
                field_type="single_center",
                use_neural_network=True,
                enable_step_tracking=False,
            )
        )

    assert factory_calls[0]["kwargs"]["model_type"] == "active_deformation"
    assert saved_results, "ADB + AC 训练没有进入结果保存阶段"
    assert events[-1][0:2] == (1000, 1000)
