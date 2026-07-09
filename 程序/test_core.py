import contextlib
import importlib
import importlib.util
import io
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
