"""Exercise one-click training and completed-result persistence through real Streamlit reruns."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '程序'))
from streamlit.testing.v1 import AppTest


@pytest.fixture
def platform(monkeypatch, tmp_path):
    import core.environment
    import simulation_engine
    config_class = core.environment.ExperimentConfig

    def local_config(name, **kwargs):
        return config_class(name, parent_dir=str(tmp_path))

    calls = []

    def engine(config, params, *args, **kwargs):
        calls.append(params)
        out = Path(config.output_dir)
        (out / 'experiment.log').write_text('UI integration result', encoding='utf-8')
        (out / 'policy_evaluation.json').write_text(json.dumps({
            'success_count': 1, 'episode_count': 1, 'goal_radius': .75,
            'episodes': [{'actual_start': [12, 5], 'success': True, 'steps': 20, 'final_distance': .5}],
        }))
        yield 5, 1000, '准备实验', {'phase': 'init'}
        yield 1000, 1000, '训练完成', {'phase': 'results'}

    monkeypatch.setattr(core.environment, 'ExperimentConfig', local_config)
    monkeypatch.setattr(simulation_engine, 'run_standard_simulation_engine', engine)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / '程序/app.py')).run(timeout=30)
    assert not app.exception
    return app, calls


def test_single_click_runs_once_and_results_survive_configuration_change(platform):
    app, calls = platform
    app.text_input(key='experiment_name').set_value('persisted_ui_result').run()
    app.button(key='main_start_experiment').click().run(timeout=30)
    assert not app.exception and not app.error
    assert len(calls) == 1
    assert app.session_state['is_simulating'] is False
    assert app.session_state['last_result']['name'] == 'persisted_ui_result'
    app.selectbox(key='field_selector_standard').select('🎯 单热源 (single_center)').run()
    assert not app.exception
    assert len(calls) == 1
    assert app.text_input(key='experiment_name').value == 'persisted_ui_result'
    assert any('最近一次实验' in heading.value for heading in app.markdown)
    assert any('冻结策略评估：1/1' in info.value for info in app.info)
    assert len(app.dataframe) == 1


def test_all_modes_and_continuous_controls_preserve_internal_selection_values(platform):
    app, calls = platform
    app.selectbox(key='body_model_selector').select('📏 连续中心线 (Continuous)').run()
    app.selectbox(key='direction_selector_standard').select('连续方向 (Actor-Critic)').run()
    assert not app.exception
    assert app.selectbox(key='reward_variant_selectbox').value == 'continuous'
    assert app.slider(key='discount_factor_slider').value == pytest.approx(.99)
    for mode in ['🔄 迁移学习', '📚 课程学习', '🎓 标准训练']:
        app.radio(key='mode_selector').set_value(mode).run()
        assert not app.exception
        assert app.button(key='main_start_experiment').disabled == (mode != '🎓 标准训练')
    assert not calls


def test_stop_unlocks_configuration_and_keeps_completed_bundle(platform):
    app, calls = platform
    app.button(key='main_start_experiment').click().run(timeout=30)
    previous = app.session_state['last_result'].copy()
    app.session_state['is_simulating'] = True
    app.run()
    app.button(key='stop_experiment_button').click().run()
    assert not app.exception
    assert app.session_state['is_simulating'] is False
    assert app.session_state['last_result'] == previous
    assert not app.text_input(key='experiment_name').disabled
    assert not app.selectbox(key='body_model_selector').disabled
    assert any('本次训练已停止' in info.value for info in app.info)
    assert len(calls) == 1
