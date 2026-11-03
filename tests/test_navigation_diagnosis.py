"""The diagnostic replay must agree with production and preserve input meaning."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from thermotaxis.navigation_window_policy import WindowActorCritic

ANALYSIS = Path(__file__).resolve().parents[1] / 'analysis'
sys.path.insert(0, str(ANALYSIS))
spec = importlib.util.spec_from_file_location('navigation_diagnosis', ANALYSIS / 'diagnose_navigation_failures.py')
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


@pytest.mark.parametrize('name', ['real8', 'rewired8-44', 'isolated8'])
@pytest.mark.parametrize('interface', ['full', 'restricted'])
def test_trace_replays_architecture_with_padding_and_does_not_modify_weights(name, interface):
    torch.manual_seed(123)
    model = WindowActorCritic(name, interface)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    x, _ = diagnosis.stimuli()
    x = x[:3].clone(); x[0, :20, 6] = 0
    with torch.no_grad():
        replay, _, _ = diagnosis.trace_actor(model, x)
        torch.testing.assert_close(replay, model.actor_mean(x), atol=1e-6, rtol=1e-5)
        features = diagnosis.representation(model, x)
        torch.testing.assert_close(model.actor(features).squeeze(-1), replay)
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in before.items())


def test_temperature_perturbation_preserves_phase_and_updates_valid_temperature():
    x, _ = diagnosis.stimuli()
    x = x[:2].clone(); x[0, :10] = 0
    delta = torch.linspace(-.002, .003, 41)[None, :]
    changed = diagnosis.temperature_perturbation(x, delta)
    mask = x[:, :, 6]
    torch.testing.assert_close((changed[:, :, 0]+changed[:, :, 1])*.05,
                               (x[:, :, 0]+x[:, :, 1])*.05+delta*mask)
    assert torch.equal(changed[:, :, 2:], x[:, :, 2:])
    assert torch.equal(changed[0, :10], x[0, :10])


def test_failure_categories_do_not_count_post_recovery_as_full_success():
    def case(entry, pre, post):
        return dict(first_entry_s=entry, pre_stable_s=pre, post_stable_s=post,
                    complete_success=pre is not None and post is not None, post_tracking=.5)
    result = diagnosis.stage_summary([case(None,None,5.), case(1.,None,None),
                                     case(1.,2.,None), case(1.,2.,3.)])
    assert result['failures'] == dict(no_pre_entry=1, entered_without_pre_stability=1,
                                      pre_stable_without_post_stability=1, complete=1)
    assert result['complete'] == .25
    assert result['post_stable'] == .5


def test_teacher_positive_control_reconstructs_local_controller_from_features_only():
    from probe_navigation_teacher import teacher_command
    from thermotaxis.navigation_window_policy import NavigationFeatures
    from thermotaxis.isothermal_scene import PhaseRegressionController, SweepReading
    controller = PhaseRegressionController()
    features = NavigationFeatures(1, 'raw')
    for t in np.arange(41)*.1:
        phase = .7+np.pi*t
        packet = np.array([[[293.15+.003*t+.006*np.sin(phase), np.sin(phase), np.cos(phase), 1, 1]]], dtype=np.float32)
        x = features.observe(packet)
        controller.observe(SweepReading(float(t), float(packet[0,0,0]),
                                       float(np.arctan2(packet[0,0,1],packet[0,0,2]))))
    assert teacher_command(x)[0] == pytest.approx(controller.command(4.), abs=2e-6)


def test_gain_one_is_identity_and_gain_does_not_mutate_history():
    from probe_navigation_temperature_gain import FluctuationGain
    model = WindowActorCritic('real8', 'restricted')
    x, _ = diagnosis.stimuli()
    x = diagnosis.temperature_perturbation(x[:2], .002*x[:2,:,4])
    original = x.clone()
    torch.testing.assert_close(FluctuationGain(model, 1.).actor_mean(x), model.actor_mean(x))
    FluctuationGain(model, 10.).actor_mean(x)
    assert torch.equal(x, original)
