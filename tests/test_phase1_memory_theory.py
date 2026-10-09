"""Independent deterministic checks of the theory's actual controller contract."""

from dataclasses import replace
import math
from pathlib import Path

import pytest

from thermotaxis.contracts import SensorReading
from thermotaxis.phase1 import TwoTimescaleTurnController
from thermotaxis.phase1_config import load_phase1_config


CONFIG = load_phase1_config(Path(__file__).resolve().parents[1] / "configs/phase1_reference.json")


@pytest.mark.parametrize("slope", [-.02, .02])
def test_sampled_error_ramp_matches_filter_lag_and_actual_clipped_rate(slope):
    # Keep error positive for the whole ramp: no crossing of abs(T-T*) cusp.
    params = CONFIG.thermotaxis_controller
    controller = TwoTimescaleTurnController(params, CONFIG.temperature.preferred_K, 17)
    dt = CONFIG.time.sensor_dt_s
    for sample in range(2001):
        time = sample * dt
        state = controller.observe(SensorReading(CONFIG.temperature.preferred_K + 10 + slope*time, time))
    lag = dt/math.expm1(dt/params.slow_memory_tau_s) - dt/math.expm1(dt/params.fast_memory_tau_s)
    expected_improvement = -slope*lag
    effective = math.copysign(max(abs(expected_improvement)-params.response_deadband_K, 0), expected_improvement)
    expected_rate = min(params.max_turn_rate_per_s, max(params.min_turn_rate_per_s,
        params.baseline_turn_rate_per_s-params.response_gain_per_K_s*effective))
    assert state.improvement_K == pytest.approx(expected_improvement, abs=2e-12)
    assert state.turn_rate_per_s == pytest.approx(expected_rate, abs=1e-11)
    if slope < 0:
        assert state.turn_rate_per_s == params.min_turn_rate_per_s
    else:
        assert state.turn_rate_per_s == pytest.approx(.4198800887154294)


def test_sub_deadband_ramp_has_no_directional_rate_response():
    params = CONFIG.thermotaxis_controller
    controller = TwoTimescaleTurnController(params, CONFIG.temperature.preferred_K, 19)
    for sample in range(2001):
        time = sample*CONFIG.time.sensor_dt_s
        state = controller.observe(SensorReading(CONFIG.temperature.preferred_K + 10 - .001*time, time))
    assert 0 < state.improvement_K < params.response_deadband_K
    assert state.turn_rate_per_s == params.baseline_turn_rate_per_s


def test_current_control_uses_event_probability_not_poisson_parity_probability():
    params = replace(CONFIG.thermotaxis_controller, response_gain_per_K_s=0)
    controller = TwoTimescaleTurnController(params, CONFIG.temperature.preferred_K, 21)
    controller.observe(SensorReading(CONFIG.temperature.preferred_K, 0))
    h = 5.
    event_p = -math.expm1(-params.baseline_turn_rate_per_s*h)
    parity_p = -math.expm1(-2*params.baseline_turn_rate_per_s*h)/2
    class FixedDraw:
        def __init__(self, draw):
            self.draw = draw
        def random(self):
            return self.draw
    # A deterministic draw between the probabilities distinguishes the contracts.
    controller._rng = FixedDraw((event_p+parity_p)/2)
    assert controller.should_turn(h) is True
    controller._rng = FixedDraw(event_p+1e-12)
    assert controller.should_turn(h) is False
