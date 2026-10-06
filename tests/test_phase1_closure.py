"""Independent core replay and limiting checks for the finite-clock surrogate."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from thermotaxis.contracts import SensorReading
from thermotaxis.phase1 import TwoTimescaleTurnController
from thermotaxis.phase1_config import MemoryControllerConfig

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sampling_analysis', ROOT/'analysis/phase1_sampling_response.py')
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)
DESIGN = json.loads((ROOT/'configs/phase1_sampling_response.json').read_text())


@pytest.mark.parametrize('control_interval', [.1, .2, 1.])
def test_collapsed_surrogate_replays_real_controller_with_same_pcg_stream(control_interval):
    # Verify real observe/should_turn, original reference rate limits, and a
    # random direction history rather than reimplementing the build's lag loop.
    d = dict(DESIGN, burn_s=10., measurement_s=50.)
    seed, gain = 985713, .1
    rng = np.random.default_rng(seed)
    direction = 1 if rng.random() < .5 else -1
    params = MemoryControllerConfig(.5, 5., .1, .01, .5, gain, 0.)
    controller = TwoTimescaleTurnController(params, 293.15, 0)
    controller._rng = rng  # Same uniform source, while keeping actual core decisions.
    error, displacement, turns = 1000., 0., 0
    controller.observe(SensorReading(293.15+error, 0.))
    stride = round(control_interval/.1)
    for n in range(round(60/control_interval)):
        if n*control_interval >= 10.-1e-12:
            displacement += direction*.0002*control_interval
        for j in range(1, stride+1):
            error += direction*.02*.1
            state = controller.observe(SensorReading(293.15+error, n*control_interval+j*.1))
        assert .01 < state.turn_rate_per_s < .5
        if controller.should_turn(control_interval):
            turns += 1
            direction *= -1
    records, _ = sampling.trajectories(d, control_interval, gain, [seed])
    assert records[0]['turn_count'] == turns
    assert records[0]['drift_m_s'] == pytest.approx(displacement/50., abs=2e-16)


@pytest.mark.parametrize('control_interval', [.1, .2, 1.])
def test_finite_clock_coefficient_matches_explicit_past_interval_kernel(control_interval):
    # Sum sensor lag contributions directly over past directional intervals.
    # E[s_now s_past] = r**j. This is independent of the geometric closed form.
    h, hs, v, b, rate = control_interval, .1, .0002, 100., .1
    r = 2*math.exp(-rate*h)-1
    correlations = []
    for tau in [.5, 5.]:
        q = math.exp(-hs/tau)
        lag_correlation = 0.
        for j in range(5000):
            for sensor_back in range(1, round(h/hs)+1):
                lag_correlation -= v*b*hs*q**(j*round(h/hs)+sensor_back)*r**j
        correlations.append(lag_correlation)
    expected = 2*v*h*math.exp(-rate*h)/(1-r)*(correlations[1]-correlations[0])
    assert sampling.coefficient(DESIGN, h) == pytest.approx(expected, rel=2e-12)


def test_finite_clock_weak_response_converges_to_continuous_coefficient():
    coarse = dict(DESIGN, sensor_interval_s=.01)
    fine = dict(DESIGN, sensor_interval_s=.001)
    continuous = sampling.continuous_coefficient(DESIGN)
    coarse_error = abs(sampling.coefficient(coarse, .01)-continuous)
    fine_error = abs(sampling.coefficient(fine, .001)-continuous)
    assert fine_error < coarse_error/5
    assert fine_error/abs(continuous) < .001


def boundary_module():
    spec = importlib.util.spec_from_file_location('boundary_analysis', ROOT/'analysis/phase1_boundary_stationarity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_occupation_integrator_splits_cross_bin_time_independent_of_direction():
    boundary = boundary_module()
    hist = boundary.segment_histogram([.004, .006], [.006, .004], .02, 4, .0002)
    assert hist == pytest.approx([10., 10., 0., 0.])


def test_occupation_integrator_rejects_unsupported_multi_bin_segment():
    boundary = boundary_module()
    with pytest.raises(ValueError, match='wider'):
        boundary.segment_histogram([.001], [.019], .02, 4, .0002)


def test_reflecting_ballistic_period_has_uniform_occupation_and_exact_thermal_moments():
    from dataclasses import replace
    from thermotaxis.phase1 import run_phase1_experiment
    boundary = boundary_module()
    cfg = boundary.config(.02, -.01, 97000, duration=200.)
    cfg = replace(cfg, point_model=replace(cfg.point_model, initial_direction='positive'),
                  thermotaxis_controller=replace(cfg.thermotaxis_controller,
                    min_turn_rate_per_s=0., baseline_turn_rate_per_s=0., response_gain_per_K_s=0.))
    result = run_phase1_experiment(cfg)
    blocks = boundary.blocks(result, cfg, [0., 100., 200.])
    for block in blocks:
        assert block['occupation'] == pytest.approx(np.full(60, 1/60), abs=2e-11)
        assert block['residence'] == pytest.approx(.25, abs=2e-11)
        assert block['rms_K'] == pytest.approx(100*.02/math.sqrt(12), abs=2e-11)
