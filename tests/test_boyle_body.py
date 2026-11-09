"""Native integration checks. Set BOYLE_WORKER to a generated worker binary."""
from dataclasses import asdict
import os

import numpy as np
import pytest

from thermotaxis.neuromechanics.boyle_body import BoyleBody
from thermotaxis.neuromechanics.boyle_output import side_lengths
from thermotaxis.neuromechanics.thermal_body import ThermalBody


@pytest.fixture
def body():
    binary = os.environ.get("BOYLE_WORKER")
    if not binary:
        pytest.skip("build native worker and set BOYLE_WORKER to run integration checks")
    with BoyleBody(binary) as body:
        yield body


def test_chunking_reset_and_independent_workers(body):
    body.reset()
    continuous = body.step(1000)
    body.reset()
    for n in [1, 19, 83, 197, 300, 400]:
        chunked = body.step(n)
    for name in ["pose", "velocity", "neural", "muscle", "stretch", "side_length"]:
        np.testing.assert_array_equal(getattr(continuous, name), getattr(chunked, name))
    with BoyleBody(body.binary) as other:
        other.reset()
        repeated = other.step(1000)
        np.testing.assert_array_equal(continuous.pose, repeated.pose)
        # The original worker remains at 1 s while the other advances.
        other.step(10, head_bias=.05)
        assert body.state.microsteps == 1000


def test_causal_clock_and_accepted_geometry(body):
    body.reset()
    for _ in range(20):
        state = body.step(100)
        assert state.internal_time_s <= state.time_s + 1e-12
        np.testing.assert_allclose(state.side_length, side_lengths(state.pose), atol=1e-18, rtol=0)


def test_invalid_control_does_not_advance(body):
    initial = body.reset()
    for args in [dict(microsteps=0), dict(microsteps=1.5), dict(drive=np.nan),
                 dict(head_bias=.2), dict(drive=-.1)]:
        with pytest.raises(ValueError):
            body.step(**args)
    with pytest.raises(ValueError):
        body.advance(.0015)
    assert body.state is initial
    assert body.step(1).microsteps == 1


def test_actual_head_temperature_and_noise_reset(body):
    env = ThermalBody(body, gradient_K_m=(40., 20.))
    observation = env.reset(head_xy_m=(.001, -.002))
    assert observation.temperature_K == pytest.approx(293.15+40*.001-20*.002)
    for _ in range(5):
        observation = env.step(sensor_samples=1)[0]
        assert observation.temperature_K == pytest.approx(293.15+env.gradient @ body.state.head_xy_m, abs=1e-12)
    assert set(asdict(observation)) == {"time_s", "temperature_K", "signed_speed_m_s", "head_curvature_per_m"}
    noisy = ThermalBody(body, noise_std_K=.01, seed=7)
    a = [noisy.reset(), *noisy.step()]
    b = [noisy.reset(), *noisy.step()]
    assert a == b
    masked = ThermalBody(body, body_feedback=False)
    masked.reset()
    observation = masked.step()[1]
    assert observation.signed_speed_m_s == 0
    assert observation.head_curvature_per_m == 0
    assert np.any(body.state.stretch != 0)  # Shared locomotor feedback stays active.


def test_initial_rigid_transform(body):
    body.reset()
    state = body.step(1000)
    angle = .5
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    offset = np.array([.001, -.002])
    body.reset(head_xy_m=offset, rotation_rad=angle)
    moved = body.step(1000)
    # Solver uses coordinate-wise tolerances; exact rotation equality is not expected.
    np.testing.assert_allclose(moved.pose[:, :2], state.pose[:, :2] @ rotation.T+offset, atol=2e-7, rtol=0)
