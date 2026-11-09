import os

import numpy as np
import pytest

from thermotaxis.neuromechanics.boyle_body import BoyleBody
from thermotaxis.neuromechanics.boyle_muscle_body import BoyleMuscleBody


@pytest.fixture
def body():
    path = os.environ.get('BOYLE_MUSCLE_WORKER')
    if not path:
        pytest.skip('set BOYLE_MUSCLE_WORKER to exercise native muscle integration')
    with BoyleMuscleBody(path) as body:
        yield body


def test_direct_drive_bypasses_background_and_respects_filter(body):
    body.reset(); zero = np.zeros((48, 2))
    s = body.step(200, muscle_drive=zero)
    assert np.all(s.neural == 0) and np.all(s.muscle == 0)
    signal = zero.copy(); signal[4, 0] = .3
    s = body.step(10, muscle_drive=signal)
    np.testing.assert_array_equal(s.neural, signal)
    assert s.muscle[4, 0] == pytest.approx(.3 * (1 - .99 ** 10), abs=1e-15)
    assert np.count_nonzero(s.muscle) == 1
    assert s.internal_time_s <= s.time_s + 1e-12


def test_invalid_muscle_command_preserves_state(body):
    initial = body.reset()
    for command in [np.zeros(96), np.full((48, 2), np.nan), np.full((48, 2), 1.1)]:
        with pytest.raises(ValueError):
            body.step(muscle_drive=command)
    assert body.state is initial
    with pytest.raises(TypeError):
        body.step(drive=.675)


def test_native_muscle_chunking_and_reset(body):
    signal = np.zeros((48, 2)); signal[:4, 0] = .2
    body.reset(); continuous = body.step(200, muscle_drive=signal)
    body.reset()
    for count in (1, 19, 80, 100):
        chunked = body.step(count, muscle_drive=signal)
    for key in ('pose', 'velocity', 'neural', 'muscle', 'stretch', 'side_length'):
        np.testing.assert_array_equal(getattr(continuous, key), getattr(chunked, key))


def test_legacy_neural_output_replay_preserves_mechanics(body):
    original = os.environ.get('BOYLE_WORKER')
    if not original:
        pytest.skip('set BOYLE_WORKER for the legacy replay comparison')
    with BoyleBody(original) as reference:
        reference.reset(); body.reset()
        for _ in range(200):
            old = reference.step(1)
            new = body.step(1, muscle_drive=old.neural)
            np.testing.assert_array_equal(new.pose, old.pose)
            np.testing.assert_array_equal(new.muscle, old.muscle)
