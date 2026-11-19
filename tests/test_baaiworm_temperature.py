import numpy as np
import pytest

from thermotaxis.baaiworm_temperature import BAAIWormTemperatureField


def test_world_scale_and_curved_target():
    field = BAAIWormTemperatureField((8, 2, -1), 10, (0, 2, 0))
    points = [(8, 2, -1), (8, 42, -1), (8, 22, -1), (48, 42, -1), (8, 42, 39)]
    np.testing.assert_allclose(field.temperature_c(points), [19.125, 21, 20, 20, 20])


def test_mirrored_fields_agree_in_the_whole_start_plane():
    origin = np.array([2.0, 4.0, 8.0])
    n = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    up = BAAIWormTemperatureField(origin, 2, n)
    down = BAAIWormTemperatureField(origin, 2, -n)
    in_plane = origin + np.array([[2, 3, -2], [-1, 0, 1], [3, 1, -3]])
    np.testing.assert_allclose(up.temperature_c(in_plane), down.temperature_c(in_plane))
    assert np.all(up.temperature_c(in_plane+n) > down.temperature_c(in_plane+n))
    np.testing.assert_allclose(up.temperature_c(in_plane+n), down.temperature_c(in_plane-n))


def test_native_head_uses_world_transform_and_requested_worm():
    class Tracker:
        def get_head_location(self, worm_id):
            assert worm_id == 2
            return np.array([1, 0, 0], dtype=np.float32)

        def worm_local_to_world(self, worm_id, head):
            assert worm_id == 2
            np.testing.assert_array_equal(head, [1, 0, 0])
            return [0, 4, 0]

    field = BAAIWormTemperatureField((0, 0, 0), 1, (0, 1, 0))
    assert field.head_temperature_c(Tracker(), 2) == 21


@pytest.mark.parametrize("length,normal", [(0, (0, 1, 0)), (1, (0, 0, 0)), (np.nan, (1, 0, 0))])
def test_invalid_calibration(length, normal):
    with pytest.raises(ValueError):
        BAAIWormTemperatureField((0, 0, 0), length, normal)
