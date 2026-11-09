import numpy as np
import pytest

from thermotaxis.neuromechanics.boyle_output import load_body, side_lengths, curvature, rod_radii


def straight_pose():
    pose = np.zeros((49, 3))
    pose[:, 0] = np.linspace(0, 1e-3, 49)
    pose[:, 2] = np.pi/2
    return pose


def test_geometry_reconstruction_is_rigid_motion_invariant():
    pose = straight_pose()
    lengths = side_lengths(pose)
    expected = np.hypot(1e-3/48, np.diff(rod_radii()))
    np.testing.assert_allclose(lengths[:, 0], expected, atol=1e-18)
    np.testing.assert_allclose(lengths[:, 1], expected, atol=1e-18)
    np.testing.assert_allclose(curvature(pose), 0, atol=1e-12)
    angle = .7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = pose.copy()
    moved[:, :2] = pose[:, :2] @ rotation.T + [.002, -.003]
    moved[:, 2] += angle
    np.testing.assert_allclose(side_lengths(moved), lengths, atol=1e-18)


def test_output_rejects_truncated_nonfinite_or_noncausal_records(tmp_path):
    path = tmp_path/'body.csv'
    data = np.column_stack([[0., .04], np.tile(straight_pose().ravel(), (2, 1))])
    np.savetxt(path, data, delimiter=',')
    t, pose = load_body(path)
    assert pose.shape == (2, 49, 3)
    assert t[-1] == .04
    np.savetxt(path, data[:, :-1], delimiter=',')
    with pytest.raises(ValueError, match='49 rods'):
        load_body(path)
    data[1, 0] = 0
    np.savetxt(path, data, delimiter=',')
    with pytest.raises(ValueError, match='strictly increasing'):
        load_body(path)
