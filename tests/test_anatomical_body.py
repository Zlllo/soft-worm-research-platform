import numpy as np
import pytest

from thermotaxis.neuromechanics.anatomical_body import load_anatomical_body_wiring, PlanarMuscleProjection


def test_full_named_wiring_keeps_all_neurons_and_nmj_types():
    w = load_anatomical_body_wiring()
    assert len(w.neuron_names) == 302 and len(w.muscle_names) == 95
    assert w.neuron_chemical.shape == (302, 302) and w.nmj_chemical.shape == (302, 95)
    assert np.count_nonzero(w.nmj_chemical) == 956
    assert np.count_nonzero(w.neuron_muscle_gap) == 5
    assert np.count_nonzero(w.nmj_chemical.any(1)) == 162
    assert w.nmj_chemical.any(0).all()
    assert not w.nmj_chemical.flags.writeable


def test_named_neuron_impulse_and_missing_edge_guard():
    w = load_anatomical_body_wiring()
    i = w.neuron_names.index('AS2')
    target = np.flatnonzero(w.nmj_chemical[i] > 0)[0]
    efficacy = np.zeros((302, 95)); efficacy[i, target] = .25
    activity = np.zeros(302); activity[i] = 1
    output = w.chemical_muscle_input(activity, efficacy=efficacy)
    assert output[target] == .25 and np.count_nonzero(output) == 1
    missing = np.argwhere(w.nmj_chemical == 0)[0]
    efficacy[tuple(missing)] = .1
    with pytest.raises(ValueError, match='absent'):
        w.chemical_muscle_input(activity, efficacy=efficacy)


def test_planar_projection_declares_left_right_loss_and_missing_muscle():
    w = load_anatomical_body_wiring(); p = PlanarMuscleProjection.rank_bins(w.muscle_names)
    a = np.zeros(95); b = np.zeros(95)
    a[w.muscle_names.index('MDL07')] = 1
    b[w.muscle_names.index('MDR07')] = 1
    np.testing.assert_array_equal(p.to_segments(a), p.to_segments(b))
    assert np.count_nonzero(p.to_segments(a)) == 2
    assert p.to_segments(a)[12, 0] == .5
    assert np.linalg.matrix_rank(p.matrix) == 48
    all_on = p.to_segments(np.ones(95))
    assert np.all(all_on[:46] == 1) and np.all(all_on[46:, 1] == .5)
    # Source ordering is an explicit contract, not a hidden simulator convention.
    reverse = PlanarMuscleProjection.rank_bins(w.muscle_names[::-1])
    np.testing.assert_array_equal(reverse.to_segments(a[::-1]), p.to_segments(a))


@pytest.mark.parametrize('values', [np.zeros(94), np.full(95, np.nan)])
def test_projection_rejects_invalid_drives(values):
    p = PlanarMuscleProjection.rank_bins(load_anatomical_body_wiring().muscle_names)
    with pytest.raises(ValueError):
        p.to_segments(values)
