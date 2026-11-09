"""Data-bound checks for simulator muscle order and the absent MVL24 slot."""
import json
from pathlib import Path

import numpy as np
import pytest

from thermotaxis.neuromechanics.muscle_mapping import MuscleMapping, body_wall_slots


def test_current_connectome_has_95_muscles_and_preserves_named_impulse():
    data = json.loads((Path(__file__).parents[1] / "data/connectome/cook2019_herm.json").read_text())
    slots = body_wall_slots()
    names = tuple(n for n in data["nodes"] if n in slots)
    assert len(names) == 95
    assert set(slots) - set(names) == {"MVL24"}
    # Deliberately reverse source order and use a different simulator order.
    mapping = MuscleMapping(names[::-1], body_wall_slots(("MDR", "MVR", "MDL", "MVL")))
    x = np.zeros((2, 95))
    x[0, mapping.anatomical_names.index("MDL07")] = .6
    x[1, mapping.anatomical_names.index("MVR24")] = .9
    y = mapping.to_slots(x)
    assert y.shape == (2, 96)
    assert y[0, mapping.slot_names.index("MDL07")] == .6
    assert y[1, mapping.slot_names.index("MVR24")] == .9
    assert np.count_nonzero(y) == 2
    assert np.all(y[:, mapping.slot_names.index("MVL24")] == 0)
    np.testing.assert_array_equal(mapping.from_slots(y), x)
    y[0, mapping.slot_names.index("MVL24")] = .1
    with pytest.raises(ValueError, match="stay silent"):
        mapping.from_slots(y)


def test_rejects_incomplete_ambiguous_and_nonfinite_input():
    with pytest.raises(ValueError, match="unique"):
        MuscleMapping(("MDL01", "MDL01"), ("MDL01",))
    with pytest.raises(ValueError, match="omit"):
        MuscleMapping(("MDL01", "MDR01"), ("MDL01",))
    with pytest.raises(ValueError, match="canonical"):
        MuscleMapping(("DR0",), ("DR0",))
    mapping = MuscleMapping(("MDL01",), ("MDL01", "MVL24"))
    with pytest.raises(ValueError, match="last axis"):
        mapping.to_slots([1., 2.])
    with pytest.raises(ValueError, match="finite"):
        mapping.to_slots([float("nan")])
