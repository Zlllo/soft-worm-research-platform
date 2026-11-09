"""Map named anatomical muscles to explicitly ordered simulator slots.

Cook's adult hermaphrodite has no MVL24. Some simulators allocate that slot;
it is held silent rather than assigned an invented anatomical connection.
No assumptions about zero-based external IDs or array order are made here.
"""
from dataclasses import dataclass
import re

import numpy as np


def body_wall_slots(quadrants=("MDL", "MDR", "MVL", "MVR")):
    if len(quadrants) != 4 or set(quadrants) != {"MDL", "MDR", "MVL", "MVR"}:
        raise ValueError("declare each anatomical quadrant exactly once")
    return tuple(f"{quadrant}{i:02d}" for quadrant in quadrants for i in range(1, 25))


@dataclass(frozen=True)
class MuscleMapping:
    anatomical_names: tuple[str, ...]
    slot_names: tuple[str, ...]

    def __post_init__(self):
        for names in (self.anatomical_names, self.slot_names):
            if not names or len(set(names)) != len(names):
                raise ValueError("muscle names must be nonempty and unique")
            if any(not re.fullmatch(r"M[DV][LR](?:0[1-9]|1[0-9]|2[0-4])", n) for n in names):
                raise ValueError("use canonical anatomical muscle names")
        if set(self.anatomical_names) - set(self.slot_names):
            raise ValueError("simulator slots omit an anatomical muscle")
        if set(self.slot_names) - set(self.anatomical_names) - {"MVL24"}:
            raise ValueError("unrecognized extra simulator slot")

    @property
    def inactive_names(self):
        return tuple(n for n in self.slot_names if n not in self.anatomical_names)

    def to_slots(self, values):
        values = np.asarray(values)
        if values.ndim < 1 or values.shape[-1] != len(self.anatomical_names):
            raise ValueError("last axis must follow anatomical_names")
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError("muscle values must be finite numeric data")
        result = np.zeros((*values.shape[:-1], len(self.slot_names)), dtype=values.dtype)
        indices = [self.slot_names.index(n) for n in self.anatomical_names]
        result[..., indices] = values
        return result

    def from_slots(self, values):
        values = np.asarray(values)
        if values.ndim < 1 or values.shape[-1] != len(self.slot_names):
            raise ValueError("last axis must follow slot_names")
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError("muscle values must be finite numeric data")
        for n in self.inactive_names:
            if np.any(values[..., self.slot_names.index(n)] != 0):
                raise ValueError(f"absent anatomical muscle {n} must stay silent")
        return values[..., [self.slot_names.index(n) for n in self.anatomical_names]].copy()

    def as_dict(self):
        return {
            "anatomical_names": list(self.anatomical_names),
            "slot_names": list(self.slot_names),
            "inactive_names": list(self.inactive_names),
            "anatomical_to_slot": [self.slot_names.index(n) for n in self.anatomical_names],
            "scope": "named index adapter; no efficacy, force or unit conversion",
        }
