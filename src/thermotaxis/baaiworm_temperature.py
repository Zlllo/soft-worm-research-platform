"""World-fixed 3-D temperature field for the BAAIWorm head tracker."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _xyz(value, name):
    xyz = np.asarray(value, dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError(f"{name} must be a finite XYZ position or vector")
    return xyz


@dataclass(frozen=True)
class BAAIWormTemperatureField:
    """19–21 °C field. The warm center is four resting body lengths along normal."""

    origin_world: Tuple[float, float, float]
    body_length_world: float
    normal_world: Tuple[float, float, float]

    def __post_init__(self):
        origin = _xyz(self.origin_world, "origin_world")
        normal = _xyz(self.normal_world, "normal_world")
        norm = np.linalg.norm(normal)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("normal_world must have finite, positive length")
        if not np.isfinite(self.body_length_world) or self.body_length_world <= 0:
            raise ValueError("body_length_world must be finite and positive")
        object.__setattr__(self, "origin_world", tuple(origin))
        object.__setattr__(self, "normal_world", tuple(normal / norm))

    def temperature_c(self, positions_world):
        """Evaluate one XYZ point or an array of points with XYZ on the last axis."""
        positions = np.asarray(positions_world, dtype=float)
        if positions.shape[-1:] != (3,) or not np.isfinite(positions).all():
            raise ValueError("positions_world must contain finite XYZ coordinates")
        relative = (positions - self.origin_world) / self.body_length_world
        height = relative @ self.normal_world
        lateral = relative - height[..., None] * np.asarray(self.normal_world)
        radius_squared = np.sum((lateral / 4.0) ** 2, axis=-1) + ((height - 4.0) / 2.0) ** 2
        return 19.0 + 2.0 * np.exp2(-radius_squared)

    def head_temperature_c(self, interact, worm_id=0):
        """Return only local temperature, using the native simulator's world transform."""
        head_local = interact.get_head_location(worm_id)
        head_world = interact.worm_local_to_world(worm_id, head_local)
        return float(self.temperature_c(_xyz(head_world, "head_world")))
