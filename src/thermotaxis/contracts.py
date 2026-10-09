"""Narrow interfaces that keep physical truth out of the controller."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Protocol, Tuple

from .config import SensorConfig, TemperatureConfig


Position2D = Tuple[float, float]


class TemperatureField(Protocol):
    def temperature_K(self, position_m: Position2D, time_s: float) -> float:
        ...


@dataclass(frozen=True)
class LinearTemperatureField:
    reference_K: float
    gradient_K_per_m: Position2D

    @classmethod
    def from_config(cls, config: TemperatureConfig) -> "LinearTemperatureField":
        return cls(config.reference_K, config.gradient_K_per_m)

    def temperature_K(self, position_m: Position2D, time_s: float) -> float:
        del time_s
        return self.reference_K + sum(
            gradient * coordinate
            for gradient, coordinate in zip(self.gradient_K_per_m, position_m)
        )


@dataclass(frozen=True)
class SensorReading:
    temperature_K: float
    sample_time_s: float


class LocalTemperatureSensor:
    """The only bridge from the truth field to a future controller."""

    def __init__(self, config: SensorConfig, seed: int):
        self._config = config
        self._rng = random.Random(seed)

    def sample(
        self, field: TemperatureField, position_m: Position2D, time_s: float
    ) -> SensorReading:
        true_temperature = field.temperature_K(position_m, time_s)
        noise = self._rng.gauss(0.0, self._config.noise_std_K)
        measured = true_temperature + noise
        if not math.isfinite(measured):
            raise RuntimeError("sensor produced a non-finite reading")
        return SensorReading(temperature_K=measured, sample_time_s=float(time_s))
