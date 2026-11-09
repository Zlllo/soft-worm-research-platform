"""Temperature at the moving head, with an explicitly restricted policy view."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .boyle_body import BoyleBody


@dataclass(frozen=True)
class ThermalObservation:
    time_s: float
    temperature_K: float
    signed_speed_m_s: float
    head_curvature_per_m: float


class ThermalBody:
    """Environment owns the field; policy receives only local sensor values.

    Speed and curvature are ideal simulated body feedback, not calibrated
    biological proprioceptors. No heading, position or field gradient occurs
    in ThermalObservation. Full body state is reserved for evaluation.
    """

    def __init__(self, body: BoyleBody, *, gradient_K_m=(40., 0.), reference_K=293.15,
                 noise_std_K=0., sensor_dt_s=.1, seed=0, body_feedback=True):
        gradient = np.asarray(gradient_K_m, dtype=float)
        if gradient.shape != (2,) or not np.isfinite(gradient).all():
            raise ValueError("gradient must be a finite vector of length two")
        if not math.isfinite(reference_K) or not math.isfinite(noise_std_K) or noise_std_K < 0:
            raise ValueError("invalid temperature scale")
        self.body = body
        self.gradient = gradient.copy()
        self.reference_K = float(reference_K)
        self.noise_std_K = float(noise_std_K)
        self.sensor_ticks = body.microstep_count(sensor_dt_s)
        self.seed = seed
        self.body_feedback = bool(body_feedback)
        self.samples = []
        self._rng = np.random.default_rng(seed)

    def reset(self, **initial_pose):
        self.body.reset(**initial_pose)
        self._rng = np.random.default_rng(self.seed)
        self.samples = [self._sample()]
        return self.samples[0]

    def _sample(self):
        state = self.body.state
        temperature = self.reference_K + self.gradient @ state.head_xy_m
        temperature += self._rng.normal(0., self.noise_std_K)
        return ThermalObservation(state.time_s, float(temperature),
            state.signed_speed_m_s if self.body_feedback else 0.,
            state.head_curvature_per_m if self.body_feedback else 0.)

    def step(self, *, drive=.675, head_bias=0., sensor_samples=2):
        if isinstance(sensor_samples, bool) or not isinstance(sensor_samples, int) or sensor_samples < 1:
            raise ValueError("sensor_samples must be a positive integer")
        observations = []
        for _ in range(sensor_samples):
            self.body.step(self.sensor_ticks, drive=drive, head_bias=head_bias)
            observations.append(self._sample())
        self.samples.extend(observations)
        return observations
