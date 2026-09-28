"""Phase-1 one-dimensional thermotaxis mechanism experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from .contracts import LinearTemperatureField, LocalTemperatureSensor, SensorReading
from .phase1_config import MemoryControllerConfig, Phase1ExperimentConfig
from .runner import config_digest


@dataclass(frozen=True)
class ControllerState:
    fast_error_K: float
    slow_error_K: float
    improvement_K: float
    turn_rate_per_s: float


class TwoTimescaleTurnController:
    """Change persistence using only a stream of local temperature readings."""

    def __init__(
        self,
        config: MemoryControllerConfig,
        preferred_temperature_K: float,
        seed: int,
    ):
        self._config = config
        self._preferred_K = preferred_temperature_K
        self._rng = random.Random(seed)
        self._last_sample_time_s: Optional[float] = None
        self._fast_error_K: Optional[float] = None
        self._slow_error_K: Optional[float] = None

    def observe(self, reading: SensorReading) -> ControllerState:
        error = abs(reading.temperature_K - self._preferred_K)
        if self._last_sample_time_s is None:
            self._fast_error_K = error
            self._slow_error_K = error
        else:
            elapsed = reading.sample_time_s - self._last_sample_time_s
            if elapsed <= 0:
                raise ValueError("sensor sample times must increase strictly")
            fast_alpha = 1.0 - math.exp(-elapsed / self._config.fast_memory_tau_s)
            slow_alpha = 1.0 - math.exp(-elapsed / self._config.slow_memory_tau_s)
            self._fast_error_K += fast_alpha * (error - self._fast_error_K)
            self._slow_error_K += slow_alpha * (error - self._slow_error_K)
        self._last_sample_time_s = reading.sample_time_s
        return self.state

    @property
    def state(self) -> ControllerState:
        if self._fast_error_K is None or self._slow_error_K is None:
            raise RuntimeError("controller requires a sensor reading before use")
        improvement = self._slow_error_K - self._fast_error_K
        if abs(improvement) <= self._config.response_deadband_K:
            effective_improvement = 0.0
        else:
            effective_improvement = math.copysign(
                abs(improvement) - self._config.response_deadband_K, improvement
            )
        rate = (
            self._config.baseline_turn_rate_per_s
            - self._config.response_gain_per_K_s * effective_improvement
        )
        rate = min(
            self._config.max_turn_rate_per_s,
            max(self._config.min_turn_rate_per_s, rate),
        )
        return ControllerState(
            fast_error_K=self._fast_error_K,
            slow_error_K=self._slow_error_K,
            improvement_K=improvement,
            turn_rate_per_s=rate,
        )

    def should_turn(self, control_interval_s: float) -> bool:
        probability = -math.expm1(-self.state.turn_rate_per_s * control_interval_s)
        return self._rng.random() < probability


def _initial_direction(mode: str, rng: random.Random) -> int:
    if mode == "positive":
        return 1
    if mode == "negative":
        return -1
    return 1 if rng.random() < 0.5 else -1


def _reflect(position_m: float, direction: int, extent_m: float) -> tuple[float, int]:
    """Reflect a one-dimensional step, including unusually large test steps."""
    period_m = 2.0 * extent_m
    folded_m = position_m % period_m
    if folded_m == 0.0:
        return 0.0, 1
    if folded_m == extent_m:
        return extent_m, -1
    if folded_m < extent_m:
        return folded_m, direction
    return period_m - folded_m, -direction


def _in_comfort(temperature_K: float, config: Phase1ExperimentConfig) -> bool:
    numerical_tolerance_K = 1e-12 * max(
        1.0, abs(temperature_K), abs(config.temperature.preferred_K)
    )
    return (
        abs(temperature_K - config.temperature.preferred_K)
        <= config.temperature.comfort_half_width_K + numerical_tolerance_K
    )


def _advance_reflecting(
    position_m: float, direction: int, distance_m: float, extent_m: float
) -> Tuple[float, int, List[Tuple[float, float]]]:
    """Advance exactly and retain each constant-direction segment for event integration."""
    segments: List[Tuple[float, float]] = []
    remaining_m = distance_m
    while remaining_m > 0.0:
        boundary_m = extent_m if direction > 0 else 0.0
        available_m = abs(boundary_m - position_m)
        if available_m == 0.0:
            direction *= -1
            continue
        travelled_m = min(remaining_m, available_m)
        next_position_m = position_m + direction * travelled_m
        segments.append((position_m, next_position_m))
        position_m = next_position_m
        remaining_m -= travelled_m
        if position_m == boundary_m:
            direction *= -1
    return position_m, direction, segments


def _comfort_fraction_on_segment(
    start_temperature_K: float,
    end_temperature_K: float,
    config: Phase1ExperimentConfig,
) -> Tuple[float, Optional[float]]:
    """Return the segment fraction in comfort and its first entry fraction."""
    tolerance_K = 1e-12 * max(
        1.0,
        abs(start_temperature_K),
        abs(end_temperature_K),
        abs(config.temperature.preferred_K),
    )
    lower_K = (
        config.temperature.preferred_K
        - config.temperature.comfort_half_width_K
        - tolerance_K
    )
    upper_K = (
        config.temperature.preferred_K
        + config.temperature.comfort_half_width_K
        + tolerance_K
    )
    change_K = end_temperature_K - start_temperature_K
    if change_K == 0.0:
        if lower_K <= start_temperature_K <= upper_K:
            return 1.0, 0.0
        return 0.0, None
    first = (lower_K - start_temperature_K) / change_K
    second = (upper_K - start_temperature_K) / change_K
    interval_start = max(0.0, min(first, second))
    interval_end = min(1.0, max(first, second))
    if interval_end < interval_start:
        return 0.0, None
    return max(0.0, interval_end - interval_start), interval_start


def run_phase1_experiment(
    config: Phase1ExperimentConfig, *, record_trajectory: bool = True
) -> Dict[str, Any]:
    """Run one reproducible trajectory with separated physics, sensing and control clocks."""
    field = LinearTemperatureField.from_config(config.temperature)
    sensor = LocalTemperatureSensor(config.sensor, config.seeds.sensor)
    controller = TwoTimescaleTurnController(
        config.thermotaxis_controller,
        config.temperature.preferred_K,
        config.seeds.controller,
    )
    initial_rng = random.Random(config.seeds.initial_state)
    direction = _initial_direction(config.point_model.initial_direction, initial_rng)
    position_m = config.point_model.initial_x_m

    physics_steps = int(round(config.time.duration_s / config.time.physics_dt_s))
    sensor_stride = int(round(config.time.sensor_dt_s / config.time.physics_dt_s))
    control_stride = int(round(config.time.control_dt_s / config.time.physics_dt_s))
    true_temperature = field.temperature_K((position_m, 0.0), 0.0)
    reading = sensor.sample(field, (position_m, 0.0), 0.0)
    controller_state = controller.observe(reading)
    sensor_count = 1
    control_count = 0
    turn_count = 0
    first_entry_time_s: Optional[float] = None
    initial_in_comfort = _in_comfort(true_temperature, config)
    if initial_in_comfort:
        first_entry_time_s = 0.0

    trajectory: List[Dict[str, Any]] = []
    if record_trajectory:
        trajectory.append({
            "time_s": 0.0,
            "position_m": position_m,
            "direction": direction,
            "true_temperature_K": true_temperature,
            "measured_temperature_K": reading.temperature_K,
            "measured_sample_time_s": reading.sample_time_s,
            "sensor_sampled": True,
            **asdict(controller_state),
        })
    comfort_time_s = 0.0
    squared_error_integral_K2_s = 0.0

    for step in range(1, physics_steps + 1):
        time_s = step * config.time.physics_dt_s
        position_m, direction, motion_segments = _advance_reflecting(
            position_m,
            direction,
            config.point_model.speed_m_s * config.time.physics_dt_s,
            config.domain.x_extent_m,
        )
        true_temperature = field.temperature_K((position_m, 0.0), time_s)
        distance_before_segment_m = 0.0
        for segment_start_m, segment_end_m in motion_segments:
            segment_distance_m = abs(segment_end_m - segment_start_m)
            start_temperature_K = field.temperature_K((segment_start_m, 0.0), time_s)
            end_temperature_K = field.temperature_K((segment_end_m, 0.0), time_s)
            comfort_fraction, entry_fraction = _comfort_fraction_on_segment(
                start_temperature_K, end_temperature_K, config
            )
            comfort_time_s += (
                comfort_fraction * segment_distance_m / config.point_model.speed_m_s
            )
            segment_time_s = segment_distance_m / config.point_model.speed_m_s
            start_error_K = start_temperature_K - config.temperature.preferred_K
            end_error_K = end_temperature_K - config.temperature.preferred_K
            squared_error_integral_K2_s += segment_time_s * (
                start_error_K * start_error_K
                + start_error_K * end_error_K
                + end_error_K * end_error_K
            ) / 3.0
            if first_entry_time_s is None and entry_fraction is not None:
                first_entry_time_s = (
                    time_s
                    - config.time.physics_dt_s
                    + (
                        distance_before_segment_m + entry_fraction * segment_distance_m
                    )
                    / config.point_model.speed_m_s
                )
            distance_before_segment_m += segment_distance_m
        sensor_sampled = False
        if step % sensor_stride == 0:
            reading = sensor.sample(field, (position_m, 0.0), time_s)
            controller_state = controller.observe(reading)
            sensor_count += 1
            sensor_sampled = True
        if step % control_stride == 0:
            if controller.should_turn(config.time.control_dt_s):
                direction *= -1
                turn_count += 1
            control_count += 1

        if record_trajectory:
            trajectory.append({
                "time_s": time_s,
                "position_m": position_m,
                "direction": direction,
                "true_temperature_K": true_temperature,
                "measured_temperature_K": reading.temperature_K,
                "measured_sample_time_s": reading.sample_time_s,
                "sensor_sampled": sensor_sampled,
                **asdict(controller_state),
            })

    result = {
        "result_type": "phase1_point_thermotaxis",
        "scientific_result": True,
        "config_sha256": config_digest(config),
        "clock_counts": {
            "physics_steps": physics_steps,
            "sensor_samples_including_t0": sensor_count,
            "control_updates": control_count,
        },
        "metrics": {
            "first_entry_time_s": first_entry_time_s,
            "entered_comfort_region": first_entry_time_s is not None,
            "comfort_residence_fraction": min(
                1.0, max(0.0, comfort_time_s / config.time.duration_s)
            ),
            "rms_temperature_error_K": math.sqrt(
                squared_error_integral_K2_s / config.time.duration_s
            ),
            "turn_count": turn_count,
        },
        "controller_observations": list(config.controller_observations),
    }
    if record_trajectory:
        result["trajectory"] = trajectory
    return result
