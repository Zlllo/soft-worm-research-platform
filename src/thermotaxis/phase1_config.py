"""Strict configuration schema for the phase-1 point-walker experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

from .config import (
    ALLOWED_OBSERVATIONS,
    BodyReferenceConfig,
    MediumConfig,
    SeedConfig,
    SensorConfig,
    TemperatureConfig,
    TimeConfig,
    _nonnegative,
    _positive,
    _require_keys,
    _strict_int,
)


PHASE1_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Phase1DomainConfig:
    dimension: int
    x_extent_m: float
    y_extent_m: float
    boundary: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase1DomainConfig":
        _require_keys("domain", data, ("dimension", "x_extent_m", "y_extent_m", "boundary"))
        result = cls(
            dimension=_strict_int("domain.dimension", data["dimension"]),
            x_extent_m=_positive("domain.x_extent_m", data["x_extent_m"]),
            y_extent_m=_positive("domain.y_extent_m", data["y_extent_m"]),
            boundary=str(data["boundary"]),
        )
        if result.dimension != 1:
            raise ValueError("phase 1 currently requires domain.dimension = 1")
        if result.boundary != "reflecting":
            raise ValueError("phase 1 currently requires a reflecting boundary")
        return result


@dataclass(frozen=True)
class PointModelConfig:
    initial_x_m: float
    speed_m_s: float
    initial_direction: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PointModelConfig":
        _require_keys(
            "point_model", data, ("initial_x_m", "speed_m_s", "initial_direction")
        )
        result = cls(
            initial_x_m=float(data["initial_x_m"]),
            speed_m_s=_positive("point_model.speed_m_s", data["speed_m_s"]),
            initial_direction=str(data["initial_direction"]),
        )
        if not math.isfinite(result.initial_x_m) or result.initial_x_m < 0:
            raise ValueError("point_model.initial_x_m must be finite and >= 0")
        if result.initial_direction not in {"random", "positive", "negative"}:
            raise ValueError(
                "point_model.initial_direction must be random, positive, or negative"
            )
        return result


@dataclass(frozen=True)
class MemoryControllerConfig:
    fast_memory_tau_s: float
    slow_memory_tau_s: float
    baseline_turn_rate_per_s: float
    min_turn_rate_per_s: float
    max_turn_rate_per_s: float
    response_gain_per_K_s: float
    response_deadband_K: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryControllerConfig":
        _require_keys(
            "thermotaxis_controller",
            data,
            (
                "fast_memory_tau_s",
                "slow_memory_tau_s",
                "baseline_turn_rate_per_s",
                "min_turn_rate_per_s",
                "max_turn_rate_per_s",
                "response_gain_per_K_s",
                "response_deadband_K",
            ),
        )
        result = cls(
            fast_memory_tau_s=_positive(
                "thermotaxis_controller.fast_memory_tau_s", data["fast_memory_tau_s"]
            ),
            slow_memory_tau_s=_positive(
                "thermotaxis_controller.slow_memory_tau_s", data["slow_memory_tau_s"]
            ),
            baseline_turn_rate_per_s=_nonnegative(
                "thermotaxis_controller.baseline_turn_rate_per_s",
                data["baseline_turn_rate_per_s"],
            ),
            min_turn_rate_per_s=_nonnegative(
                "thermotaxis_controller.min_turn_rate_per_s",
                data["min_turn_rate_per_s"],
            ),
            max_turn_rate_per_s=_positive(
                "thermotaxis_controller.max_turn_rate_per_s",
                data["max_turn_rate_per_s"],
            ),
            response_gain_per_K_s=_nonnegative(
                "thermotaxis_controller.response_gain_per_K_s",
                data["response_gain_per_K_s"],
            ),
            response_deadband_K=_nonnegative(
                "thermotaxis_controller.response_deadband_K",
                data["response_deadband_K"],
            ),
        )
        if result.slow_memory_tau_s <= result.fast_memory_tau_s:
            raise ValueError("slow_memory_tau_s must be greater than fast_memory_tau_s")
        if not (
            result.min_turn_rate_per_s
            <= result.baseline_turn_rate_per_s
            <= result.max_turn_rate_per_s
        ):
            raise ValueError("turn rates must satisfy min <= baseline <= max")
        return result


@dataclass(frozen=True)
class Phase1ExperimentConfig:
    schema_version: int
    name: str
    purpose: str
    domain: Phase1DomainConfig
    temperature: TemperatureConfig
    medium: MediumConfig
    body_reference: BodyReferenceConfig
    time: TimeConfig
    sensor: SensorConfig
    seeds: SeedConfig
    controller_observations: Tuple[str, ...]
    point_model: PointModelConfig
    thermotaxis_controller: MemoryControllerConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase1ExperimentConfig":
        _require_keys(
            "phase1 experiment",
            data,
            (
                "schema_version",
                "name",
                "purpose",
                "domain",
                "temperature",
                "medium",
                "body_reference",
                "time",
                "sensor",
                "seeds",
                "controller_observations",
                "point_model",
                "thermotaxis_controller",
            ),
        )
        observations = tuple(str(value) for value in data["controller_observations"])
        if observations != ALLOWED_OBSERVATIONS:
            raise ValueError(
                "controller_observations must be exactly temperature_K and sample_time_s; "
                "true gradient, position, preferred-region coordinates and field access are forbidden"
            )
        result = cls(
            schema_version=_strict_int("schema_version", data["schema_version"]),
            name=str(data["name"]),
            purpose=str(data["purpose"]),
            domain=Phase1DomainConfig.from_dict(data["domain"]),
            temperature=TemperatureConfig.from_dict(data["temperature"]),
            medium=MediumConfig.from_dict(data["medium"]),
            body_reference=BodyReferenceConfig.from_dict(data["body_reference"]),
            time=TimeConfig.from_dict(data["time"]),
            sensor=SensorConfig.from_dict(data["sensor"]),
            seeds=SeedConfig.from_dict(data["seeds"]),
            controller_observations=observations,
            point_model=PointModelConfig.from_dict(data["point_model"]),
            thermotaxis_controller=MemoryControllerConfig.from_dict(
                data["thermotaxis_controller"]
            ),
        )
        if result.schema_version != PHASE1_SCHEMA_VERSION:
            raise ValueError(f"unsupported phase-1 schema_version {result.schema_version}")
        if not result.name.strip() or not result.purpose.strip():
            raise ValueError("name and purpose must be non-empty")
        if result.point_model.initial_x_m > result.domain.x_extent_m:
            raise ValueError("point_model.initial_x_m must lie inside the domain")
        if result.temperature.gradient_K_per_m[1] != 0.0:
            raise ValueError("phase 1 requires zero y temperature gradient")
        return result

    @property
    def reynolds_number(self) -> float:
        return (
            self.medium.density_kg_m3
            * self.body_reference.speed_m_s
            * self.body_reference.length_m
            / self.medium.dynamic_viscosity_Pa_s
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_phase1_config(path: Path) -> Phase1ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("top-level configuration must be a JSON object")
    return Phase1ExperimentConfig.from_dict(data)
