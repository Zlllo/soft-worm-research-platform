"""Versioned experiment configuration and model-contract validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple


SCHEMA_VERSION = 1
ALLOWED_OBSERVATIONS = ("temperature_K", "sample_time_s")


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


def _integer_multiple(slower_name: str, slower: float, physics_dt: float) -> None:
    ratio = slower / physics_dt
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{slower_name} must be an integer multiple of physics_dt_s")


@dataclass(frozen=True)
class DomainConfig:
    dimension: int
    x_extent_m: float
    y_extent_m: float
    boundary: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainConfig":
        result = cls(
            dimension=int(data["dimension"]),
            x_extent_m=_positive("domain.x_extent_m", data["x_extent_m"]),
            y_extent_m=_positive("domain.y_extent_m", data["y_extent_m"]),
            boundary=str(data["boundary"]),
        )
        if result.dimension != 2:
            raise ValueError("phase 0 fixes the simulation domain to two dimensions")
        if result.boundary not in {"unbounded_probe", "reflecting", "periodic_y"}:
            raise ValueError("unsupported domain.boundary")
        return result


@dataclass(frozen=True)
class TemperatureConfig:
    field_type: str
    preferred_K: float
    comfort_half_width_K: float
    reference_K: float
    gradient_K_per_m: Tuple[float, float]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemperatureConfig":
        gradient = tuple(float(v) for v in data["gradient_K_per_m"])
        if len(gradient) != 2 or not all(math.isfinite(v) for v in gradient):
            raise ValueError("temperature.gradient_K_per_m must contain two finite values")
        result = cls(
            field_type=str(data["field_type"]),
            preferred_K=float(data["preferred_K"]),
            comfort_half_width_K=_positive(
                "temperature.comfort_half_width_K", data["comfort_half_width_K"]
            ),
            reference_K=float(data["reference_K"]),
            gradient_K_per_m=(gradient[0], gradient[1]),
        )
        if result.field_type != "linear_static":
            raise ValueError("phase 0 supports only the explicit linear_static field contract")
        if not math.isfinite(result.preferred_K) or not math.isfinite(result.reference_K):
            raise ValueError("temperature values must be finite")
        if math.hypot(*result.gradient_K_per_m) == 0:
            raise ValueError("temperature gradient must be non-zero")
        return result


@dataclass(frozen=True)
class MediumConfig:
    model: str
    density_kg_m3: float
    dynamic_viscosity_Pa_s: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MediumConfig":
        result = cls(
            model=str(data["model"]),
            density_kg_m3=_positive("medium.density_kg_m3", data["density_kg_m3"]),
            dynamic_viscosity_Pa_s=_positive(
                "medium.dynamic_viscosity_Pa_s", data["dynamic_viscosity_Pa_s"]
            ),
        )
        if result.model != "newtonian_viscous":
            raise ValueError("phase 0 fixes the working medium to newtonian_viscous")
        return result


@dataclass(frozen=True)
class BodyReferenceConfig:
    length_m: float
    speed_m_s: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BodyReferenceConfig":
        return cls(
            length_m=_positive("body_reference.length_m", data["length_m"]),
            speed_m_s=_positive("body_reference.speed_m_s", data["speed_m_s"]),
        )


@dataclass(frozen=True)
class TimeConfig:
    physics_dt_s: float
    sensor_dt_s: float
    control_dt_s: float
    duration_s: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeConfig":
        result = cls(
            physics_dt_s=_positive("time.physics_dt_s", data["physics_dt_s"]),
            sensor_dt_s=_positive("time.sensor_dt_s", data["sensor_dt_s"]),
            control_dt_s=_positive("time.control_dt_s", data["control_dt_s"]),
            duration_s=_positive("time.duration_s", data["duration_s"]),
        )
        for name, value in (
            ("sensor_dt_s", result.sensor_dt_s),
            ("control_dt_s", result.control_dt_s),
            ("duration_s", result.duration_s),
        ):
            _integer_multiple(name, value, result.physics_dt_s)
        if result.sensor_dt_s < result.physics_dt_s or result.control_dt_s < result.physics_dt_s:
            raise ValueError("sensor and control clocks cannot be faster than the physics clock")
        return result


@dataclass(frozen=True)
class SensorConfig:
    location: str
    noise_model: str
    noise_std_K: float
    noise_correlation_s: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorConfig":
        result = cls(
            location=str(data["location"]),
            noise_model=str(data["noise_model"]),
            noise_std_K=_nonnegative("sensor.noise_std_K", data["noise_std_K"]),
            noise_correlation_s=_nonnegative(
                "sensor.noise_correlation_s", data["noise_correlation_s"]
            ),
        )
        if result.location != "head":
            raise ValueError("phase 0 permits only a single head temperature sensor")
        if result.noise_model != "independent_gaussian":
            raise ValueError("phase 0 permits only independent_gaussian measurement noise")
        if result.noise_correlation_s != 0:
            raise ValueError("independent_gaussian noise requires noise_correlation_s = 0")
        return result


@dataclass(frozen=True)
class SeedConfig:
    environment: int
    sensor: int
    initial_state: int
    controller: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedConfig":
        values = {name: int(data[name]) for name in ("environment", "sensor", "initial_state", "controller")}
        if any(value < 0 for value in values.values()):
            raise ValueError("all seeds must be non-negative integers")
        return cls(**values)


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    name: str
    purpose: str
    domain: DomainConfig
    temperature: TemperatureConfig
    medium: MediumConfig
    body_reference: BodyReferenceConfig
    time: TimeConfig
    sensor: SensorConfig
    seeds: SeedConfig
    controller_observations: Tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        observations = tuple(str(v) for v in data["controller_observations"])
        if observations != ALLOWED_OBSERVATIONS:
            raise ValueError(
                "controller_observations must be exactly temperature_K and sample_time_s; "
                "true gradient, position, preferred-region coordinates and field access are forbidden"
            )
        result = cls(
            schema_version=int(data["schema_version"]),
            name=str(data["name"]),
            purpose=str(data["purpose"]),
            domain=DomainConfig.from_dict(data["domain"]),
            temperature=TemperatureConfig.from_dict(data["temperature"]),
            medium=MediumConfig.from_dict(data["medium"]),
            body_reference=BodyReferenceConfig.from_dict(data["body_reference"]),
            time=TimeConfig.from_dict(data["time"]),
            sensor=SensorConfig.from_dict(data["sensor"]),
            seeds=SeedConfig.from_dict(data["seeds"]),
            controller_observations=observations,
        )
        if result.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {result.schema_version}")
        if not result.name.strip() or not result.purpose.strip():
            raise ValueError("name and purpose must be non-empty")
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


def load_config(path: Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("top-level configuration must be a JSON object")
    return ExperimentConfig.from_dict(data)
