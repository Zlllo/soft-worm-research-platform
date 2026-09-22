"""Phase-0 contract probe and reproducible result bundle writer."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
from typing import Any, Dict

from . import __version__
from .config import ExperimentConfig
from .contracts import LinearTemperatureField, LocalTemperatureSensor


def canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_digest(config: ExperimentConfig) -> str:
    return hashlib.sha256(canonical_json(config.to_dict()).encode("utf-8")).hexdigest()


def _git_revision(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_phase0_probe(config: ExperimentConfig) -> Dict[str, Any]:
    """Exercise clocks, local observation and isolated sensor randomness.

    This is an infrastructure check, not a thermotaxis result.
    """
    field = LinearTemperatureField.from_config(config.temperature)
    sensor = LocalTemperatureSensor(config.sensor, config.seeds.sensor)
    position = (0.5 * config.domain.x_extent_m, 0.5 * config.domain.y_extent_m)
    sample_count = int(round(config.time.duration_s / config.time.sensor_dt_s)) + 1
    readings = [
        sensor.sample(field, position, index * config.time.sensor_dt_s)
        for index in range(sample_count)
    ]
    values = [reading.temperature_K for reading in readings]
    true_temperature = field.temperature_K(position, 0.0)
    noise = [value - true_temperature for value in values]
    mean_noise = sum(noise) / len(noise)
    variance = sum((value - mean_noise) ** 2 for value in noise) / len(noise)
    return {
        "result_type": "phase0_contract_probe",
        "scientific_result": False,
        "config_sha256": config_digest(config),
        "clock_counts": {
            "physics_steps": int(round(config.time.duration_s / config.time.physics_dt_s)),
            "sensor_samples_including_t0": sample_count,
            "control_updates": int(round(config.time.duration_s / config.time.control_dt_s)),
        },
        "probe": {
            "position_m": list(position),
            "true_temperature_K": true_temperature,
            "measured_temperature_K": values,
            "sample_time_s": [reading.sample_time_s for reading in readings],
            "noise_mean_K": mean_noise,
            "noise_rms_K": variance ** 0.5,
        },
        "derived": {"reynolds_number": config.reynolds_number},
        "controller_observations": list(config.controller_observations),
    }


def write_result_bundle(
    config: ExperimentConfig, output_root: Path, repository: Path
) -> Path:
    result = run_phase0_probe(config)
    run_id = result["config_sha256"][:12]
    run_dir = Path(output_root) / f"{config.name}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = config.to_dict()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "thermotaxis_core_version": __version__,
        "config_sha256": result["config_sha256"],
        "git_revision": _git_revision(Path(repository)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "purpose": config.purpose,
    }
    for name, payload in (
        ("config.resolved.json", resolved),
        ("manifest.json", manifest),
        ("result.json", result),
    ):
        with (run_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    return run_dir
