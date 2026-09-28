"""Paired multi-seed comparison for the phase-1 thermotaxis mechanism."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
from typing import Any, Dict, List

from . import __version__
from .phase1 import run_phase1_experiment
from .phase1_config import Phase1ExperimentConfig
from .phase1_runner import _git_revision
from .runner import config_digest


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _summarize_metrics(
    records: List[Dict[str, Any]], duration_s: float
) -> Dict[str, Any]:
    reached_times = [
        float(record["first_entry_time_s"])
        for record in records
        if record["entered_comfort_region"]
    ]
    restricted_times = [
        min(float(record["first_entry_time_s"]), duration_s)
        if record["entered_comfort_region"]
        else duration_s
        for record in records
    ]
    reached_count = len(reached_times)
    return {
        "replicate_count": len(records),
        "reached_count": reached_count,
        "reached_rate": reached_count / len(records),
        "conditional_mean_first_entry_time_s": (
            _mean(reached_times) if reached_times else None
        ),
        "restricted_mean_first_entry_time_s": _mean(restricted_times),
        "mean_comfort_residence_fraction": _mean(
            [float(record["comfort_residence_fraction"]) for record in records]
        ),
        "mean_rms_temperature_error_K": _mean(
            [float(record["rms_temperature_error_K"]) for record in records]
        ),
        "mean_turn_count": _mean([float(record["turn_count"]) for record in records]),
    }


def run_phase1_paired_ensemble(
    config: Phase1ExperimentConfig, replicates: int
) -> Dict[str, Any]:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")

    paired_records: List[Dict[str, Any]] = []
    response_metrics: List[Dict[str, Any]] = []
    no_response_metrics: List[Dict[str, Any]] = []
    no_response_controller = replace(
        config.thermotaxis_controller, response_gain_per_K_s=0.0
    )

    for replicate in range(replicates):
        seeds = replace(
            config.seeds,
            sensor=config.seeds.sensor + replicate,
            initial_state=config.seeds.initial_state + replicate,
            controller=config.seeds.controller + replicate,
        )
        response_config = replace(config, seeds=seeds)
        no_response_config = replace(
            response_config, thermotaxis_controller=no_response_controller
        )
        response = run_phase1_experiment(response_config, record_trajectory=False)["metrics"]
        no_response = run_phase1_experiment(
            no_response_config, record_trajectory=False
        )["metrics"]
        response_metrics.append(response)
        no_response_metrics.append(no_response)
        paired_records.append(
            {
                "replicate": replicate,
                "seeds": asdict(seeds),
                "response": response,
                "no_response": no_response,
            }
        )

    return {
        "result_type": "phase1_paired_ensemble",
        "scientific_result": True,
        "config_sha256": config_digest(config),
        "replicates": replicates,
        "pairing": {
            "replicate_seed_rule": (
                "sensor, initial_state, and controller base seeds plus replicate index; "
                "environment seed unchanged"
            ),
            "shared_within_pair": [
                "environment",
                "sensor",
                "initial_state",
                "controller",
            ],
            "control_change": "response_gain_per_K_s = 0",
        },
        "restricted_mean_definition": (
            "mean(min(first_entry_time_s, duration_s)); an unreached replicate contributes duration_s"
        ),
        "groups": {
            "response": _summarize_metrics(response_metrics, config.time.duration_s),
            "no_response": _summarize_metrics(
                no_response_metrics, config.time.duration_s
            ),
        },
        "paired_replicates": paired_records,
    }


def write_phase1_ensemble_bundle(
    config: Phase1ExperimentConfig,
    replicates: int,
    output_root: Path,
    repository: Path,
) -> Path:
    """Write a compact, reproducible paired-ensemble result bundle.

    The summary intentionally contains per-replicate metrics rather than full
    trajectories.  This keeps the ensemble artifact suitable for parameter
    sweeps while retaining the paired observations needed for later analysis.
    """
    summary = run_phase1_paired_ensemble(config, replicates)
    run_id = summary["config_sha256"][:12]
    run_dir = (
        Path(output_root)
        / f"{config.name}-ensemble-{replicates}-replicates-{run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "thermotaxis_core_version": __version__,
        "config_sha256": summary["config_sha256"],
        "git_revision": _git_revision(Path(repository)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "purpose": config.purpose,
        "result_type": summary["result_type"],
        "replicates": replicates,
    }
    for name, payload in (
        ("config.resolved.json", config.to_dict()),
        ("manifest.json", manifest),
        ("summary.json", summary),
    ):
        with (run_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    return run_dir
