from dataclasses import replace
import json
from pathlib import Path

import pytest

from thermotaxis.config import ExperimentConfig, load_config
from thermotaxis.contracts import SensorReading
from thermotaxis.runner import run_phase0_probe, write_result_bundle


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "configs" / "phase0_reference.json"


def test_reference_config_has_explicit_low_reynolds_regime_and_clocks():
    config = load_config(REFERENCE)
    assert config.reynolds_number == pytest.approx(0.002)
    assert config.time.sensor_dt_s / config.time.physics_dt_s == pytest.approx(10)
    assert config.time.control_dt_s / config.time.physics_dt_s == pytest.approx(20)


def test_probe_is_reproducible_and_observation_is_narrow():
    config = load_config(REFERENCE)
    first = run_phase0_probe(config)
    second = run_phase0_probe(config)
    assert first == second
    assert first["scientific_result"] is False
    assert first["controller_observations"] == ["temperature_K", "sample_time_s"]
    assert set(SensorReading.__dataclass_fields__) == {"temperature_K", "sample_time_s"}


def test_sensor_seed_changes_measurements_but_not_truth_or_physical_scale():
    config = load_config(REFERENCE)
    changed = replace(config, seeds=replace(config.seeds, sensor=config.seeds.sensor + 1))
    original_result = run_phase0_probe(config)
    changed_result = run_phase0_probe(changed)
    assert original_result["probe"]["true_temperature_K"] == changed_result["probe"]["true_temperature_K"]
    assert original_result["derived"] == changed_result["derived"]
    assert original_result["probe"]["measured_temperature_K"] != changed_result["probe"]["measured_temperature_K"]


def test_privileged_observation_and_incommensurate_clock_are_rejected():
    raw = json.loads(REFERENCE.read_text(encoding="utf-8"))
    raw["controller_observations"].append("true_gradient_K_per_m")
    with pytest.raises(ValueError, match="forbidden"):
        ExperimentConfig.from_dict(raw)

    raw = json.loads(REFERENCE.read_text(encoding="utf-8"))
    raw["time"]["sensor_dt_s"] = 0.105
    with pytest.raises(ValueError, match="integer multiple"):
        ExperimentConfig.from_dict(raw)


def test_result_bundle_contains_config_manifest_and_result(tmp_path):
    config = load_config(REFERENCE)
    run_dir = write_result_bundle(config, tmp_path, ROOT)
    assert {path.name for path in run_dir.iterdir()} == {
        "config.resolved.json",
        "manifest.json",
        "result.json",
    }
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result["config_sha256"] == manifest["config_sha256"]
    assert result["clock_counts"] == {
        "physics_steps": 200,
        "sensor_samples_including_t0": 21,
        "control_updates": 10,
    }
