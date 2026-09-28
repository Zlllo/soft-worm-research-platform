from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from thermotaxis.config import TimeConfig
from thermotaxis.cli import main
from thermotaxis.contracts import SensorReading
from thermotaxis.phase1 import TwoTimescaleTurnController, _reflect, run_phase1_experiment
from thermotaxis.phase1_config import (
    MemoryControllerConfig,
    Phase1ExperimentConfig,
    load_phase1_config,
)
from thermotaxis.phase1_ensemble import (
    run_phase1_paired_ensemble,
    write_phase1_ensemble_bundle,
)
from thermotaxis.phase1_runner import write_phase1_result_bundle


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "configs" / "phase1_reference.json"


def test_phase1_config_is_strict_and_uses_commensurate_clocks():
    config = load_phase1_config(REFERENCE)
    assert config.domain.dimension == 1
    assert config.time.duration_s / config.time.sensor_dt_s == pytest.approx(1800)
    assert config.time.duration_s / config.time.control_dt_s == pytest.approx(900)

    raw = json.loads(REFERENCE.read_text(encoding="utf-8"))
    raw["point_model"]["undeclared_hint"] = 1
    with pytest.raises(ValueError, match="unknown"):
        Phase1ExperimentConfig.from_dict(raw)


def test_two_timescale_memory_reduces_turn_rate_while_error_improves():
    parameters = MemoryControllerConfig(
        fast_memory_tau_s=0.2,
        slow_memory_tau_s=2.0,
        baseline_turn_rate_per_s=0.3,
        min_turn_rate_per_s=0.05,
        max_turn_rate_per_s=0.8,
        response_gain_per_K_s=2.0,
        response_deadband_K=0.0,
    )
    controller = TwoTimescaleTurnController(parameters, preferred_temperature_K=293.0, seed=1)
    initial = controller.observe(SensorReading(temperature_K=291.0, sample_time_s=0.0))
    improving = controller.observe(SensorReading(temperature_K=292.0, sample_time_s=1.0))
    worsening = controller.observe(SensorReading(temperature_K=290.0, sample_time_s=2.0))

    assert initial.turn_rate_per_s == pytest.approx(parameters.baseline_turn_rate_per_s)
    assert improving.improvement_K > 0
    assert improving.turn_rate_per_s == parameters.min_turn_rate_per_s
    assert worsening.improvement_K < 0
    assert worsening.turn_rate_per_s == parameters.max_turn_rate_per_s


def test_reflection_handles_multiple_domain_crossings_and_points_inward_at_walls():
    assert _reflect(0.08, 1, 0.03) == pytest.approx((0.02, 1))
    assert _reflect(-0.04, -1, 0.03) == pytest.approx((0.02, -1))
    assert _reflect(0.0, -1, 0.03) == pytest.approx((0.0, 1))
    assert _reflect(0.03, 1, 0.03) == pytest.approx((0.03, -1))


def test_phase1_run_is_reproducible_and_reports_required_metrics():
    config = load_phase1_config(REFERENCE)
    first = run_phase1_experiment(config)
    second = run_phase1_experiment(config)

    assert first == second
    assert first["clock_counts"] == {
        "physics_steps": 18000,
        "sensor_samples_including_t0": 1801,
        "control_updates": 900,
    }
    assert first["metrics"]["entered_comfort_region"] is True
    assert first["metrics"]["first_entry_time_s"] == pytest.approx(89.1)
    assert 0.0 <= first["metrics"]["comfort_residence_fraction"] <= 1.0
    assert first["metrics"]["rms_temperature_error_K"] > 0.0
    assert len(first["trajectory"]) == 18001
    assert first["controller_observations"] == ["temperature_K", "sample_time_s"]


def test_physics_timestep_does_not_change_sensor_or_control_schedule():
    config = load_phase1_config(REFERENCE)
    refined_time = TimeConfig(
        physics_dt_s=0.005,
        sensor_dt_s=config.time.sensor_dt_s,
        control_dt_s=config.time.control_dt_s,
        duration_s=config.time.duration_s,
    )
    refined = run_phase1_experiment(replace(config, time=refined_time))
    reference = run_phase1_experiment(config)

    assert refined["clock_counts"]["sensor_samples_including_t0"] == 1801
    assert refined["clock_counts"]["control_updates"] == 900
    assert refined["metrics"]["turn_count"] == reference["metrics"]["turn_count"]
    assert refined["trajectory"][-1]["position_m"] == pytest.approx(
        reference["trajectory"][-1]["position_m"], abs=1e-12
    )


def test_sensor_seed_changes_measurements_without_changing_motion_when_response_is_off():
    config = load_phase1_config(REFERENCE)
    no_response = replace(
        config,
        thermotaxis_controller=replace(
            config.thermotaxis_controller, response_gain_per_K_s=0.0
        ),
    )
    changed_sensor_seed = replace(
        no_response, seeds=replace(no_response.seeds, sensor=no_response.seeds.sensor + 1)
    )
    first = run_phase1_experiment(no_response)
    second = run_phase1_experiment(changed_sensor_seed)

    first_measurements = [row["measured_temperature_K"] for row in first["trajectory"]]
    second_measurements = [row["measured_temperature_K"] for row in second["trajectory"]]
    first_positions = [row["position_m"] for row in first["trajectory"]]
    second_positions = [row["position_m"] for row in second["trajectory"]]
    assert first_measurements != second_measurements
    assert first_positions == second_positions


def test_first_entry_and_metrics_use_true_temperature_not_noisy_reading():
    config = load_phase1_config(REFERENCE)
    deterministic_motion = replace(
        config,
        point_model=replace(
            config.point_model, initial_x_m=0.0115, initial_direction="positive"
        ),
        time=TimeConfig(
            physics_dt_s=0.01,
            sensor_dt_s=0.1,
            control_dt_s=0.2,
            duration_s=10.0,
        ),
        thermotaxis_controller=replace(
            config.thermotaxis_controller,
            baseline_turn_rate_per_s=0.0,
            min_turn_rate_per_s=0.0,
            response_gain_per_K_s=0.0,
        ),
    )
    result = run_phase1_experiment(deterministic_motion)

    assert result["metrics"]["first_entry_time_s"] == pytest.approx(5.0)
    assert result["metrics"]["comfort_residence_fraction"] == pytest.approx(0.5, abs=1e-8)
    expected_rms_K = math.sqrt((0.35**2 + 0.35 * 0.15 + 0.15**2) / 3.0)
    assert result["metrics"]["rms_temperature_error_K"] == pytest.approx(expected_rms_K)
    assert result["trajectory"][500]["true_temperature_K"] == pytest.approx(292.9)


def test_trajectory_marks_fresh_samples_and_detects_crossing_the_full_comfort_band():
    config = load_phase1_config(REFERENCE)
    result = run_phase1_experiment(config)
    assert result["trajectory"][0]["sensor_sampled"] is True
    assert result["trajectory"][1]["sensor_sampled"] is False
    assert result["trajectory"][10]["sensor_sampled"] is True
    assert result["trajectory"][9]["measured_sample_time_s"] == pytest.approx(0.0)
    assert result["trajectory"][10]["measured_sample_time_s"] == pytest.approx(0.1)

    raw = json.loads(REFERENCE.read_text(encoding="utf-8"))
    raw["point_model"]["speed_m_s"] = 0.5
    raw["time"] = {
        "physics_dt_s": 0.1,
        "sensor_dt_s": 0.1,
        "control_dt_s": 0.2,
        "duration_s": 0.2,
    }
    raw["point_model"]["initial_x_m"] = 0.003
    raw["point_model"]["initial_direction"] = "positive"
    crossing = run_phase1_experiment(Phase1ExperimentConfig.from_dict(raw))
    assert crossing["trajectory"][1]["true_temperature_K"] < 292.9
    assert crossing["metrics"]["entered_comfort_region"] is True
    assert crossing["metrics"]["first_entry_time_s"] == pytest.approx(0.019, abs=1e-10)
    assert 0.0 < crossing["metrics"]["comfort_residence_fraction"] <= 1.0


def test_phase1_result_bundle_contains_resolved_config_manifest_and_trajectory(tmp_path):
    config = load_phase1_config(REFERENCE)
    run_dir = write_phase1_result_bundle(config, tmp_path, ROOT)
    assert {path.name for path in run_dir.iterdir()} == {
        "config.resolved.json",
        "manifest.json",
        "result.json",
    }
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result["scientific_result"] is True
    assert result["result_type"] == "phase1_point_thermotaxis"
    assert result["config_sha256"] == manifest["config_sha256"]
    assert len(result["trajectory"]) == result["clock_counts"]["physics_steps"] + 1


@pytest.mark.parametrize("replicates", [True, False, 0, -1, 1.0, "1"])
def test_phase1_ensemble_requires_a_strict_positive_integer(replicates):
    config = load_phase1_config(REFERENCE)
    with pytest.raises(ValueError, match="positive integer"):
        run_phase1_paired_ensemble(config, replicates)


def test_phase1_ensemble_pairs_seed_streams_and_omits_trajectories():
    config = load_phase1_config(REFERENCE)
    short_config = replace(
        config,
        time=TimeConfig(
            physics_dt_s=0.01,
            sensor_dt_s=0.1,
            control_dt_s=0.2,
            duration_s=0.2,
        ),
    )
    summary = run_phase1_paired_ensemble(short_config, 3)

    assert summary["replicates"] == 3
    for index, record in enumerate(summary["paired_replicates"]):
        assert record["replicate"] == index
        assert record["seeds"] == {
            "environment": config.seeds.environment,
            "sensor": config.seeds.sensor + index,
            "initial_state": config.seeds.initial_state + index,
            "controller": config.seeds.controller + index,
        }
        assert set(record) == {"replicate", "seeds", "response", "no_response"}
    assert "trajectory" not in json.dumps(summary)


def test_phase1_ensemble_reports_unreached_conditional_and_restricted_means():
    config = load_phase1_config(REFERENCE)
    unreachable_config = replace(
        config,
        point_model=replace(
            config.point_model, initial_x_m=0.0, initial_direction="negative"
        ),
        time=TimeConfig(
            physics_dt_s=0.01,
            sensor_dt_s=0.1,
            control_dt_s=0.2,
            duration_s=0.2,
        ),
    )
    summary = run_phase1_paired_ensemble(unreachable_config, 2)

    for group in summary["groups"].values():
        assert group["reached_count"] == 0
        assert group["reached_rate"] == 0.0
        assert group["conditional_mean_first_entry_time_s"] is None
        assert group["restricted_mean_first_entry_time_s"] == pytest.approx(0.2)


def test_phase1_ensemble_bundle_and_cli_are_reproducible_and_compact(
    tmp_path, capsys
):
    config = load_phase1_config(REFERENCE)
    short_config = replace(
        config,
        time=TimeConfig(
            physics_dt_s=0.01,
            sensor_dt_s=0.1,
            control_dt_s=0.2,
            duration_s=0.2,
        ),
    )
    config_path = tmp_path / "phase1-short.json"
    config_path.write_text(
        json.dumps(short_config.to_dict(), ensure_ascii=False), encoding="utf-8"
    )

    first_dir = write_phase1_ensemble_bundle(
        short_config, 2, tmp_path / "direct-first", ROOT
    )
    second_dir = write_phase1_ensemble_bundle(
        short_config, 2, tmp_path / "direct-second", ROOT
    )
    assert "2-replicates" in first_dir.name
    assert {path.name for path in first_dir.iterdir()} == {
        "config.resolved.json",
        "manifest.json",
        "summary.json",
    }
    first_summary = json.loads((first_dir / "summary.json").read_text(encoding="utf-8"))
    second_summary = json.loads((second_dir / "summary.json").read_text(encoding="utf-8"))
    assert first_summary == second_summary
    assert "trajectory" not in json.dumps(first_summary)

    cli_root = tmp_path / "cli-first"
    assert main(
        [
            "phase1-ensemble",
            "--config",
            str(config_path),
            "--replicates",
            "2",
            "--output-root",
            str(cli_root),
            "--repository",
            str(ROOT),
        ]
    ) == 0
    cli_dir = Path(capsys.readouterr().out.strip())
    cli_summary = json.loads((cli_dir / "summary.json").read_text(encoding="utf-8"))
    assert cli_summary == first_summary

    with pytest.raises(SystemExit):
        main(
            [
                "phase1-ensemble",
                "--config",
                str(config_path),
                "--replicates",
                "0",
                "--output-root",
                str(tmp_path / "invalid"),
            ]
        )
