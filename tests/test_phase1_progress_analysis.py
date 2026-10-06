"""Independent checks of finite-window, paired progress-analysis contracts."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_path = Path(__file__).resolve().parents[1] / "analysis/phase1_progress_scan.py"
_spec = importlib.util.spec_from_file_location("progress_analysis", _path)
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)


def test_survival_handles_initial_event_endpoint_event_and_censoring():
    # Four subjects: immediate entry, 30 s entry, endpoint entry, no entry.
    curve = analysis.survival(np.array([0., 30., 180., 180.]), np.array([1., 1., 1., 0.]))
    assert curve["time_s"] == [0., 30., 180.]
    assert curve["survival_probability"] == [.75, .5, .25]
    assert curve["rmst_s"] == pytest.approx(97.5)
    assert curve["events"] == 3
    assert curve["censored"] == 1


def test_wilson_keeps_uncertainty_when_every_subject_succeeds():
    lower, upper = analysis.wilson(128, 128)
    assert lower == pytest.approx(.9708630437264916)
    assert upper == pytest.approx(1.)
    assert upper > lower


def test_bootstrap_resamples_whole_pairs_preserving_constant_pair_effect():
    records = []
    for entry in [30., 60., 90., 120.]:
        baseline = {"entered_comfort_region": True, "first_entry_time_s": entry,
                    "comfort_residence_fraction": .2, "rms_temperature_error_K": 1., "turn_count": 4}
        response = dict(baseline, first_entry_time_s=entry-10., comfort_residence_fraction=.3)
        records.append({"response": response, "no_response": baseline})
    summary = analysis.summarize({"paired_replicates": records}, np.random.default_rng(923))
    result = summary["paired_difference"]["restricted_first_entry_s"]
    assert result["mean"] == -10.
    assert result["bootstrap95"] == [-10., -10.]
