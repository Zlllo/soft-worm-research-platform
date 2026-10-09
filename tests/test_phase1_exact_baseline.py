"""Independent event/geometry checks of the absorbing discrete baseline."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
spec = importlib.util.spec_from_file_location("sensitivity_analysis", ROOT / "analysis/phase1_theory_sensitivity.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_zero_rate_ballistic_reference_has_exact_unrestricted_mean():
    early = analysis.discrete_survival(.2, 50., rate=0)
    complete = analysis.discrete_survival(.2, 90., rate=0)
    assert early["arrival_probability"] == pytest.approx(.5)
    assert early["rmst_s"] == pytest.approx((47.5+50)/2)
    assert complete["arrival_probability"] == pytest.approx(1)
    assert complete["rmst_s"] == pytest.approx((47.5+77.5)/2)
    assert len(complete["event_masses"]) == 2
    for actual, expected in zip(complete["event_masses"], [[47.5, .5], [77.5, .5]]):
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize("dt", [.1, .2, .5, 1.])
def test_exact_baseline_survival_area_matches_rmst_including_fractional_hits(dt):
    result = analysis.discrete_survival(dt, 180.)
    # Integrate through actual fractional movement hit times, not grid endpoints.
    previous = 0.
    survival = 1.
    area = 0.
    for time, mass in result["event_masses"]:
        assert time >= previous
        area += (time-previous)*survival
        survival -= mass
        previous = time
    area += (180.-previous)*survival
    assert area == pytest.approx(result["rmst_s"], abs=3e-10)
    assert survival == pytest.approx(result["survival_probability"][-1], abs=2e-12)
    assert result["arrival_probability"] + survival == pytest.approx(1., abs=2e-12)
