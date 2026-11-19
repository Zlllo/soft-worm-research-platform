"""Audit archived endpoints and summarize paired layouts and perturbations."""

import csv
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path("docs/reports/assets/pathway-specificity")
PARENT = Path("docs/reports/assets/pathway-stage1-2026-11-03/reproduction.zip")


def read_cases(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def lesions():
    summaries = {}
    with zipfile.ZipFile(PARENT) as z:
        for prefix, cohort in (("cases", "new"), ("matched", "holdout")):
            for mid in range(6):
                rows = read_cases(ROOT / f"{prefix}-{mid}.csv")
                spec = (rows[0]["interface"], int(rows[0]["training_seed"]))
                if prefix == "cases":
                    ref = json.loads(z.read(f"evidence/evaluation-{mid:03d}-All.json"))
                    for label, key in (("intact", "fresh"), ("AFD-AIY", "lesion")):
                        subset = [
                            x
                            for x in rows
                            if x["case_set"] == "archive" and x["treatment"] == label
                        ]
                        assert len(subset) == 32
                        assert [
                            (int(x["case_seed"]), bool(int(x["complete"]))) for x in subset
                        ] == [(x["seed"], x["complete_success"]) for x in ref[key]["cases"]]
                rows = [x for x in rows if x["case_set"] == cohort]
                expected = [int(x["case_seed"]) for x in rows if x["treatment"] == "intact"]
                assert len(set(expected)) == 64
                norms = []
                for label in dict.fromkeys(x["treatment"] for x in rows):
                    group = [x for x in rows if x["treatment"] == label]
                    assert [int(x["case_seed"]) for x in group] == expected
                    cells = (
                        summaries.setdefault(prefix, {})
                        .setdefault(spec[0], {})
                        .setdefault(label, {})
                    )
                    cells[spec[1]] = {
                        m: float(np.mean([float(x[m]) for x in group]))
                        for m in (
                            "complete",
                            "tracking",
                            "pre",
                            "post",
                            "mae_K",
                            "removed_weight_l2",
                        )
                    }
                    if label != "intact":
                        norms.append(cells[spec[1]]["removed_weight_l2"])
                if prefix == "matched":
                    np.testing.assert_allclose(*norms, rtol=1e-6)
    return summaries


def routing():
    result = {}
    for interface in ("restricted", "full"):
        arms = []
        cases = None
        for layout in ("original", "swap"):
            runs = []
            for seed in (7, 17, 27, 37, 47, 57, 67, 77):
                rec = json.loads((ROOT / f"fit-{interface}-{seed}-{layout}.json").read_text())
                assert (rec["seed"], rec["interface"], rec["layout"], rec["updates"]) == (
                    seed,
                    interface,
                    layout,
                    2048,
                )
                rows = rec["cases"]
                ids = [(r["seed"], r["field"], r["initial_error_K"] > 0) for r in rows]
                cases = cases or ids
                assert cases == ids and len(set(ids)) == 64
                runs.append(
                    [
                        [r["complete_success"], r["full_tracking"], r["center_error_mae_K"]]
                        for r in rows
                    ]
                )
            arms.append(np.array(runs))
        diff = arms[0] - arms[1]
        strata = [
            [i for i, (_, f, s) in enumerate(cases) if f == field and s == side]
            for field in ("linear", "radial")
            for side in (False, True)
        ]
        assert all(len(ix) == 16 for ix in strata)
        rng = np.random.default_rng(191119)
        boot = []
        for _ in range(10000):
            models = rng.integers(0, 8, 8)
            sample = np.concatenate([rng.choice(ix, 16) for ix in strata])
            boot.append(diff[models][:, sample].mean(axis=(0, 1)))
        result[interface] = dict(
            metric_order=["success", "tracking", "mae_K"],
            original=arms[0].mean(axis=(0, 1)).tolist(),
            swap=arms[1].mean(axis=(0, 1)).tolist(),
            paired_by_seed=diff.mean(axis=1).tolist(),
            paired_mean=diff.mean(axis=(0, 1)).tolist(),
            crossed_stratified_percentile_95=np.quantile(boot, [0.025, 0.975], axis=0).tolist(),
            paired_by_stratum=[diff[:, ix].mean(axis=(0, 1)).tolist() for ix in strata],
        )
    return result


if __name__ == "__main__":
    out = {"lesions": lesions(), "routing": routing()}
    (ROOT / "summary.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
