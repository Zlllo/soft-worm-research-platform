"""Check native commands, paired starts and report world-space body deflection."""

import json
from pathlib import Path

import numpy as np

ROOT = Path("docs/reports/assets/pathway-specificity")
CONDITIONS = ("sham", "right", "left", "dorsal", "ventral", "sham2")


def main():
    runs, hashes = {}, []
    for label in CONDITIONS:
        raw = np.loadtxt(ROOT / "body" / (label + ".csv"), delimiter=",", skiprows=1)
        np.testing.assert_array_equal(raw[:, 0], np.arange(0, 151, 3))
        np.testing.assert_allclose(raw[:, 1], raw[:, 0] / 30)
        assert np.isfinite(raw).all()
        runs[label] = raw
        status = json.loads((ROOT / "body" / (label + ".json")).read_text())
        assert status["command_frames"] == 150 and status["condition"] == label
        hashes.append(status["original_muscle_sha256"])
    assert len(set(hashes)) == 1
    baseline = runs["sham"]
    length = json.loads((ROOT / "baaiworm-calibration.json").read_text())["body_length_world"]
    passive = np.loadtxt(ROOT / "baaiworm-head.csv", delimiter=",", skiprows=1)
    pre = baseline[:, 1] <= 2.0
    # Same 30 Hz reference as mirror replay, determined before the pulse.
    _, _, v = np.linalg.svd(
        passive[passive[:, 1] <= 2.0, 2:5] - passive[0, 2:5], full_matrices=False
    )
    normal = v[-1] * np.sign(v[-1, 1])
    compare = {}
    for label, raw in runs.items():
        dh = (raw[:, 2:5] - baseline[:, 2:5]) / length
        dc = (raw[:, 5:8] - baseline[:, 5:8]) / length
        mean = (raw[:, 1] >= 4.0) & (raw[:, 1] <= 5.0)
        end = np.flatnonzero(raw[:, 1] == 4.0).item()
        compare[label] = dict(
            max_pre_center_difference_BL=float(np.linalg.norm(dc[pre], axis=1).max()),
            mean_center_normal_BL=float((dc[mean] @ normal).mean()),
            mean_head_normal_BL=float((dh[mean] @ normal).mean()),
            at_4s_center_normal_BL=float(dc[end] @ normal),
            at_5s_center_normal_BL=float(dc[-1] @ normal),
            mean_center_delta_xyz_BL=dc[mean].mean(axis=0).tolist(),
            absolute_center_normal_since_2s_at_5s_BL=float(
                (raw[-1, 5:8] - raw[pre][-1, 5:8]) @ normal / length
            ),
        )
    out = dict(
        reference_normal=normal.tolist(),
        body_length_world=length,
        mean_window_s=[4, 5],
        native_baseline_head_max_difference=float(
            np.abs(passive[::3, 2:5] - baseline[:, 2:5]).max()
        ),
        repeat_max_center_distance_BL=float(
            np.max(np.linalg.norm(runs["sham2"][:, 5:8] - baseline[:, 5:8], axis=1)) / length
        ),
        by_group=compare,
    )
    (ROOT / "body-summary.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
