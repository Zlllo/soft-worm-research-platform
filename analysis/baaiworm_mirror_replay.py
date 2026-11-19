"""Replay a measured native swim in two fixed, mirrored temperature fields."""

import json
from pathlib import Path

import numpy as np

from thermotaxis.baaiworm_temperature import BAAIWormTemperatureField

ROOT = Path("docs/reports/assets/pathway-specificity")


def main():
    calibration = json.loads((ROOT / "baaiworm-calibration.json").read_text())
    rows = np.loadtxt(ROOT / "baaiworm-head.csv", delimiter=",", skiprows=1)
    t, xyz = rows[:, 1], rows[:, 2:5]
    origin = np.array(calibration["origin_world"])
    length = calibration["body_length_world"]
    np.testing.assert_allclose(xyz[0], origin)
    original = BAAIWormTemperatureField(origin, length, calibration["normal_world"])
    np.testing.assert_allclose(original.temperature_c(xyz), rows[:, 5], atol=1e-12, rtol=0)
    result = []
    for end in (1.0, 2.0, 3.0):
        # Constrain the plane to contain the initial head. Use only past frames.
        past = xyz[t <= end] - origin
        _, singular, axes = np.linalg.svd(past, full_matrices=False)
        normal = axes[-1] * np.sign(axes[-1, 1])
        up = BAAIWormTemperatureField(origin, length, normal)
        down = BAAIWormTemperatureField(origin, length, -normal)
        test = xyz[t > end]
        height = (test - origin) @ normal
        flat = test - height[:, None] * normal
        np.testing.assert_allclose(
            up.temperature_c(flat), down.temperature_c(flat), atol=1e-12, rtol=0
        )
        delta = 1000 * (up.temperature_c(test) - down.temperature_c(test))
        result.append(
            dict(
                calibration_s=end,
                holdout_frames=len(test),
                normal=normal.tolist(),
                normal_angle_from_y_deg=float(np.degrees(np.arccos(normal[1]))),
                singular_values=singular.tolist(),
                rms_height_in_body_lengths=float(np.sqrt(np.mean(height**2)) / length),
                mirror_difference_rms_mK=float(np.sqrt(np.mean(delta**2))),
            )
        )
    (ROOT / "swim-mirror.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
