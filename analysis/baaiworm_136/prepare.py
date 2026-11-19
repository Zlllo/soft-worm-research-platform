"""Generate paired three-dimensional fields from the native swim calibration."""
import json
from pathlib import Path

import numpy as np


def main():
    calibration = Path("docs/reports/assets/pathway-specificity")
    root = Path("results/baaiworm-136")
    cal = json.loads((calibration / "baaiworm-calibration.json").read_text())
    normal = np.asarray(json.loads((calibration / "body-summary.json").read_text())["reference_normal"])
    in_plane = np.array([1., 0., 0.]) - normal * normal[0]
    in_plane /= np.linalg.norm(in_plane)
    trials = [("blind", normal, .02)]
    for label, axis, gain in [("normal", normal, .02), ("plane", in_plane, .02),
                              ("plane-inverted", in_plane, -.02)]:
        trials.extend([(label + "-pos", axis, gain), (label + "-neg", -axis, gain)])

    for name, axis, gain in trials:
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        spec = dict(origin=cal["origin_world"], length=cal["body_length_world"],
                    normal=axis.tolist(), feedback=name != "blind", target_C=20.,
                    onset_frame=60, last_frame=300, gain_nA=gain, scale_C=.02,
                    adaptation_s=.3, name=name)
        (out / "design.json").write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    main()
