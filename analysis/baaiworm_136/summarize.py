"""Validate and score native trials against a field-matched blind swimmer."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thermotaxis.baaiworm_temperature import BAAIWormTemperatureField


def xyz(frames, label):
    return np.column_stack([frames[label + "_" + axis] for axis in "xyz"])


def read_trial(folder):
    spec = json.loads((folder / "design.json").read_text())
    done = json.loads((folder / "done.json").read_text())
    frames = np.genfromtxt(folder / "frames.csv", names=True, delimiter=",")
    with np.load(folder / "network.npz") as data:
        motor, muscle = data["motor_mV"], data["muscle"]
    count = spec["last_frame"] + 1
    assert done["cells"] == 136 and len(done["input_cells"]) == 15
    assert done["samples"] == count and done["substeps"] == 60
    assert motor.shape == (count, len(done["motor_cells"])) == (count, 80)
    assert muscle.shape == (count, 96)
    assert np.isfinite(motor).all() and np.isfinite(muscle).all()
    assert all(np.isfinite(frames[key]).all() for key in frames.dtype.names)
    np.testing.assert_array_equal(frames["frame"], np.arange(count))
    np.testing.assert_allclose(frames["body_s"], np.arange(count) / 30)
    np.testing.assert_allclose(frames["neural_s"], (np.arange(count) + 1) / 10)
    np.testing.assert_allclose(frames["clamp_nA"], frames["base_nA"] + frames["thermal_nA"], atol=1e-8, rtol=1e-6)
    field = BAAIWormTemperatureField(spec["origin"], spec["length"], spec["normal"])
    for point in ["head", "center"]:
        np.testing.assert_allclose(field.temperature_c(xyz(frames, point)), frames[point + "_C"], rtol=0, atol=1e-10)
    expected = frames["head_C"] if spec["feedback"] else field.temperature_c(spec["origin"])
    np.testing.assert_allclose(expected, frames["sensed_C"], atol=1e-12, rtol=0)
    value = -abs(frames["sensed_C"] - spec["target_C"])
    ref = value[0]
    for i, sample in enumerate(value):
        if i <= spec["onset_frame"]:
            ref = sample
        thermal = spec["gain_nA"] * np.tanh((sample - ref) / spec["scale_C"])
        np.testing.assert_allclose(thermal, frames["thermal_nA"][i], rtol=0, atol=1e-12)
        ref += -np.expm1(-1 / (30 * spec["adaptation_s"])) * (sample - ref)
    return spec, field, frames, motor, muscle


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def summarize(root):
    groups = ["normal", "plane", "plane-inverted"]
    trials = {name: read_trial(root / name) for name in
              ["blind"] + [g + "-" + s for g in groups for s in ("pos", "neg")]}
    _, _, blind, blind_motor, blind_muscle = trials["blind"]
    history = np.genfromtxt("docs/reports/assets/pathway-specificity/body/sham.csv", names=True, delimiter=",")
    for point in ["head", "center"]:
        np.testing.assert_allclose(xyz(blind, point)[history["frame"].astype(int)], xyz(history, point), atol=1e-6, rtol=0)
    rows = {}
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9), sharex="col", constrained_layout=True)
    for r, group in enumerate(groups):
        for suffix, color in [("pos", "#a63e16"), ("neg", "#155a9c")]:
            name = group + "-" + suffix
            spec, field, f, motor, muscle = trials[name]
            other = "neg" if suffix == "pos" else "pos"
            opposite_spec, _, opposite, _, _ = trials[group + "-" + other]
            np.testing.assert_allclose(spec["normal"], -np.asarray(opposite_spec["normal"]))
            assert spec["gain_nA"] == opposite_spec["gain_nA"]
            t = f["body_s"]
            onset = spec["onset_frame"]
            pre, active, tail = t <= t[onset], t > t[onset], t >= 8
            for point in ["head", "center"]:
                np.testing.assert_allclose(xyz(f, point)[pre], xyz(blind, point)[pre], atol=1e-7, rtol=0)
            np.testing.assert_array_equal(motor[pre], blind_motor[pre])
            np.testing.assert_array_equal(muscle[pre], blind_muscle[pre])
            np.testing.assert_array_equal(f["base_nA"], blind["base_nA"])
            center, baseline = xyz(f, "center"), xyz(blind, "center")
            error = abs(f["center_C"] - spec["target_C"])
            blind_error = abs(field.temperature_c(baseline) - spec["target_C"])
            opposite_error = abs(field.temperature_c(xyz(opposite, "center")) - spec["target_C"])
            axial = (center - center[onset]) @ spec["normal"] / spec["length"]
            paired = (center - baseline) @ spec["normal"] / spec["length"]
            rows[name] = dict(center_error_onset_C=float(error[onset]), center_error_final_C=float(error[-1]),
                              center_error_drop_C=float(error[onset] - error[-1]),
                              final_benefit_C=float(blind_error[-1] - error[-1]),
                              tail_benefit_C=float((blind_error - error)[tail].mean()),
                              tail_vs_opposite_track_C=float((opposite_error - error)[tail].mean()),
                              occupancy_active=float(np.mean(error[active] <= .1)),
                              blind_occupancy_active=float(np.mean(blind_error[active] <= .1)),
                              final_axial_BL=float(axial[-1]), final_paired_axial_BL=float(paired[-1]),
                              tail_paired_axial_BL=float(paired[tail].mean()),
                              head_C_range=[float(f["head_C"].min()), float(f["head_C"].max())],
                              thermal_rms_nA=rms(f["thermal_nA"][active]),
                              motor_difference_rms_mV=rms((motor - blind_motor)[active]),
                              muscle_difference_rms=rms((muscle - blind_muscle)[active]))
            label = "+" if suffix == "pos" else "−"
            axes[r, 0].plot(t, f["head_C"], color=color, label=label + " field")
            axes[r, 0].plot(t, field.temperature_c(xyz(blind, "head")), "--", lw=1, color=color)
            axes[r, 1].plot(t, blind_error - error, color=color)
            axes[r, 2].plot(t, paired, color=color)
        for ax in axes[r]:
            ax.axvline(2, lw=1, color="#555555", ls=":")
            ax.grid(alpha=.18)
        axes[r, 0].set_ylabel(["Plane normal", "In swim plane", "In plane, inverted gain"][r] + "\nHead (°C)")
        for ax in axes[r, 1:]:
            ax.axhline(0, color="#555555", lw=.7)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Head temperature\nDashed = blind track in same field")
    axes[0, 1].set_title("Center temperature benefit (°C)\nPositive = closer to 20 °C than blind")
    axes[0, 2].set_title("Center displacement vs blind (body lengths)\nPositive = toward warm side")
    for ax in axes[-1]:
        ax.set_xlabel("Body time (s)")
    fig.savefig(root / "temperature-and-response.png", dpi=155)
    plt.close(fig)

    fig = plt.figure(figsize=(13.5, 4.7))
    fig.subplots_adjust(left=.015, right=.97, bottom=.09, top=.91, wspace=.12)
    origin = xyz(blind, "center")[0]
    paths = {name: (xyz(run[2], "center") - origin) / run[0]["length"] for name, run in trials.items()}
    points = np.concatenate(list(paths.values()))
    mid = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = np.ptp(points, axis=0).max() * .55
    for i, group in enumerate(groups, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        for name, color, label in [(group+"-pos", "#a63e16", "+ field"), (group+"-neg", "#155a9c", "− field"),
                                   ("blind", "#444444", "Blind")]:
            p = paths[name]
            ax.plot(*p.T, color=color, label=label, ls="--" if name == "blind" else "-", lw=1.25)
            ax.scatter(*p[-1], color=color, s=12)
        ax.set(xlabel="X (BL)", ylabel="Y (BL)", zlabel="Z (BL)", title=["Plane normal", "In swim plane", "In plane, inverted gain"][i-1])
        for setter, middle in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
            setter(middle - radius, middle + radius)
        ax.locator_params(nbins=5)
        ax.set_box_aspect((1, 1, 1), zoom=.88)
    ax.legend(fontsize=8)
    fig.savefig(root / "center-trajectories.png", dpi=155)
    plt.close(fig)
    (root / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
