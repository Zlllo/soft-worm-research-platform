"""Validate named mappings and actual upstream outputs; write a stage-0 summary."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess

import numpy as np

from thermotaxis.neuromechanics.muscle_mapping import MuscleMapping, body_wall_slots


ROOT = Path(__file__).resolve().parents[2]


def read_body(out):
    with (out / "position_buffer.txt").open() as f:
        header = np.array([float(next(f)) for _ in range(11)])
        values = np.loadtxt(f)
    elastic, liquid, boundary = map(int, header[6:9])
    total, dynamic = elastic + liquid + boundary, elastic + liquid
    if (len(values) - total) % dynamic:
        raise ValueError("unexpected truncated body position frames")
    first = values[:total]
    active = first[np.floor(first[:, 3]) != 3]
    later = values[total:].reshape(-1, dynamic, 4)
    frames = np.concatenate([active[None], later], axis=0)
    elastic_mask = np.floor(active[:, 3]) == 2
    if not np.all(np.floor(frames[:, :, 3]) == np.floor(active[:, 3])):
        raise ValueError("particle identity/type changed")
    if not np.isfinite(frames).all():
        raise ValueError("nonfinite particle data")
    body = frames[:, elastic_mask, :3]
    muscle = np.atleast_2d(np.loadtxt(out / "muscles_activity_buffer.txt"))
    if muscle.shape[1] != 96 or not np.isfinite(muscle).all():
        raise ValueError("invalid muscle signals")
    log = (out / "run.log").read_text()
    steps = re.findall(r"\[\[ Step (\d+) .*?t in sim: ([\deE.+-]+)s", log)
    if not steps or "Simulation has reached the time limit" not in log:
        raise ValueError("body simulation has no confirmed end condition")
    nsteps = int(steps[-1][0]) + 1
    times = np.arange(len(frames)) * header[9] * header[10]
    displacement = np.sqrt(np.mean(np.sum((body - body[0]) ** 2, axis=2), axis=1))
    process = json.loads((out / "process.json").read_text())
    if process["returncode"] != 0 or process["timed_out"]:
        raise ValueError("body process did not finish")
    record = dict(physical_steps=nsteps, physics_dt_s=float(header[9]),
                  integrated_s=nsteps*float(header[9]), saved_frames=len(frames),
                  last_saved_time_s=float(times[-1]), elastic_particles=elastic,
                  liquid_particles=liquid, boundary_particles=boundary,
                  wall_s=process["wall_s"], muscle_min=float(muscle.min()),
                  muscle_max=float(muscle.max()), particle_data_finite=True,
                  rms_displacement_last_saved_upstream_units=float(displacement[-1]),
                  seconds_wall_per_second_simulated=process["wall_s"]/(nsteps*header[9]),
                  coordinate_units="raw upstream units; absolute length calibration pending",
                  scope="short numerical run; no steady locomotion, sensory feedback or thermotaxis")
    return record, times, displacement, body


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    base = args.run_dir.resolve()
    outputs = base / "outputs"
    lock_path = ROOT / "configs/neuromechanics/upstream-lock.json"
    lock = json.loads(lock_path.read_text())
    provenance = {}
    for name in ("c302", "sibernetic"):
        source = base / "upstream" / name
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if commit != lock[name]["commit"]:
            raise ValueError(f"{name} does not match the source lock")
        license_file = source / lock[name]["license_file"]
        provenance[name] = dict(commit=commit, license_sha256=hashlib.sha256(license_file.read_bytes()).hexdigest())
    raw = (ROOT / "data/connectome/cook2019_herm.json").read_bytes()
    names = json.loads(raw)["nodes"]
    muscle_names = tuple(n for n in names if n in body_wall_slots())
    if len(muscle_names) != 95:
        raise ValueError("unexpected anatomical muscle coverage")
    mapping = MuscleMapping(muscle_names, body_wall_slots(tuple(lock["sibernetic_signal_order"]["quadrants"])))
    x = np.arange(95, dtype=float)
    np.testing.assert_array_equal(mapping.from_slots(mapping.to_slots(x)), x)
    (base / "muscle-mapping.json").write_text(json.dumps(mapping.as_dict(), indent=2))

    intact = np.load(outputs / "c302-c1-syns/traces.npz")
    disconnected = np.load(outputs / "c302-c1-disconnected/traces.npz")
    np.testing.assert_array_equal(intact["t"], disconnected["t"])
    pre, ca = "AS2/0/GenericNeuronCell/v", "MDL07/0/GenericMuscleCell/caConc"
    effect = dict(presynaptic_voltage_max_abs_difference_V=float(np.abs(intact[pre]-disconnected[pre]).max()),
                  intact_muscle_calcium_peak=float(intact[ca].max()),
                  disconnected_muscle_calcium_peak=float(disconnected[ca].max()),
                  calcium_units="mol/m3 in exported model traces; not experimental calibration",
                  scope="isolating one simulated AS2->MDL07 projection")
    if effect["presynaptic_voltage_max_abs_difference_V"] > 1e-10:
        raise ValueError("presynaptic input differs between projection conditions")
    if effect["intact_muscle_calcium_peak"] <= effect["disconnected_muscle_calcium_peak"]:
        raise ValueError("expected muscle response difference was not observed")
    bodies = {}
    body_samples = {}
    plot_body = None
    for path in sorted(outputs.glob("sibernetic-*")):
        if not (path / "process.json").exists():
            continue
        process = json.loads((path / "process.json").read_text())
        if process["returncode"] == 0:
            result, times, displacement, body = read_body(path)
            bodies[path.name] = result
            body_samples[path.name] = body
            if "20ms" in path.name and "optimized" not in path.name:
                plot_body = (times, displacement, body)
    body_comparison = None
    if all(k in body_samples for k in ("sibernetic-body-20ms", "sibernetic-body-20ms-optimized")):
        body_comparison = dict(max_saved_elastic_position_abs_difference=float(np.max(np.abs(
            body_samples["sibernetic-body-20ms"]-body_samples["sibernetic-body-20ms-optimized"]))),
            scope="saved decimal precision over 20ms only; not full numerical equivalence")
    neuron_comparison = None
    nrn_file = outputs / "c302-c1-neuron-wrapper/traces.npz"
    if nrn_file.exists():
        nrn = np.load(nrn_file)
        np.testing.assert_array_equal(intact["t"], nrn["t"])
        neuron_comparison = dict(samples=len(nrn["t"]), simulated_s=float(nrn["t"][-1]),
            muscle_calcium_peak=float(nrn[ca].max()),
            relative_calcium_peak_difference=float(abs(nrn[ca].max()-intact[ca].max())/intact[ca].max()),
            scope="same model in jNeuroML and NEURON; peak comparison is not a convergence proof")
    # Java diagnostics can dump all process variables. Retain error context,
    # but remove unrelated environment values from every archived log.
    redacted = 0
    for log in (base / "logs").glob("*.log"):
        text = log.read_text(errors="replace")
        text, n = re.subn(r"(?m)^(\s*(?:jNeuroML >>\s*)?[A-Z][A-Z0-9_]*\s+=).*$", r"\1 [redacted]", text)
        if n:
            log.write_text(text)
            redacted += n
    summary = dict(status="source probes executed; closed-loop coupling pending",
                   platform=dict(system=platform.system(), machine=platform.machine(), python=platform.python_version()),
                   sources=provenance, connectome_sha256=hashlib.sha256(raw).hexdigest(),
                   anatomical_muscles=len(muscle_names), simulator_slots=len(mapping.slot_names),
                   inactive_slots=list(mapping.inactive_names), nmj_probe=effect, body_probes=bodies,
                   body_optimization_comparison=body_comparison, neuron_engine_comparison=neuron_comparison,
                   packages={name: importlib.metadata.version(name) for name in
                             ("c302", "cect", "pyNeuroML", "neuron", "jdk4py", "numpy")},
                   redacted_environment_lines=redacted,
                   source_data_reader="c302 upstream default SpreadsheetDataReader; not current Cook import")
    (base / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    freeze = subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True)
    (base / "requirements-installed.txt").write_text(freeze)
    portable_freeze = re.sub(r"(?mi)^c302\s*@.*$",
        "c302 @ git+https://github.com/openworm/c302.git@"+lock["c302"]["commit"], freeze)
    (base / "requirements-lock.txt").write_text(portable_freeze)

    os.environ.setdefault("MPLCONFIGDIR", str(outputs / "c302-c1-syns/mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True)
    t = intact["t"]
    axes[0].plot(t, intact[pre]*1000, color="#2563a6", lw=.7)
    axes[0].set(ylabel="AS2 voltage (mV)", title="Upstream c302 C1: neuron-to-muscle electrical probe")
    axes[1].plot(t, intact[ca], color="#2563a6", lw=.8, label="AS2 → MDL07 intact")
    axes[1].plot(t, disconnected[ca], color="#b66b32", lw=1, label="Projection removed")
    axes[1].set(xlabel="Neural simulation time (s)", ylabel="Muscle Ca (mol/m³)")
    axes[1].legend(frameon=False)
    if plot_body:
        times, displacement, _ = plot_body
        axes[2].plot(times*1000, displacement, marker=".", color="#54805a")
    axes[2].set(xlabel="Body simulation time (ms)", ylabel="RMS particle displacement\n(upstream coordinate units)",
                title="Separate Sibernetic body run: prescribed drive, no sensory feedback")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    fig.savefig(base / "stage0-probes.png", dpi=180)
    fig.savefig(base / "stage0-probes.svg")
    plt.close(fig)
    print(json.dumps({"anatomical_muscles":len(muscle_names), "body_runs":list(bodies),
                      "summary":str(base / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
