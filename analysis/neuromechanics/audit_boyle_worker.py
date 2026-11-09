"""Run persistent-body and actual-head thermal-interface checks, without training."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thermotaxis.neuromechanics.boyle_body import BoyleBody
from thermotaxis.neuromechanics.boyle_output import body_metrics, load_body, side_lengths
from thermotaxis.neuromechanics.thermal_body import ThermalBody


def simulate(binary, out, name, *, causal=True, duration_s=10., schedule=None):
    states, observations, controls = [], [], []
    start = time.monotonic()
    with BoyleBody(binary, causal=causal) as body:
        env = ThermalBody(body, gradient_K_m=(40., 20.))
        observations.append(asdict(env.reset()))
        states.append(body.state)
        for i in range(round(duration_s/.1)):
            drive, bias = schedule(i*.1) if schedule else (.675, 0.)
            observations.append(asdict(env.step(drive=drive, head_bias=bias, sensor_samples=1)[0]))
            states.append(body.state)
            controls.append((drive, bias))
    wall = time.monotonic()-start
    t = np.array([s.time_s for s in states])
    pose = np.stack([s.pose for s in states])
    side = np.stack([s.side_length for s in states])
    internal = np.array([s.internal_time_s for s in states])
    temp = np.array([s["temperature_K"] for s in observations])
    expected = 293.15 + pose[:, 0, :2] @ np.array([40., 20.])
    np.savez_compressed(out / (name+".npz"), time_s=t, pose=pose,
        velocity=np.stack([s.velocity for s in states]), side_length=side,
        neural=np.stack([s.neural for s in states]), muscle=np.stack([s.muscle for s in states]),
        stretch=np.stack([s.stretch for s in states]), internal_time_s=internal,
        temperature_K=temp, controls=np.array(controls))
    metrics = body_metrics(t, pose)
    metrics.update(wall_s=wall, max_solver_ahead_s=float(np.max(internal-t)),
        max_side_geometry_error_m=float(np.max(np.abs(side-side_lengths(pose)))),
        max_head_temperature_error_K=float(np.max(np.abs(temp-expected))),
        temperature_range_K=float(np.ptp(temp)))
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for name, options in [("causal-baseline", {}), ("lookahead-compatibility", dict(causal=False)),
        ("positive-bias", dict(schedule=lambda t:(.675,.05))),
        ("negative-bias", dict(schedule=lambda t:(.675,-.05))),
        ("stop-resume", dict(duration_s=15., schedule=lambda t:(0. if 5<=t<10 else .675,0.))),
        ("changing-steering", dict(duration_s=15., schedule=lambda t:(.675, .05 if 5<=t<10 else (-.05 if t>=10 else 0.))))]:
        summary[name] = simulate(a.binary, a.output, name, **options)
        print(name, json.dumps(summary[name]), flush=True)
    old_t, old_pose = load_body(a.previous / "simdata.csv")
    compat = np.load(a.output / "lookahead-compatibility.npz")
    # 0.1 s vs 0.04 s output share 0.2 s timestamps; no interpolation.
    pairs = [(i,int(np.argmin(abs(old_t-t)))) for i,t in enumerate(compat["time_s"])
             if np.min(abs(old_t-t))<1e-8]
    delta = np.stack([compat["pose"][i]-old_pose[j] for i,j in pairs])
    summary["previous_accepted_state_comparison"] = dict(shared_frames=len(pairs),
        max_position_difference_m=float(np.max(np.abs(delta[:,:,:2]))),
        max_rod_angle_difference_rad=float(np.max(np.abs(delta[:,:,2]))),
        note="integer tick clock versus accumulated source clock; same accepted-state feedback; old solver lookahead retained only in compatibility mode")
    base = np.load(a.output / "causal-baseline.npz")
    summary["causal_vs_lookahead"] = dict(
        max_position_difference_m=float(np.max(abs(base["pose"][:,:,:2]-compat["pose"][:,:,:2]))),
        speed_change_percent=100*(summary["causal-baseline"]["mean_center_speed_m_s"]/summary["lookahead-compatibility"]["mean_center_speed_m_s"]-1))
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    for name in ["causal-baseline", "positive-bias", "negative-bias"]:
        d = np.load(a.output/(name+".npz"))
        center = d["pose"][:,:,:2].mean(axis=1)*1000
        axes[0,0].plot(center[:,0],center[:,1],label=name)
        axes[0,1].plot(d["time_s"],d["temperature_K"]-293.15,label=name)
    axes[0,0].set(xlabel="Center x (mm)",ylabel="Center y (mm)",title="Effective head bias changes body path", aspect="equal")
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(xlabel="Time (s)",ylabel="Head temperature - reference (K)",title="Temperature sampled at actual head")
    for name in ["stop-resume", "changing-steering"]:
        d = np.load(a.output/(name+".npz"))
        center = d["pose"][:,:,:2].mean(axis=1)
        speed = np.linalg.norm(np.diff(center,axis=0),axis=1)/np.diff(d["time_s"])*1000
        axes[1,0].plot(d["time_s"][1:],speed,label=name)
    axes[1,0].axvspan(5,10,color="gray",alpha=.15)
    axes[1,0].set(xlabel="Time (s)",ylabel="Center path speed (mm/s)",title="Time-varying controls; no heading assignment")
    axes[1,0].legend(fontsize=8)
    axes[1,1].plot(base["time_s"],(base["internal_time_s"]-base["time_s"])*1000,label="causal")
    axes[1,1].plot(compat["time_s"],(compat["internal_time_s"]-compat["time_s"])*1000,label="lookahead compatibility")
    axes[1,1].set(xlabel="Time (s)",ylabel="IDA internal time - control time (ms)",title="Solver timing is part of the interface")
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.savefig(a.output / "interface-checks.png",dpi=160)
    fig.savefig(a.output / "interface-checks.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
