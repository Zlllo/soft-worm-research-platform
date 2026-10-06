"""Reproducible phase-1 progress analysis; run from repository root.

PYTHONPATH=src MPLCONFIGDIR=/private/tmp/softworm-mpl python3 analysis/phase1_progress_scan.py
"""
from __future__ import annotations

import copy
import csv
import json
import math
import hashlib
from statistics import NormalDist
from pathlib import Path
import subprocess
import platform
from datetime import datetime, timezone
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermotaxis.phase1_config import Phase1ExperimentConfig
from thermotaxis.phase1_ensemble import run_phase1_paired_ensemble

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase1-progress-2026-10-04"
DESIGN = json.loads((ROOT / "configs/phase1_progress_scan.json").read_text())
REFERENCE = json.loads((ROOT / DESIGN["reference_config"]).read_text())
METRICS = ["reached", "restricted_first_entry_s", "comfort_residence_fraction", "rms_temperature_error_K", "turn_count"]


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def config(noise, slow, x, offset, dt):
    raw = copy.deepcopy(REFERENCE)
    raw["name"] = f"progress-noise{noise}-memory{slow}-x{x}-offset{offset}"
    raw["time"].update(physics_dt_s=dt, sensor_dt_s=DESIGN["sensor_dt_s"], control_dt_s=DESIGN["control_dt_s"], duration_s=float(DESIGN["duration_s"]))
    raw["sensor"]["noise_std_K"] = noise
    raw["thermotaxis_controller"].update(fast_memory_tau_s=DESIGN["scan"]["fast_memory_s"], slow_memory_tau_s=slow)
    raw["point_model"]["initial_x_m"] = x
    for key in ("sensor", "initial_state", "controller"):
        raw["seeds"][key] = DESIGN["seed_bases"][key] + offset
    return Phase1ExperimentConfig.from_dict(raw)


def values(record):
    return [float(record["entered_comfort_region"]), float(record["first_entry_time_s"]) if record["entered_comfort_region"] else float(DESIGN["duration_s"]),
            record["comfort_residence_fraction"], record["rms_temperature_error_K"], float(record["turn_count"])]


def wilson(k, n):
    z = NormalDist().inv_cdf((1 + DESIGN["statistics"]["confidence"]) / 2)
    p = k / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [center-half, center+half]


def survival(times, reached):
    # All censoring is administrative at 180 s. The empirical survival integral
    # equals mean(min(T, 180)); retain event/censor counts for independent audit.
    events = sorted(set(times[reached.astype(bool)].tolist()))
    grid = np.array(sorted(set([0.0, float(DESIGN["duration_s"])] + events)))
    surv = np.array([np.mean(~((times <= t) & reached.astype(bool))) for t in grid])
    rmst = float(np.sum(np.diff(grid) * surv[:-1]))
    return {"time_s": grid.tolist(), "survival_probability": surv.tolist(), "rmst_s": rmst,
            "events": int(reached.sum()), "censored": int(len(times)-reached.sum())}


def summarize(result, rng):
    arrays = {g: np.array([values(p[g]) for p in result["paired_replicates"]], dtype=float) for g in ("response", "no_response")}
    n = len(arrays["response"])
    delta = arrays["response"] - arrays["no_response"]
    ix = rng.integers(0, n, size=(DESIGN["statistics"]["bootstrap_resamples"], n))
    bootstrap = delta[ix].mean(axis=1)
    summary = {"replicates": n, "groups": {}, "paired_difference": {}}
    for group, a in arrays.items():
        curve = survival(a[:, 1], a[:, 0])
        assert abs(curve["rmst_s"]-a[:, 1].mean()) < 1e-9
        summary["groups"][group] = {"means": dict(zip(METRICS, a.mean(axis=0).tolist())), "reached_wilson95": wilson(int(a[:,0].sum()), n), "survival": curve}
    for j, metric in enumerate(METRICS):
        summary["paired_difference"][metric] = {"mean": float(delta[:,j].mean()), "bootstrap95": np.quantile(bootstrap[:,j], [(1-DESIGN["statistics"]["confidence"])/2, 1-(1-DESIGN["statistics"]["confidence"])/2]).tolist()}
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / "design.json", DESIGN)
    assert DESIGN["statistics"]["confidence"] == .95, "Only 95% CI output names supported"
    start = time.monotonic()
    created_at = datetime.now(timezone.utc).isoformat()
    assert DESIGN["numerical_gate"]["configuration_count"] == len(DESIGN["numerical_gate"]["noise_K"]) * len(DESIGN["numerical_gate"]["slow_memory_s"])
    assert DESIGN["numerical_gate"]["paired_replicates_per_configuration"] == 1, "Numerical gate supports one pair per configuration"
    gates = []
    for i, (noise, slow) in enumerate([(n,s) for n in DESIGN["numerical_gate"]["noise_K"] for s in DESIGN["numerical_gate"]["slow_memory_s"]]):
        outputs = [run_phase1_paired_ensemble(config(noise, slow, .003, 6000+i*100, dt), 1) for dt in [DESIGN["fallback_physics_dt_s"],DESIGN["candidate_physics_dt_s"]]]
        diffs = [abs(a-b) for group in ("response","no_response") for a,b in zip(values(outputs[0]["paired_replicates"][0][group]),values(outputs[1]["paired_replicates"][0][group]))]
        gates.append({"noise_K":noise,"slow_memory_s":slow,"max_absolute_metric_difference":max(diffs)})
        print(f"numerical gate {i+1}/4: max difference {max(diffs):.3g}", flush=True)
    passed = all(g["max_absolute_metric_difference"] < DESIGN["numerical_gate"]["absolute_metric_tolerance"] for g in gates)
    dt = DESIGN["candidate_physics_dt_s"] if passed else DESIGN["fallback_physics_dt_s"]
    numerical = {"passed":passed,"selected_physics_dt_s":dt,"checks":gates,"tolerance":DESIGN["numerical_gate"]["absolute_metric_tolerance"]}
    dump(OUT / "numerical-gate.json", numerical)
    rng = np.random.default_rng(DESIGN["statistics"]["bootstrap_seed"])
    jobs = [(f"scan-{i:02d}", n,s,DESIGN["scan"]["initial_x_m"],i*DESIGN["seed_bases"]["scan_cell_stride"],DESIGN["scan"]["paired_replicates_per_cell"]) for i,(n,s) in enumerate([(n,s) for n in DESIGN["scan"]["noise_K"] for s in DESIGN["scan"]["slow_memory_s"]])]
    jobs += [(f"confirmation-{side}",DESIGN["confirmation"]["noise_K"],DESIGN["confirmation"]["slow_memory_s"],DESIGN["confirmation"][f"{side}_initial_x_m"],DESIGN["seed_bases"][f"confirmation_{side}_offset"],DESIGN["confirmation"]["paired_replicates_per_side"]) for side in ("cold","hot")]
    all_summaries = {}
    for index,(name,noise,slow,x,offset,count) in enumerate(jobs):
        cfg = config(noise,slow,x,offset,dt)
        result = run_phase1_paired_ensemble(cfg,count)
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        dump(folder / "config.resolved.json", cfg.to_dict())
        dump(folder / "paired-records.json", result)
        with (folder / "paired-records.csv").open("w", newline="") as handle:
            writer=csv.writer(handle)
            writer.writerow(["replicate","group","sensor_seed","initial_state_seed","controller_seed"]+METRICS)
            for pair in result["paired_replicates"]:
                for group in ("response","no_response"):
                    writer.writerow([pair["replicate"],group,pair["seeds"]["sensor"],pair["seeds"]["initial_state"],pair["seeds"]["controller"]]+values(pair[group]))
        summary=summarize(result,rng)
        summary.update(noise_K=noise,slow_memory_s=slow,initial_x_m=x)
        dump(folder / "statistics.json",summary)
        all_summaries[name]=summary
        print(f"completed {index+1}/{len(jobs)} {name}, {count} pairs; elapsed {time.monotonic()-start:.1f}s",flush=True)
    dump(OUT / "summary.json", all_summaries)
    manifest={"code_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"analysis_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"core_sources_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "src/thermotaxis").glob("*.py"))},"git_dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()),"design":DESIGN,"numerical_acceleration":numerical,"elapsed_s":time.monotonic()-start,"created_at_utc":created_at,"python_version":platform.python_version(),"numpy_version":np.__version__,"classification":DESIGN["classification"],"interpretation_limits":["Exploratory scan: 32 pairs per cell; no multiple-comparison correction; no optimum selected.","Confirmation uses fixed n=128 per side and fresh seed blocks, not a precision-based stopping rule.","Similar cold/hot means do not establish equivalence."],"command":"PYTHONPATH=src MPLCONFIGDIR=/private/tmp/softworm-mpl python3 analysis/phase1_progress_scan.py"}
    dump(OUT / "manifest.json",manifest)
    plot(all_summaries)


def plot(summaries):
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,metric,title in zip(axes,["comfort_residence_fraction","restricted_first_entry_s"],["Residence fraction difference","Restricted first-entry difference (s)"]):
        noise_values=DESIGN["scan"]["noise_K"]
        memory_values=DESIGN["scan"]["slow_memory_s"]
        matrix=np.array([[summaries[f"scan-{i*len(memory_values)+j:02d}"]["paired_difference"][metric]["mean"] for j in range(len(memory_values))] for i in range(len(noise_values))])
        im=ax.imshow(matrix,origin="lower",aspect="auto",cmap="coolwarm",vmin=-np.max(np.abs(matrix)),vmax=np.max(np.abs(matrix)))
        ax.set(xticks=range(len(memory_values)),xticklabels=memory_values,yticks=range(len(noise_values)),yticklabels=noise_values,xlabel="Slow memory (s)",ylabel="Sensor noise (K)",title=title)
        for i in range(len(noise_values)):
            for j in range(len(memory_values)): ax.text(j,i,f"{matrix[i,j]:.3f}",ha="center",va="center")
        fig.colorbar(im,ax=ax)
    for ext in ("png","svg"):fig.savefig(OUT / f"exploratory-scan.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,side in zip(axes,["cold","hot"]):
        s=summaries[f"confirmation-{side}"]
        for group,label in [("response","Response"),("no_response","Zero response")]:
            curve=s["groups"][group]["survival"]
            ax.step(curve["time_s"],curve["survival_probability"],where="post",label=label,color="#0072B2" if group=="response" else "#D55E00",linestyle="-" if group=="response" else "--")
        ax.set(xlabel="Time (s)",ylabel="Probability of not yet reaching comfort band",title=f"{side.title()} start, {s['replicates']} paired replicates",ylim=(0,1.03),xlim=(0,DESIGN["duration_s"]))
        ax.legend()
    for ext in ("png","svg"):fig.savefig(OUT / f"confirmation-survival.{ext}",dpi=180)
    plt.close(fig)


if __name__ == "__main__":main()
