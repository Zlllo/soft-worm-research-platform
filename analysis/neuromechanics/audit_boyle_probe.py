"""Summarize actual WormSim runs and independently check body feedback geometry."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from thermotaxis.neuromechanics.boyle_output import load_body, side_lengths, body_metrics, curvature


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    a = p.parse_args()
    base = a.run_dir.resolve()
    summaries, arrays = {}, {}
    for folder in sorted((base / "outputs").iterdir()):
        process_path = folder / "process.json"
        if not process_path.exists():
            continue
        process = json.loads(process_path.read_text())
        if process.get("returncode") != 0:
            summaries[folder.name] = dict(status="failed", process=process)
            continue
        design = json.loads((folder / "design.json").read_text())
        t, pose = load_body(folder / "simdata.csv")
        metric = body_metrics(t, pose)
        axis = pose[:, 0, :2]-pose[:, -1, :2]
        heading = np.unwrap(np.arctan2(axis[:, 1], axis[:, 0]))
        metric["body_axis_change_deg"] = float(np.degrees(heading[-1]-heading[0]))
        diagnostic = dict()
        if (folder / "neuromuscle.csv").exists():
            state_raw = np.loadtxt(folder / "neuromuscle.csv", delimiter=",")
            if state_raw.shape[1] not in (385, 386) or len(state_raw) != len(t):
                raise ValueError("invalid neuromuscle records: " + str(folder))
            np.testing.assert_allclose(state_raw[:, 0], t, atol=1e-12, rtol=0)
            payload = state_raw[:, 1:] if state_raw.shape[1] == 385 else state_raw[:, 1:-1]
            state = payload.reshape(-1,48,2,4)
            if not np.isfinite(state).all():
                raise ValueError("nonfinite model states")
            # Initial save precedes the first residual evaluation.
            error = abs(side_lengths(pose)[1:]-state[1:,:,:,3])
            diagnostic["max_stretch_length_error_m"] = float(error.max())
            diagnostic["rms_stretch_length_error_m"] = float(np.sqrt(np.mean(error**2)))
            diagnostic["muscle_state_min_max"] = [float(state[:,:,:,1].min()),float(state[:,:,:,1].max())]
            if state_raw.shape[1] == 386:
                offset = state_raw[1:,-1]-t[1:]
                diagnostic["residual_time_minus_control_time_min_max_s"] = [float(offset.min()),float(offset.max())]
                diagnostic["saved_frames_with_future_residual_cache"] = int(np.count_nonzero(offset>1e-9))
            if design.get("feedback_geometry") == "accepted control-time state" and diagnostic["max_stretch_length_error_m"] > 1e-15:
                raise ValueError("accepted-state feedback disagrees with independent geometry")
        summaries[folder.name] = dict(status="completed", design=design, process=process,
             metrics=metric, feedback_geometry_check=diagnostic,
             output_sha256=hashlib.sha256((folder / "simdata.csv").read_bytes()).hexdigest())
        arrays[folder.name] = (t,pose)
    original = arrays["original-agar-10s"][1]
    recorded = arrays["audited-agar-10s"][1]
    recording_check = dict(max_position_difference_m=float(abs(original[:,:,:2]-recorded[:,:,:2]).max()),
                           max_rod_angle_difference_rad=float(abs(original[:,:,2]-recorded[:,:,2]).max()),
                           scope="original CSV has six significant digits; instrumentation preserves the model")
    if recording_check["max_position_difference_m"] > 6e-9 or recording_check["max_rod_angle_difference_rad"] > 6e-6:
        raise ValueError("instrumented run differs beyond original CSV precision")
    reference = summaries["audited-agar-10s"]["metrics"]["mean_center_speed_m_s"]
    effects = dict(no_feedback_speed_fraction=summaries["agar-no-feedback-10s"]["metrics"]["mean_center_speed_m_s"]/reference,
                   accepted_feedback_speed_fraction=summaries["agar-accepted-feedback-10s"]["metrics"]["mean_center_speed_m_s"]/reference)
    dt_metrics=[]
    for key in ('audited-agar-10s','agar-half-dt-10s'):
        time,pose=arrays[key];mask=time<=9.96+1e-9
        dt_metrics.append(body_metrics(time[mask],pose[mask]))
    dt_check=dict(common_saved_interval_s=[0.,9.96], warmup_s=3.,
                  reference_speed_m_s=dt_metrics[0]['mean_center_speed_m_s'],
                  half_dt_speed_m_s=dt_metrics[1]['mean_center_speed_m_s'],
                  relative_speed_difference=abs(dt_metrics[1]['mean_center_speed_m_s']/dt_metrics[0]['mean_center_speed_m_s']-1),
                  scope='one timestep-halving diagnostic; not full convergence across media or interventions')
    summary = dict(source_commit="1f825daff066d56e5c26384d71648914ca169d6b", source="Boyle, Berri & Cohen 2012 WormSim; OpenSourceBrain archive",
                   scope="forward neuromechanical model evaluation; no thermotaxis or full connectome",
                   recording_check=recording_check, effects=effects, timestep_check=dt_check, runs=summaries)
    (base / "summary.json").write_text(json.dumps(summary,indent=2))

    os.environ.setdefault("MPLCONFIGDIR", str(base / "mpl-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(base / "cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,key,title in zip(axes[0], ("audited-agar-10s","agar-no-feedback-10s"),
                            ("Native stretch feedback", "Stretch gain = 0")):
        t,pose=arrays[key]
        im=ax.imshow(curvature(pose).T/1000,origin="lower",aspect="auto",extent=[t[0],t[-1],0,1],
                     cmap="RdBu_r",vmin=-12,vmax=12)
        ax.set(xlabel="Time (s)",ylabel="Head → tail position / body length",title=title)
    fig.colorbar(im,ax=list(axes[0]),label="Centerline curvature (mm⁻¹)",shrink=.8)
    for name,color in [("audited-agar-10s","#31608a"),("agar-no-feedback-10s","#b0743c"),
                        ("agar-no-drive-10s","#777777")]:
        t,pose=arrays[name];c=pose[:,:,:2].mean(1)
        speed=np.linalg.norm(np.diff(c,axis=0),axis=1)/np.diff(t)
        axes[1,0].plot(t[1:],speed*1000,label=name.replace("-10s", ""),lw=.8,color=color)
    axes[1,0].set(xlabel="Time (s)",ylabel="Center speed (mm/s)",title="Drive and body feedback matter")
    axes[1,0].legend(frameon=False,fontsize=8)
    for name,color in [("agar-accepted-feedback-10s","#31608a"),
                       ("agar-head-plus-10s","#b0743c"),("agar-head-minus-10s","#56805a")]:
        if name not in arrays:continue
        _,pose=arrays[name];c=pose[:,:,:2].mean(1);c-=c[0]
        axes[1,1].plot(c[:,0]*1000,c[:,1]*1000,label=name.replace("agar-","").replace("-10s",""),color=color)
    axes[1,1].set(xlabel="Center x displacement (mm)",ylabel="Center y displacement (mm)",
                   title="Engineering head-bias probes")
    axes[1,1].axis("equal");axes[1,1].legend(frameon=False,fontsize=8)
    for ax in axes[1]:ax.spines[["top","right"]].set_visible(False)
    fig.savefig(base / "fastbody-probes.png",dpi=180)
    fig.savefig(base / "fastbody-probes.svg")
    plt.close(fig)
    print(json.dumps({"completed":sum(x['status']=='completed' for x in summaries.values()),
                      "effects":effects,"summary":str(base/'summary.json')}))


if __name__ == "__main__":
    main()
