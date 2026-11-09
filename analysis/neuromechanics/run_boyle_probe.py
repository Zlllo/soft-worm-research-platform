"""Build/run explicit variants of the published WormSim source in new folders.

Keep the upstream checkout unchanged. Record every substitution and preserve
the original source's copyright/use notice in each generated translation unit.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


COMMIT = "1f825daff066d56e5c26384d71648914ca169d6b"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("source anchor is not unique: " + old[:70])
    return text.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration-s", type=float, default=10.)
    p.add_argument("--medium", type=float, default=1.)
    p.add_argument("--feedback-gain", type=float, default=1.)
    p.add_argument("--drive", type=float, default=.675)
    p.add_argument("--dt-s", type=float, default=.001)
    p.add_argument("--timeout-s", type=float, default=120.)
    p.add_argument("--accepted-state-feedback", action="store_true")
    p.add_argument("--head-bias", type=float, default=0.)
    a = p.parse_args()
    if not (0 <= a.medium <= 1 and a.duration_s > 0 and a.dt_s > 0 and a.timeout_s > 0):
        p.error("invalid physical or execution scales")
    src, out = a.source.resolve(), a.output.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip()
    if commit != COMMIT:
        raise ValueError("source does not match pinned upstream commit")
    if out.exists():
        raise FileExistsError("use a new output folder to preserve results")
    out.mkdir(parents=True)
    original = (src / "WormSim/Model/worm.cc").read_text()
    text = original
    for macro, value in (("DURATION", a.duration_s), ("MEDIUM", a.medium), ("DELTAT", a.dt_s)):
        text, n = re.subn(r"(?m)^(#define " + macro + r"\s+)\S+", lambda m:m[1]+format(value,".12g"), text)
        if n != 1:
            raise ValueError("expected one macro: " + macro)
    # Preserve the original literal when requesting the source default.
    if a.drive != .675:
        text = replace_once(text, "const float I_on = 0.675;", f"const float I_on = {a.drive:.12g};")
    if a.feedback_gain != 1:
        text = replace_once(text, "SR_weight[i]*I_SR_D[i]", f"({a.feedback_gain:.12g})*SR_weight[i]*I_SR_D[i]")
        text = replace_once(text, "SR_weight[i]*I_SR_V[i]", f"({a.feedback_gain:.12g})*SR_weight[i]*I_SR_V[i]")
    if a.head_bias:
        anchor = 'V_neuron[i][1] = NMJ_weight[i]*State[(int)(i*N_units/NSEG)][1] - NMJ_weight[i]*State[(int)(i*N_units/NSEG)][0];'
        text = replace_once(text, anchor, anchor + f'''\n        if(i < 8) {{
            V_neuron[i][0] += ({a.head_bias:.12g});
            V_neuron[i][1] -= ({a.head_bias:.12g});
        }}''')
    text = replace_once(text, 'outfile.open("simdata.csv");', 'outfile.open("simdata.csv");\n  outfile.precision(17);')
    text = replace_once(text, 'ofstream outfile;', 'ofstream outfile;\nrealtype audit_last_residual_time = 0;')
    text = replace_once(text, '  \t// Import data from vectors',
                        '    audit_last_residual_time = tres;\n  \t// Import data from vectors')
    if a.accepted_state_feedback:
        # Refresh the exact physical configuration returned for this control time,
        # rather than using side lengths left by an arbitrary IDA residual call.
        refresh = '''
        for(int j = 0; j < NSEG; ++j) {
            for(int side = 0; side < 2; ++side) {
                realtype sign = side == 0 ? 1.0 : -1.0;
                realtype dx = yval[3*(j+1)] + sign*R[j+1]*cos(yval[3*(j+1)+2])
                            - yval[3*j] - sign*R[j]*cos(yval[3*j+2]);
                realtype dy = yval[3*(j+1)+1] + sign*R[j+1]*sin(yval[3*(j+1)+2])
                            - yval[3*j+1] - sign*R[j]*sin(yval[3*j+2]);
                L_SR[j][side] = sqrt(dx*dx + dy*dy);
            }
        }
'''
        text = replace_once(text, '    \tupdate_SR(tout);', refresh+'    \tupdate_SR(tout);')
    instrumentation = '''
        static ofstream state_out("neuromuscle.csv");
        state_out.precision(17);
        state_out << t;
        for(int i = 0; i < NSEG; ++i) {
            for(int side = 0; side < 2; ++side) {
                state_out << ", " << V_neuron[i][side]
                          << ", " << V_muscle[i][side]
                          << ", " << I_SR[i][side]
                          << ", " << L_SR[i][side];
            }
        }
        state_out << ", " << audit_last_residual_time << "\\n";
'''
    text = replace_once(text, '  \t\toutfile << t;', instrumentation+'  \t\toutfile << t;')
    generated = out / "worm-probe.cc"
    generated.write_text(text)
    (out / "source.patch").write_text(''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
        fromfile="upstream/WormSim/Model/worm.cc", tofile="worm-probe.cc")))
    prefix = src / "WormSim/Sundials"
    cmd = ["g++", "-O3", "-Wno-deprecated", "-I", str(prefix / "include"), str(generated),
           "-L", str(prefix / "lib"), "-lsundials_ida", "-lsundials_cvode", "-lsundials_nvecserial", "-lm", "-o", str(out / "program")]
    design = dict(source_commit=commit, source_sha256=hashlib.sha256((src / "WormSim/Model/worm.cc").read_bytes()).hexdigest(),
                  generated_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(), duration_s=a.duration_s,
                  medium=a.medium, proprioceptive_gain=a.feedback_gain, AVB_drive=a.drive,
                  feedback_geometry="accepted control-time state" if a.accepted_state_feedback else "upstream residual cache",
                  engineering_head_bias=a.head_bias,
                  head_bias_scope="constant opposite effective drive added to dorsal/ventral first 8 segments; not an anatomical synapse",
                  update_dt_s=a.dt_s, output_interval_s=.04, build_command=cmd,
                  neuromuscle_columns="time; for each of 48 segments, dorsal then ventral: effective neural drive, muscle state, stretch signal, side length; last IDA residual evaluation time",
                  scope="published simplified forward-locomotion system; no connectome, thermal input or anatomical muscle mapping")
    (out / "design.json").write_text(json.dumps(design, indent=2))
    t = time.monotonic()
    with (out / "build.log").open("w") as log:
        build = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    record = dict(build_returncode=build.returncode, build_wall_s=time.monotonic()-t)
    if build.returncode == 0:
        record["binary_sha256"] = hashlib.sha256((out / "program").read_bytes()).hexdigest()
        t = time.monotonic()
        with (out / "run.log").open("w") as log:
            try:
                proc = subprocess.run([str(out / "program")], cwd=out, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=a.timeout_s)
                record.update(returncode=proc.returncode, timed_out=False)
            except subprocess.TimeoutExpired:
                record.update(returncode=None, timed_out=True)
        record["wall_s"] = time.monotonic()-t
    (out / "process.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
    if build.returncode or record.get("returncode") != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
