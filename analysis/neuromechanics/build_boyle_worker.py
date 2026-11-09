"""Generate a persistent, clocked worker from the pinned published source.

The generated translation unit retains the upstream use/copyright notice.
Only the worker's stdout is a protocol; model diagnostics go to stderr.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

from run_boyle_probe import COMMIT, replace_once


REFRESH = r'''
void adapter_refresh(realtype *yval) {
    for(int j=0; j<NSEG; ++j) for(int side=0; side<2; ++side) {
        realtype sign = side == 0 ? 1.0 : -1.0;
        realtype dx = yval[3*(j+1)] + sign*R[j+1]*cos(yval[3*(j+1)+2])
                    - yval[3*j] - sign*R[j]*cos(yval[3*j+2]);
        realtype dy = yval[3*(j+1)+1] + sign*R[j+1]*sin(yval[3*(j+1)+2])
                    - yval[3*j+1] - sign*R[j]*sin(yval[3*j+2]);
        L_SR[j][side] = sqrt(dx*dx+dy*dy);
    }
}
void adapter_emit(void *mem, long steps, realtype *yval, realtype *ypval) {
    realtype internal=0;
    IDAGetCurrentTime(mem, &internal);
    cout << setprecision(17) << "OK " << steps*DELTAT << " " << steps << " " << internal;
    for(int j=0; j<NEQ; ++j) cout << " " << yval[j];
    for(int j=0; j<NEQ; ++j) cout << " " << ypval[j];
    realtype (*arrays[4])[2] = {V_neuron, V_muscle, I_SR, L_SR};
    for(int a=0; a<4; ++a) for(int j=0; j<NSEG; ++j)
        for(int side=0; side<2; ++side) cout << " " << arrays[a][j][side];
    cout << endl;
}
'''

LOOP = r'''
  // A new process resets upstream function-local neural statics as well.
  const bool causal = argc < 5 || atoi(argv[4]) != 0;
  long steps=0;
  if(causal && IDASetMaxStep(mem, DELTAT) < 0) return 2;
  adapter_refresh(yval);
  adapter_emit(mem, steps, yval, ypval);
  string line;
  while(getline(cin,line)) {
    istringstream input(line);
    string op, extra;
    input >> op;
    if(op == "QUIT") break;
    if(op == "GET" && !(input >> extra)) {
        adapter_emit(mem, steps, yval, ypval);
        continue;
    }
    long count=0;
    double drive=0, bias=0;
    if(op != "STEP" || !(input >> count >> drive >> bias) || (input >> extra)
       || count<1 || count>1000000 || !isfinite(drive) || !isfinite(bias)
       || drive<0 || drive>1 || fabs(bias)>.1) {
        cout << "ERR invalid command" << endl;
        continue;
    }
    adapter_drive=drive;
    adapter_head_bias=bias;
    for(long j=0; j<count; ++j) {
        tout=(steps+1)*DELTAT;
        if(causal && IDASetStopTime(mem,tout)<0) return 3;
        // Sundials 2.3 requires the explicit TSTOP task, unlike newer APIs.
        retval=IDASolve(mem,tout,&tret,yy,yp,causal ? IDA_NORMAL_TSTOP : IDA_NORMAL);
        if(retval<0 || fabs(tret-tout)>1e-10) return 4;
        adapter_refresh(yval);
        update_SR(tout);
        update_neurons(tout);
        update_muscles(tout);
        ++steps;
    }
    adapter_emit(mem,steps,yval,ypval);
  }
  IDAFree(&mem);
  N_VDestroy_Serial(yy);
  N_VDestroy_Serial(yp);
  return 0;
}

'''


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != COMMIT:
        raise ValueError("source commit differs from the lock")
    original_path = source / "WormSim/Model/worm.cc"
    if subprocess.check_output(["git", "diff", "HEAD", "--", "WormSim/Model/worm.cc"], cwd=source):
        raise ValueError("upstream source has tracked modifications")
    output.mkdir(parents=True, exist_ok=False)
    original = original_path.read_text()
    text = replace_once(original, "#include <cstdlib>", "#include <cstdlib>\n#include <sstream>\n#include <iomanip>\n#include <string>")
    text = replace_once(text, "ofstream outfile;", "ofstream outfile;\nfloat adapter_drive=.675f;\nrealtype adapter_head_bias=0;\n" + REFRESH)
    text = replace_once(text, "int main(void)", "int main(int argc, char **argv)")
    # Initial pose only: transform before IDA owns the state/history.
    text = replace_once(text, "  // Integration start time", r'''
  if(argc != 5) return 5;
  double px=atof(argv[1]), py=atof(argv[2]), angle=atof(argv[3]);
  for(int j=0;j<NBAR;++j) {
    double x=yval[3*j], y=yval[3*j+1];
    yval[3*j]=px+cos(angle)*x-sin(angle)*y;
    yval[3*j+1]=py+sin(angle)*x+cos(angle)*y;
    yval[3*j+2]+=angle;
  }
  // Integration start time''')
    start = text.index("  // Open output file")
    end = text.index("/*\n *--------------------------------------------------------------------\n * Model Functions", start)
    text = text[:start] + LOOP + text[end:]
    text = replace_once(text, "const float I_on = 0.675;", "const float I_on = adapter_drive;")
    anchor = "V_neuron[i][1] = NMJ_weight[i]*State[(int)(i*N_units/NSEG)][1] - NMJ_weight[i]*State[(int)(i*N_units/NSEG)][0];"
    text = replace_once(text, anchor, anchor + "\n        if(i<8) { V_neuron[i][0]+=adapter_head_bias; V_neuron[i][1]-=adapter_head_bias; }")
    generated = output / "worm-worker.cc"
    generated.write_text(text)
    (output / "source.patch").write_text("".join(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile="upstream/WormSim/Model/worm.cc", tofile="worm-worker.cc")))
    prefix = source / "WormSim/Sundials"
    cmd = ["g++", "-O3", "-Wno-deprecated", "-I", str(prefix / "include"), str(generated), "-L", str(prefix / "lib"), "-lsundials_ida", "-lsundials_cvode", "-lsundials_nvecserial", "-lm", "-o", str(output / "worker")]
    with (output / "build.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    meta = dict(source_commit=commit, source_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
                generated_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
                binary_sha256=hashlib.sha256((output / "worker").read_bytes()).hexdigest(),
                dt_s=.001, medium=1., build_command=cmd,
                protocol="v1: OK; time, microstep count, IDA internal time; pose147, derivative147; neural96, muscle96, stretch96, side_length96 (681 numbers)",
                coupling="accepted state stretch feedback; stop time at each microstep by default; persistent IDA history; muscle update affects next physical interval",
                controls="AVB-inspired drive [0,1]; engineered dorsal/ventral bias [-.1,.1] on first 8 effective segments; no anatomical mapping")
    (output / "build.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))
