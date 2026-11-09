"""Build a separate direct-muscle body, keeping the original worker unchanged."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

from build_boyle_worker import LOOP
from run_boyle_probe import COMMIT, replace_once


MUSCLE_LOOP = r'''
  // External drives replace the simplified locomotor network, not its muscles.
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
        adapter_emit(mem,steps,yval,ypval);
        continue;
    }
    long count=0;
    double commands[NSEG][2];
    bool valid = op == "MUSCLE" && bool(input >> count) && count>=1 && count<=1000000;
    for(int i=0;i<NSEG && valid;++i) for(int s=0;s<2;++s) {
        if(!(input >> commands[i][s]) || !isfinite(commands[i][s]) || fabs(commands[i][s])>1) valid=false;
    }
    if(!valid || (input >> extra)) {
        cout << "ERR invalid muscle command" << endl;
        continue;
    }
    for(long j=0;j<count;++j) {
        tout=(steps+1)*DELTAT;
        if(causal && IDASetStopTime(mem,tout)<0) return 3;
        retval=IDASolve(mem,tout,&tret,yy,yp,causal ? IDA_NORMAL_TSTOP : IDA_NORMAL);
        if(retval<0 || fabs(tret-tout)>1e-10) return 4;
        adapter_refresh(yval);
        update_SR(tout);  // Observation only: no hidden neural update follows.
        for(int i=0;i<NSEG;++i) for(int s=0;s<2;++s) V_neuron[i][s]=commands[i][s];
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


def build(source, base_build, output):
    source, base_build, output = [Path(p).resolve() for p in (source, base_build, output)]
    meta = json.loads((base_build / 'build.json').read_text())
    text = (base_build / 'worm-worker.cc').read_text()
    if hashlib.sha256((base_build / 'worm-worker.cc').read_bytes()).hexdigest() != meta['generated_sha256']:
        raise ValueError('base source differs from its lock')
    if meta['source_commit'] != COMMIT or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != COMMIT:
        raise ValueError('upstream commit differs from lock')
    if hashlib.sha256((source / 'WormSim/Model/worm.cc').read_bytes()).hexdigest() != meta['source_sha256']:
        raise ValueError('upstream source differs from base worker')
    generated = replace_once(text, LOOP, MUSCLE_LOOP)
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'worm-muscle-worker.cc'
    path.write_text(generated)
    (output / 'source.patch').write_text(''.join(difflib.unified_diff(text.splitlines(True), generated.splitlines(True), fromfile='causal-base/worm-worker.cc', tofile='worm-muscle-worker.cc')))
    prefix = source / 'WormSim/Sundials'
    command = ['g++', '-O3', '-Wno-deprecated', '-I', str(prefix / 'include'), str(path), '-L', str(prefix / 'lib'),
               '-lsundials_ida', '-lsundials_cvode', '-lsundials_nvecserial', '-lm', '-o', str(output / 'worker')]
    with (output / 'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = dict(meta, control_mode='external_muscle_drive',
        generated_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256((output / 'worker').read_bytes()).hexdigest(),
        base_generated_sha256=meta['generated_sha256'], build_command=command,
        controls='96 signed pre-filter muscle drives [-1,1]; no drive/head_bias or legacy neural updates',
        coupling='accepted physical state -> stretch observation -> external drive -> existing 100ms muscle filter; affects next physical interval',
        neural_packet_scope='neural96 now reports commanded muscle drive, not simulated neuron voltages')
    (output / 'build.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--base-build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.source, a.base_build, a.output), indent=2))
