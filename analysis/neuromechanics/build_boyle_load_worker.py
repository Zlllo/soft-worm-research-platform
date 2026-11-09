"""Add a local drag pulse and explicit stretch-length replay to the causal worker.

The baseline generated source and upstream checkout remain unchanged. All
generated variants retain the original Boyle/Berri/Cohen use notice.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

from run_boyle_probe import COMMIT, replace_once


HELPERS = r'''
// Explicit external-load and sensory-replay intervention; no pose edits.
double experiment_CN0[NBAR], experiment_CL0[NBAR];
vector<double> experiment_reference;
ofstream experiment_trace, experiment_record;
double experiment_load=1;
int experiment_begin() {
    if(sizeof(realtype)!=sizeof(double)) return 20;
    for(int i=0;i<NBAR;++i) {experiment_CN0[i]=CN[i];experiment_CL0[i]=CL[i];}
    experiment_trace.open(EXPERIMENT_TRACE,ios::binary);
    if(!experiment_trace) return 21;
    if(string(EXPERIMENT_RECORD).size()) {
        experiment_record.open(EXPERIMENT_RECORD,ios::binary);
        if(!experiment_record) return 22;
    }
    if(string(EXPERIMENT_REPLAY).size()) {
        ifstream in(EXPERIMENT_REPLAY,ios::binary|ios::ate);
        if(!in) return 23;
        streamsize bytes=in.tellg();
        if(bytes<=0 || bytes % (97*sizeof(double))) return 24;
        experiment_reference.resize(bytes/sizeof(double));
        in.seekg(0);in.read(reinterpret_cast<char*>(&experiment_reference[0]),bytes);
        if(!in) return 25;
    }
    return 0;
}
void experiment_apply_load(double interval_start) {
    experiment_load=(interval_start>=EXPERIMENT_ONSET-1e-10 &&
                     interval_start<EXPERIMENT_ONSET+EXPERIMENT_DURATION-1e-10)
                    ? EXPERIMENT_FACTOR : 1.;
    for(int i=0;i<NBAR;++i) {
        double multiplier=i<=NSEG/4 ? experiment_load : 1.;
        CN[i]=experiment_CN0[i]*multiplier;
        CL[i]=experiment_CL0[i]*multiplier;
    }
}
int experiment_feedback(double timenow,long step) {
    double actual[96],used[96];
    for(int i=0;i<NSEG;++i) for(int s=0;s<2;++s) actual[2*i+s]=L_SR[i][s];
    if(experiment_record.is_open()) {
        experiment_record.write(reinterpret_cast<char*>(&timenow),sizeof(double));
        experiment_record.write(reinterpret_cast<char*>(actual),sizeof(actual));
    }
    if(!experiment_reference.empty()) {
        long offset=step*97;
        if(offset+97>static_cast<long>(experiment_reference.size()) ||
           fabs(experiment_reference[offset]-timenow)>1e-10) {
            cerr << "replay time/length does not match current causal tick" << endl;
            return 26;
        }
        for(int i=0;i<NSEG;++i) for(int s=0;s<2;++s)
            L_SR[i][s]=experiment_reference[offset+1+2*i+s];
    }
    for(int i=0;i<NSEG;++i) for(int s=0;s<2;++s) used[2*i+s]=L_SR[i][s];
    update_SR(timenow);
    if((step+1)%EXPERIMENT_TRACE_STRIDE==0) {
        experiment_trace.write(reinterpret_cast<char*>(&timenow),sizeof(double));
        experiment_trace.write(reinterpret_cast<char*>(actual),sizeof(actual));
        experiment_trace.write(reinterpret_cast<char*>(used),sizeof(used));
        experiment_trace.write(reinterpret_cast<char*>(&experiment_load),sizeof(double));
    }
    // The public packet always reports actual geometry, even during replay.
    for(int i=0;i<NSEG;++i) for(int s=0;s<2;++s) L_SR[i][s]=actual[2*i+s];
    return 0;
}
'''


def build(source, base_build, output, *, factor=1., onset=6., duration=2., dt=.001,
          replay=None, record=None, trace_interval=.02):
    source,base_build,output=map(lambda p:Path(p).resolve(),(source,base_build,output))
    metadata=json.loads((base_build/'build.json').read_text())
    base=base_build/'worm-worker.cc'
    if hashlib.sha256(base.read_bytes()).hexdigest()!=metadata['generated_sha256']:
        raise ValueError('causal base source differs from its build record')
    if metadata['source_commit']!=COMMIT or subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=COMMIT:
        raise ValueError('upstream source lock mismatch')
    if not 0<factor<=4 or onset<0 or duration<=0 or dt not in (.001,.0005):
        raise ValueError('invalid experimental scales')
    stride=round(trace_interval/dt)
    if stride<1 or abs(stride*dt-trace_interval)>1e-10:
        raise ValueError('trace clock must align')
    if replay is not None and not Path(replay).is_file():raise FileNotFoundError(replay)
    output.mkdir(parents=True,exist_ok=False)
    trace=output/'feedback-trace.bin'
    constants={'EXPERIMENT_FACTOR':factor,'EXPERIMENT_ONSET':onset,'EXPERIMENT_DURATION':duration,
        'EXPERIMENT_TRACE_STRIDE':stride}
    definitions='\n'.join(f'#define {k} {v:.17g}' for k,v in constants.items())+'\n'
    for key,path in [('EXPERIMENT_TRACE',trace),('EXPERIMENT_RECORD',record),('EXPERIMENT_REPLAY',replay)]:
        value=str(Path(path).resolve()) if path is not None else ''
        definitions+=f'#define {key} '+json.dumps(value,ensure_ascii=False)+'\n'
    original=base.read_text()
    text=replace_once(original,'#include <string>','#include <string>\n#include <vector>')
    text=replace_once(text,'void adapter_refresh(realtype *yval) {',definitions+HELPERS+'\nvoid adapter_refresh(realtype *yval) {')
    # Prototype is required because the original update_SR declaration follows helpers.
    text=replace_once(text,'// Explicit external-load','void update_SR(realtype timenow);\n// Explicit external-load')
    text=replace_once(text,'  long steps=0;','  long steps=0;\n  int experiment_status=experiment_begin();\n  if(experiment_status) return experiment_status;')
    text=replace_once(text,'        tout=(steps+1)*DELTAT;','        experiment_apply_load(steps*DELTAT);\n        tout=(steps+1)*DELTAT;')
    text=replace_once(text,'        update_SR(tout);','        experiment_status=experiment_feedback(tout,steps);\n        if(experiment_status) return experiment_status;')
    text,n=re.subn(r'(?m)^#define DELTAT \S+',f'#define DELTAT {dt:.12g}',text)
    if n!=1:raise ValueError('ambiguous physical clock')
    generated=output/'worm-load-worker.cc';generated.write_text(text)
    (output/'source.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='causal/worm-worker.cc',tofile='worm-load-worker.cc')))
    prefix=source/'WormSim/Sundials'
    cmd=['g++','-O3','-Wno-deprecated','-I',str(prefix/'include'),str(generated),'-L',str(prefix/'lib'),
         '-lsundials_ida','-lsundials_cvode','-lsundials_nvecserial','-lm','-o',str(output/'worker')]
    with (output/'build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    meta=dict(metadata,dt_s=dt,build_command=cmd,base_generated_sha256=metadata['generated_sha256'],
        generated_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256((output/'worker').read_bytes()).hexdigest(),
        intervention=dict(factor=factor,onset_s=onset,duration_s=duration,affected_rods=list(range(13)),
            replay=str(Path(replay).resolve()) if replay is not None else None,
            replay_sha256=hashlib.sha256(Path(replay).read_bytes()).hexdigest() if replay is not None else None,
            reference_record=str(Path(record).resolve()) if record is not None else None,
            trace_format='native float64 rows: time, actual96, used96, load_factor; 194 columns',
            reference_format='native float64 rows: time, actual96; 97 columns',trace_dt_s=trace_interval))
    (output/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
    return output/'worker'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--base-build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(build(args.source,args.base_build,args.output))
