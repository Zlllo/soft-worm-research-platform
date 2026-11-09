"""Equalize agar drag eigenvalues while preserving their arithmetic mean."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def build(base,output,sundials):
    output.mkdir(parents=True,exist_ok=False)
    t=(base/'worm-muscle-worker.cc').read_text()
    old=['CL_agar = RCONST(3.2e-3/(2.0*NBAR))','CN_agar = RCONST(128e-3/(2.0*NBAR))']
    for key in old:assert t.count(key)==1
    for key in old:t=t.replace(key,key.split('=')[0]+'= RCONST(65.6e-3/(2.0*NBAR))')
    (output/'worm-muscle-worker.cc').write_text(t)
    meta=json.loads((base/'build.json').read_text())
    command=['g++','-O3','-Wno-deprecated','-I',str((sundials/'include').resolve()),
        str((output/'worm-muscle-worker.cc').resolve()),'-L',str((sundials/'lib').resolve()),
        '-lsundials_ida','-lsundials_cvode','-lsundials_nvecserial','-lm','-o',str((output/'worker').resolve())]
    with (output/'build.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    meta.update(build_command=command,
        generated_sha256=hashlib.sha256(t.encode()).hexdigest(),
        binary_sha256=hashlib.sha256((output/'worker').read_bytes()).hexdigest(),
        drag_intervention='CN=CL=65.6e-3/(2*NBAR), preserving original mean eigenvalue; medium1')
    (output/'build.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sundials',type=Path,default=Path('results/neuromechanics-fastbody-2026-10-27/upstream/boyle2012/WormSim/Sundials'))
    a=p.parse_args();build(a.base,a.output,a.sundials)
