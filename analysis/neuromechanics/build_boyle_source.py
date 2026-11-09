"""Build the pinned source's bundled Sundials and unmodified WormSim locally."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from run_boyle_probe import COMMIT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    src, out = a.source.resolve(), a.output.resolve()
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=src,text=True).strip() != COMMIT:
        raise ValueError("upstream commit does not match source lock")
    out.mkdir(parents=True, exist_ok=False)
    sundials=src/'WormSim/sundials-2.3.0'
    commands=[('configure',['./configure','CC=g++',f'--prefix={src}/WormSim/Sundials','--disable-mpi','--disable-fcmix'],sundials),
              ('make',['make','-j2'],sundials),('install',['make','install'],sundials),
              ('model',['make'],src/'WormSim/Model')]
    results=[]
    for name,cmd,cwd in commands:
        t=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            r=subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(step=name,command=cmd,returncode=r.returncode,wall_s=time.monotonic()-t))
        (out/'build.json').write_text(json.dumps(dict(source_commit=COMMIT,steps=results),indent=2))
        print(name,r.returncode,flush=True)
        if r.returncode:raise SystemExit(r.returncode)


if __name__=='__main__':main()
