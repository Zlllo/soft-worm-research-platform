"""Build the isolated numerical variant without changing original C1 exports."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def build(source,output):
    output.mkdir(exist_ok=False)
    for p in source.iterdir():
        if p.suffix in ('.mod','.hoc'):shutil.copyfile(p,output/p.name)
    for name in ['neuron_to_neuron_exc_syn.mod','neuron_to_neuron_inh_syn.mod']:
        p=output/name;t=p.read_text()
        assert t.count("s' = rate_s")==1
        t=t.replace("s' = rate_s","if ((1-inf) > 1e-4) {\n        s' = (inf-s)/tau\n    } else {\n        s' = 0\n    }")
        p.write_text(t)
    with (output/'build.log').open('w') as f:
        subprocess.run([str(Path(sys.executable).parent/'nrnivmodl')],cwd=output,stdout=f,stderr=subprocess.STDOUT,check=True)
    manifest={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in output.rglob('*') if p.is_file() and p.suffix in ('.mod','.hoc','.dylib','.so')}
    (output/'variant.json').write_text(json.dumps(dict(source=str(source),sha256=manifest,
        scope='Only direct linear synaptic state expression; original fast-reset condition retained'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.source,a.output)
