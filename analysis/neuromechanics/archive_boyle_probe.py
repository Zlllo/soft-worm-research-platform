"""Save validated source/trace evidence for the fast-body evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();base=a.run_dir.resolve();dest=a.destination.resolve()
    dest.mkdir(parents=True,exist_ok=False)
    for name in ('summary.json','fastbody-probes.png','fastbody-probes.svg'):
        shutil.copy2(base/name,dest/name)
    files={}
    for name in ('run_boyle_probe.py','audit_boyle_probe.py','build_boyle_source.py','archive_boyle_probe.py'):
        path=ROOT/'analysis/neuromechanics'/name;files[str(path.relative_to(ROOT))]=path
    for name in ('src/thermotaxis/neuromechanics/boyle_output.py','src/thermotaxis/neuromechanics/__init__.py',
                 'tests/test_boyle_output.py','configs/neuromechanics/boyle-source-lock.json'):
        files[name]=ROOT/name
    for path in base.rglob('*'):
        relative=path.relative_to(base)
        if not path.is_file() or relative.parts[0]=='upstream':continue
        if path.suffix in ('.json','.csv','.log','.cc','.patch','.png','.svg'):
            files['evidence/'+str(relative)]=path
    src=base/'upstream/boyle2012'
    for name in ('WormSim/Model/worm.cc','WormSim/README.md','WormSim/sundials-2.3.0/LICENSE'):
        files['upstream-reference/'+name]=src/name
    manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sorted(files.items())}
    archive=dest/'reproduction.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,path in sorted(files.items()):z.write(path,name)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,sha in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
    (dest/'manifest.json').write_text(json.dumps(dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=manifest),indent=2))
    print(json.dumps(dict(files=len(files),bytes=archive.stat().st_size)))


if __name__=='__main__':main()
