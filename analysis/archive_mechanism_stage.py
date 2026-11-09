"""Package a completed mechanism stage and verify every archived payload."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--include',action='append',default=[])
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    files={}
    for path in root.rglob('*'):
        if out in path.parents:continue
        if path.is_file() and path.suffix in ('.npz','.pt','.bin','.cc','.patch','.json','.log'):
            files['evidence/'+str(path.relative_to(root))]=path
    for name in a.include+['analysis/archive_mechanism_stage.py']:
        path=(REPO/name).resolve();files[str(path.relative_to(REPO))]=path
    for path in out.iterdir():
        if path.is_file() and path.suffix in ('.png','.svg','.json') and path.name!='manifest.json':
            files['artifacts/'+path.name]=path
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sorted(files.items())}
    archive=out/'reproduction.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,path in sorted(files.items()):z.write(path,name)
        z.writestr('manifest.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,sha in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
    (out/'manifest.json').write_text(json.dumps(dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=hashes),indent=2)+'\n')
    print(json.dumps(dict(files=len(files),bytes=archive.stat().st_size)))


if __name__=='__main__':main()
