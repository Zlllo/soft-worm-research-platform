"""Archive and independently recompute only this branch's saved experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ROOT=Path('results/network-output-replay-2026-11-13')
ASSETS=Path('docs/reports/assets/network-output-replay-2026-11-13')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root,assets):
    for p,h in json.loads((root/'analysis-v2-lock.json').read_text()).items():assert digest(p)==h,p
    files=set()
    trees=[root,assets,Path('analysis/neuromechanics/network_output_replay'),Path('src/thermotaxis'),Path('data/connectome')]
    for tree in trees:
        for p in tree.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name not in {
                'reproduction.zip','reproduction.zip.sha256','archive-validation.json'}:files.add(p)
    files.add(Path('docs/reports/network-output-replay-2026-11-13.md'))
    for p in json.loads((root/'source-lock.json').read_text()):files.add(Path(p))
    license_source=Path('results/full302-mechanism-stage-2026-11-07/licenses/Sundials-2.3.0-LICENSE.txt')
    if license_source.exists():
        (root/'licenses').mkdir(exist_ok=True)
        target=root/'licenses'/'Sundials-2.3.0-LICENSE.txt'
        target.write_bytes(license_source.read_bytes());files.add(target)
    manifest={str(p):digest(p) for p in sorted(files)}
    archive=assets/'reproduction.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,str(p))
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    extracted=Path(tempfile.mkdtemp(prefix='worm-network-replay-'))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;z.extractall(extracted)
    for p,h in manifest.items():assert digest(extracted/p)==h,p
    before=(extracted/assets/'summary.json').read_bytes()
    env=dict(os.environ,PYTHONPATH='src',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/worm-network-replay-mpl')
    with (root/'independent-recompute.log').open('w') as log:
        subprocess.run([sys.executable,'analysis/neuromechanics/network_output_replay/analyze_v2.py',
                        '--root',str(root),'--output',str(assets)],cwd=extracted,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    assert before==(extracted/assets/'summary.json').read_bytes()
    h=digest(archive);(assets/'reproduction.zip.sha256').write_text(h+'  reproduction.zip\n')
    record=dict(payload_files=len(files),all_crc_passed=True,all_sha256_passed=True,
                archive_only_recompute_byte_identical=True,new_simulations_during_audit=0,
                archive_sha256=h,archive_bytes=archive.stat().st_size,extracted_to=str(extracted))
    (assets/'archive-validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--assets',type=Path,default=ASSETS);a=p.parse_args();main(a.root,a.assets)
