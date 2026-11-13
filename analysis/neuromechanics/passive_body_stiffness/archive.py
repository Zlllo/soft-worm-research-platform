"""Archive passive stiffness inputs/results and recompute without simulations."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ROOT=Path('results/passive-body-stiffness-2026-11-13')
ASSETS=Path('docs/reports/assets/passive-body-stiffness-2026-11-13')
REPORT=Path('docs/reports/passive-body-stiffness-2026-11-13.md')


def main():
    files=set()
    for tree in [ROOT,ASSETS,Path('analysis/neuromechanics/passive_body_stiffness'),
                 Path('src/thermotaxis'),Path('data/connectome')]:
        for path in tree.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.name not in {
                'reproduction.zip','reproduction.zip.sha256','archive-validation.json','independent-recompute.log'}:
                files.add(path)
    files.add(REPORT)
    for name in json.loads((ROOT/'source-lock.json').read_text()):
        path=Path(name);assert not path.is_absolute();files.add(path)
    sundials=Path('results/neuromechanics-fastbody-2026-10-27/upstream/boyle2012/WormSim/Sundials')
    files.update((sundials/'include').rglob('*.h'))
    files.update(sundials/'lib'/name for name in ['libsundials_ida.a','libsundials_cvode.a','libsundials_nvecserial.a'])
    files.add(Path('results/full302-mechanism-stage-2026-11-07/licenses/Sundials-2.3.0-LICENSE.txt'))
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    archive=ASSETS/'reproduction.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,str(p))
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    target=Path(tempfile.mkdtemp(prefix='worm-passive-elasticity-'))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;z.extractall(target)
    for name,digest in manifest.items():
        assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest,name
    before=(target/ASSETS/'summary.json').read_bytes()
    environment=dict(os.environ,PYTHONPATH='src',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/worm-passive-mpl')
    with (ROOT/'independent-recompute.log').open('w') as stream:
        subprocess.run([sys.executable,'analysis/neuromechanics/passive_body_stiffness/study.py','summarize'],
                       cwd=target,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
    assert before==(target/ASSETS/'summary.json').read_bytes(),'Independent summary differs'
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (ASSETS/'reproduction.zip.sha256').write_text(digest+'  reproduction.zip\n')
    record=dict(payload_files=len(manifest),archive_bytes=archive.stat().st_size,archive_sha256=digest,
                crc_passed=True,all_payload_sha256_passed=True,independent_archive_summary_byte_identical=True,
                source_locks_verified_without_original_tree=True,new_simulations_in_archive_check=0,
                extracted_to=str(target))
    (ASSETS/'archive-validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
