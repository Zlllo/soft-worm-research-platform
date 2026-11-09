"""Archive a paused stage and independently recalculate saved-data summaries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


REPO=Path(__file__).resolve().parents[2]
ROOT=Path('results/b-motor-mechanisms-2026-11-07')
ASSETS=Path('docs/reports/assets/b-motor-mechanisms-2026-11-07')


def main():
    os.chdir(REPO);files={}
    for base in [ROOT,ASSETS]:
        for p in sorted(base.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name not in ['reproduction.zip','reproduction.zip.sha256','archive-validation.json']:
                files[str(p)]=p
    for p in sorted(Path('analysis/neuromechanics').glob('*b_motor*.py')):files[str(p)]=p
    for p in [Path('src/thermotaxis/neuromechanics/b_motor_oscillator.py'),
        Path('analysis/neuromechanics/patches/BMotorPoint.mod'),
        Path('docs/reports/b-motor-mechanisms-2026-11-07.md'),Path('docs/research-mainline.md')]:files[str(p)]=p
    base=Path('results/full-network-body-interface-2026-11-05/build')
    for name in ['worker','worm-muscle-worker.cc','build.json']:files[str(base/name)]=base/name
    manifest={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sorted(files.items())}
    archive=ASSETS/'reproduction.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,p in sorted(files.items()):z.write(p,name)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    extracted=Path(tempfile.mkdtemp(prefix='worm-b-motor-archive-'))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;z.extractall(extracted)
    for name,digest in manifest.items():assert hashlib.sha256((extracted/name).read_bytes()).hexdigest()==digest,name
    original=(extracted/ROOT/'summary.json').read_bytes()
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['OPENBLAS_NUM_THREADS']='1'
    subprocess.run([sys.executable,'analysis/neuromechanics/summarize_b_motor_stage.py'],cwd=extracted,env=env,check=True)
    assert original==(extracted/ROOT/'summary.json').read_bytes()
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(digest+'  reproduction.zip\n')
    record=dict(archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=digest,payload_files=len(files),
        crc_passed=True,all_payload_hashes_passed=True,independent_summary_byte_identical=True,
        new_simulations_run_during_archive_check=False,extracted_to=str(extracted))
    (ROOT/'archive-validation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':main()
