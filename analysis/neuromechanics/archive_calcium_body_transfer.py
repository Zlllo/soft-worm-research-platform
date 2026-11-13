"""Freeze this assay and independently recompute its summary after extraction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main(root, assets):
    files = set()
    for tree in [root, assets, Path('src/thermotaxis'), Path('data/connectome')]:
        for path in tree.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc' and path.name not in {
                'reproduction.zip', 'reproduction.zip.sha256', 'archive-validation.json'}:
                files.add(path)
    for name in json.loads((root/'source-lock.json').read_text()):
        path = Path(name)
        files.add(path.relative_to(Path.cwd()) if path.is_absolute() else path)
    files.update(Path('analysis/neuromechanics')/name for name in [
        'calcium_body_transfer.py', 'summarize_calcium_body_transfer.py', 'archive_calcium_body_transfer.py',
        'plot_phase_matched_transfer.py', 'recompute_calcium_archive.py'])
    files.update([Path('docs/calcium-body-transfer-2026-11-13.md'),
                  Path('docs/reports/calcium-body-transfer-2026-11-13.md'),
                  Path('tests/test_boyle_muscle_body.py')])
    sundials=Path('results/neuromechanics-fastbody-2026-10-27/upstream/boyle2012/WormSim/Sundials')
    files.update((sundials/'include').rglob('*.h'))
    files.update(sundials/'lib'/name for name in [
        'libsundials_ida.a', 'libsundials_cvode.a', 'libsundials_nvecserial.a'])
    files.add(Path('results/full302-mechanism-stage-2026-11-07/licenses/Sundials-2.3.0-LICENSE.txt'))
    manifest={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}
    archive=assets/'reproduction.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as handle:
        for path in sorted(files):handle.write(path,str(path))
        handle.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    target=Path(tempfile.mkdtemp(prefix='worm-calcium-transfer-'))
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        handle.extractall(target)
    for name,digest in manifest.items():
        assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest,name
    before=(target/assets/'summary.json').read_bytes()
    environment=dict(os.environ, PYTHONPATH='src', OPENBLAS_NUM_THREADS='1', MPLCONFIGDIR='/tmp/worm-transfer-mpl')
    # Keep absolute __file__ lock bytes intact while restricting all reads to
    # the extracted payload via the shipped, explicit relocation entry point.
    with (root/'independent-recompute.log').open('w') as stream:
        subprocess.run([sys.executable,'analysis/neuromechanics/recompute_calcium_archive.py',
                        '--root',str(root),'--output',str(assets)],cwd=target,env=environment,
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    assert before==(target/assets/'summary.json').read_bytes(), 'Recomputed summary differs'
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (assets/'reproduction.zip.sha256').write_text(digest+'  reproduction.zip\n')
    record=dict(payload_files=len(files),crc_passed=True,all_payload_sha256_passed=True,
                independent_archive_summary_byte_identical=True,archive_sha256=digest,
                archive_bytes=archive.stat().st_size,
                source_and_protocol_rechecked_after_extraction=True,
                absolute_lock_path_relocation=str(Path.cwd())+' -> extracted archive root; original hash bytes retained',
                new_simulations_in_archive_check=0,extracted_to=str(target))
    (assets/'archive-validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/calcium-body-transfer-2026-11-13'))
    parser.add_argument('--assets',type=Path,default=Path('docs/reports/assets/calcium-body-transfer-2026-11-13'))
    args=parser.parse_args();main(args.root,args.assets)
