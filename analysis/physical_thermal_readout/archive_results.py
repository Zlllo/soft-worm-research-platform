"""Archive branch C and recompute its fixed estimator from extracted inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import numpy as np


ROOT=Path('results/physical-thermal-readout-2026-11-13')
ASSETS=Path('docs/reports/assets/physical-thermal-readout-2026-11-13')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths=set()
    for tree in [ROOT,ASSETS,Path('analysis/physical_thermal_readout')]:
        for path in tree.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.name not in {
                'reproduction.zip','reproduction.zip.sha256','archive-validation.json'}:
                paths.add(path)
    paths.add(Path('docs/reports/physical-thermal-readout-2026-11-13.md'))
    manifest={str(p):sha(p) for p in sorted(paths)}
    archive=ASSETS/'reproduction.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as handle:
        for path in sorted(paths):handle.write(path,str(path))
        handle.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    target=Path(tempfile.mkdtemp(prefix='worm-physical-thermal-'))
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        handle.extractall(target)
    for name,digest in manifest.items():assert sha(target/name)==digest,name
    output=Path('independently-recomputed')
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1')
    with (ROOT/'independent-recompute.log').open('w') as stream:
        subprocess.run([sys.executable,'analysis/physical_thermal_readout/assay.py','analyze',
                        '--output',str(output)],cwd=target,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
    byte_files=['summary.json','window-results.csv','geometry.json','controls.json']
    for name in byte_files:
        assert (target/ASSETS/name).read_bytes()==(target/output/name).read_bytes(),name
    before=np.load(target/ASSETS/'estimator-inputs.npz')
    after=np.load(target/output/'estimator-inputs.npz')
    assert before.files==after.files
    for name in before.files:assert np.array_equal(before[name],after[name],equal_nan=True),name
    digest=sha(archive)
    (ASSETS/'reproduction.zip.sha256').write_text(digest+'  reproduction.zip\n')
    record=dict(payload_files=len(paths),crc_passed=True,all_payload_sha256_passed=True,
                independent_summary_csv_geometry_controls_byte_identical=True,
                all_saved_estimator_inputs_and_outputs_array_identical=True,
                archive_sha256=digest,archive_bytes=archive.stat().st_size,
                original_worktree_reads=False,new_simulations=0,new_training=0,
                extracted_to=str(target))
    (ASSETS/'archive-validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
