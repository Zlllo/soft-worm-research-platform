"""Package only B-owned artifacts; hash and independently recompute after extract."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'results/b-motor-feedback-causality-2026-11-09'
ASSETS=REPO/'docs/reports/assets/b-motor-feedback-causality-2026-11-09'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    paths=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='archive-validation.json']
    paths+=list((REPO/'analysis/neuromechanics/b_motor_validation').glob('feedback*.py'))
    paths+=[REPO/'docs/reports/b-motor-feedback-causality-2026-11-09.md']
    manifest={str(p.relative_to(REPO)):sha(p) for p in sorted(paths)}
    archive=ASSETS/'reproduction.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in manifest:z.write(REPO/name,name)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    extracted=Path(tempfile.mkdtemp(prefix='worm-feedback-causality-'))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        z.extractall(extracted)
    assert all(sha(extracted/name)==digest for name,digest in manifest.items())
    old_summary=(extracted/ROOT.relative_to(REPO)/'summary.json').read_bytes()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None)
    subprocess.run([sys.executable,'analysis/neuromechanics/b_motor_validation/feedback_summarize.py'],cwd=extracted,env=env,check=True)
    assert old_summary==(extracted/ROOT.relative_to(REPO)/'summary.json').read_bytes()
    localroot=extracted/ROOT.relative_to(REPO)
    numerical_before=json.loads((localroot/'numerical-crosscheck.json').read_text())
    report_path=extracted/'docs/reports/b-motor-feedback-causality-2026-11-09.md'
    report_before=report_path.read_bytes()
    subprocess.run([sys.executable,'analysis/neuromechanics/b_motor_validation/feedback_numerical.py'],cwd=extracted,env=env,check=True,stdout=subprocess.DEVNULL)
    numerical_after=json.loads((localroot/'numerical-crosscheck.json').read_text())
    for d in [numerical_before,numerical_after]:
        d.pop('branch_A_summary_available',None);d.pop('numerical_reference_source',None)
    assert numerical_before==numerical_after
    subprocess.run([sys.executable,'analysis/neuromechanics/b_motor_validation/feedback_report.py'],cwd=extracted,env=env,check=True,stdout=subprocess.DEVNULL)
    assert report_before==report_path.read_bytes()
    result=dict(archive=str(archive.relative_to(REPO)),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        payload_files=len(manifest),CRC_passed=True,all_payload_SHA_passed=True,
        independent_summary_byte_identical=True,independent_numerical_values_identical=True,
        independent_report_byte_identical=True,new_simulations_during_verification=0,extracted_to=str(extracted))
    (ROOT/'archive-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    (ASSETS/'reproduction.zip.sha256').write_text(sha(archive)+'  reproduction.zip\n')
    print(json.dumps(result))

if __name__=='__main__':main()
