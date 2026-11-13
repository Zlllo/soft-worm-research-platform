"""Package and independently recompute current review only, then record the pause."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from timing_review_static import ROOT, REPO, A, REPORT, sha, write

ASSETS=REPO/'docs/reports/assets/c1-current-timing-review-2026-11-10'
def main():
    result=json.loads((ROOT/'main-review.json').read_text())
    assert result['data_audit_passed'] and result['A_summary_crosscheck']['status']=='complete' and result['review_status']=='complete'
    ASSETS.mkdir(parents=True,exist_ok=True)
    files={}
    for p in ROOT.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ['archive-validation.json','stage-pause.json']:
            files[str(p.relative_to(REPO))]=p
    folders=['main-runs','reused-main','controls','frozen','proposed-variants','legacy-code','legacy',
             'protocol-history','review-context','context','approvals','execution-code']
    for folder in folders:
        for p in (A/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(REPO))]=p
    names=['protocol.json','protocol-lock.json','protocol-preflight.json','sources.json','reuse-audit.json','proposal-source-hashes.json',
        'presynaptic-input.npz','reused-main-sources.json','main-execution-lock.json','main-batch-status.json',
        'control-execution-lock.json','control-batch-status.json','generated-code-validation-v2.json','control-summary.json',
        'control-validation.json','review-source-manifest.json','current-runtime-defaults.json']
    for name in names:files[str((A/name).relative_to(REPO))]=A/name
    files[str((A/'summary.json').relative_to(REPO))]=ROOT/'A-final-summary-snapshot.json'
    for name in ['timing_main_run.py','timing_main_batch.py','timing_control_run.py']:
        p=REPO/'analysis/neuromechanics/b_motor_validation'/name;files[str(p.relative_to(REPO))]=p
    for p in (REPO/'analysis/neuromechanics/b_motor_validation').glob('timing_review*.py'):files[str(p.relative_to(REPO))]=p
    files[str(REPORT.relative_to(REPO))]=REPORT
    files['docs/reports/c1-current-timing-2026-11-10.md']=ROOT/'A-final-report-snapshot.md'
    manifest={name:sha(p) for name,p in sorted(files.items())}
    archive=ASSETS/'reproduction.zip'
    print('writing archive',len(files),'files',flush=True)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name,p in sorted(files.items()):
            # NPZ files already compress their numerical payload; avoid a second expensive deflate.
            z.write(p,name,compress_type=zipfile.ZIP_STORED if p.suffix=='.npz' else zipfile.ZIP_DEFLATED)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        z.writestr('READ_ME_REVIEW.txt','This archive recomputes saved main/control data only. Do not run timing_main_run.py or timing_control_run.py.\nRun the independent timing_review_controls.py, timing_review_main.py, timing_review_main_compare.py, timing_review_report.py from the extracted project root. Requires NumPy; NEURON is not imported.\nThe early static/revision audits retain source snapshots but are not the current recomputation entry points.\nUser requested pause after current review/archive; no new science is authorized.\n')
    extracted=Path(tempfile.mkdtemp(prefix='worm-current-timing-review-'))
    print('checking CRC and extracting',flush=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;z.extractall(extracted)
    print('verifying',len(manifest),'payload hashes',flush=True)
    for name,digest in manifest.items():assert sha(extracted/name)==digest,name
    localroot=extracted/ROOT.relative_to(REPO)
    expected={name:(localroot/name).read_bytes() for name in ['control-review.json','main-review-calculations.json','A-summary-crosscheck.json','main-review.json']}
    expected_report=(extracted/REPORT.relative_to(REPO)).read_bytes()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
    for script in ['timing_review_controls.py','timing_review_main.py','timing_review_main_compare.py','timing_review_report.py']:
        print('independent offline recomputation',script,flush=True)
        subprocess.run([sys.executable,'analysis/neuromechanics/b_motor_validation/'+script],cwd=extracted,env=env,check=True)
    for name,original in expected.items():assert (localroot/name).read_bytes()==original,name
    assert (extracted/REPORT.relative_to(REPO)).read_bytes()==expected_report
    digest=sha(archive)
    validation=dict(archive=str(archive.relative_to(REPO)),archive_sha256=digest,archive_bytes=archive.stat().st_size,
        payload_files=len(manifest),CRC_passed=True,all_payload_SHA_passed=True,
        independent_control_review_byte_identical=True,independent_main_calculations_byte_identical=True,
        independent_A_crosscheck_byte_identical=True,independent_final_main_review_byte_identical=True,
        independent_final_report_byte_identical=True,raw_main_traces=12,raw_control_traces=12,
        NEURON_loaded_or_scientific_simulations_during_validation=False,extracted_to=str(extracted))
    write(ROOT/'archive-validation.json',validation)
    (ASSETS/'reproduction.zip.sha256').write_text(digest+'  reproduction.zip\n')
    write(ROOT/'stage-pause.json',dict(status='paused',reason='user requested finish current report/archive then pause',
        completed=['12 control independent audit','12 main raw-trace audit','12 timestep pairs','same-dt signal contrasts/interactions','A final JSON/text comparison','self-contained archive CRC/SHA and independent offline recomputation'],
        not_completed=['full302/body validation with changed kernels','shared284 C1 neuron effects','remaining h/CaPool/NMJ timing questions','autonomous propulsion and thermotaxis'],
        new_scientific_runs_by_review_branch=0,review_branch_scientific_processes_running=0,
        no_next_stage_started=True,archive_validation_sha256=sha(ROOT/'archive-validation.json')))
    print(json.dumps(validation),flush=True)

if __name__=='__main__':main()
