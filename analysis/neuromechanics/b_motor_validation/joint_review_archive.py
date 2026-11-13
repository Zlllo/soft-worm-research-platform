"""Small review archive; raw data remain in hash-pinned A and parent archives.

No simulator import or execution. Extract all three archives together and run
the included review calculations using the declared NumPy/SciPy environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from joint_review_compute import REPO, OUT, A, OLD, read, save, digest, check

ASSETS=REPO/'docs/reports/assets/full302-joint-numerics-review-2026-11-12'


def safe_extract(archive,destination):
    with zipfile.ZipFile(archive) as z:
        check(z.testzip() is None,f'Archive CRC {archive}')
        for name in z.namelist():
            check(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe ZIP path')
        z.extractall(destination)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--a-archive',type=Path,required=True);args=parser.parse_args()
    args.a_archive=args.a_archive.resolve()
    check(read(OUT/'independent-results.json')['adaptive_batch_complete'],'Candidate batch incomplete')
    check(read(OUT/'A-summary-crosscheck.json')['passed'],'Candidate cross-check failed')
    parent=REPO/'docs/reports/assets/b-motor-convergence-2026-11-09/reproduction.zip'
    check(args.a_archive.exists() and parent.exists(),'Required raw archive missing')
    ASSETS.mkdir(parents=True,exist_ok=True)
    official=REPO/'results/c1-current-timing-review-2026-11-10'
    target=OUT/'review-inputs/official';target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(official/'official-sources.json',target/'official-sources.json')
    for name in ('fadvance.cpp','treeset.cpp','hh.mod'):
        (target/'official-9.0.2').mkdir(exist_ok=True)
        shutil.copy2(official/'official-9.0.2'/name,target/'official-9.0.2'/name)
    dependencies={
        'candidate_raw_archive':dict(path=str(args.a_archive.relative_to(REPO)),sha256=digest(args.a_archive),bytes=args.a_archive.stat().st_size),
        'parent_raw_archive':dict(path=str(parent.relative_to(REPO)),sha256=digest(parent),bytes=parent.stat().st_size),
        'runtime':dict(python='3.12.14',numpy='2.3.5',scipy='1.16.3',neuron_required=False),
        'instructions':'Verify both dependency SHA256, extract parent archive, candidate archive and this review archive into the same empty directory. Run only joint_review*.py in the order recorded below; these do not load the simulator.',
        'recompute_commands':[
            'python analysis/neuromechanics/b_motor_validation/joint_review_source.py',
            'python analysis/neuromechanics/b_motor_validation/joint_review_compute.py --parent',
            'python analysis/neuromechanics/b_motor_validation/joint_review_compare.py --parent',
            'python analysis/neuromechanics/b_motor_validation/joint_review_compute.py',
            'python analysis/neuromechanics/b_motor_validation/joint_review_compare.py',
            'python analysis/neuromechanics/b_motor_validation/joint_review_report.py']}
    save(OUT/'archive-dependencies.json',dependencies)
    # Do not put archive validation itself in the archive: it is a receipt about
    # these immutable bytes, avoiding a self-referential hash.
    excluded={'archive-validation.json','stage-complete.json'}
    payload=[p for p in OUT.rglob('*') if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts]
    payload+=list((REPO/'analysis/neuromechanics/b_motor_validation').glob('joint_review*.py'))
    payload+=[REPO/'docs/reports/full302-joint-numerics-review-2026-11-12.md',REPO/'docs/full302-joint-numerics-2026-11-12.md']
    manifest={str(p.relative_to(REPO)):digest(p) for p in sorted(payload)}
    archive=ASSETS/'reproduction.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(payload):z.write(p,str(p.relative_to(REPO)))
        z.writestr('REVIEW_MANIFEST.json',json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    with tempfile.TemporaryDirectory(prefix='worm-joint-B-extract-') as temporary:
        stage=Path(temporary)
        safe_extract(parent,stage);safe_extract(args.a_archive,stage);safe_extract(archive,stage)
        for path,sha in manifest.items():check(digest(stage/path)==sha,'Extracted review SHA '+path)
        expected={name:digest(OUT/name) for name in ('source-protocol-review.json','parent-independent.json',
            'parent-crosscheck.json','independent-results.json','A-summary-crosscheck.json','historical-comparison.json')}
        logs=[]
        bootstrap='''import pathlib,runpy,sys
script=pathlib.Path(sys.argv[1]).resolve()
original=pathlib.Path(sys.argv[2]).resolve()
arguments=sys.argv[3:]
def audit(event,args):
    if event=='import' and args[0].split('.')[0]=='neuron':
        raise RuntimeError('Simulator import forbidden in independent review replay')
    if event=='open' and isinstance(args[0],(str,bytes)):
        raw=args[0].decode() if isinstance(args[0],bytes) else args[0]
        if pathlib.Path(raw).resolve().is_relative_to(original):
            raise RuntimeError('Original workspace read forbidden: '+raw)
sys.addaudithook(audit)
sys.path.insert(0,str(script.parent))
sys.argv=[str(script)]+arguments
runpy.run_path(str(script),run_name='__main__')
'''
        environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        environment.pop('PYTHONPATH',None)
        for command in dependencies['recompute_commands']:
            tokens=command.split()
            invocation=[sys.executable,'-c',bootstrap,str(stage/tokens[1]),str(REPO),*tokens[2:]]
            result=subprocess.run(invocation,cwd=stage,env=environment,capture_output=True,text=True,timeout=120)
            logs.append(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
            if result.returncode:
                save(OUT/'archive-failed-validation.json',dict(logs=logs))
                raise AssertionError(dict(command=command,stdout=result.stdout,stderr=result.stderr))
        recomputed={name:digest(stage/OUT.relative_to(REPO)/name) for name in expected}
        check(expected==recomputed,'Independent extracted recomputation differs')
        report_path=Path('docs/reports/full302-joint-numerics-review-2026-11-12.md')
        check(digest(REPO/report_path)==digest(stage/report_path),'Recomputed report differs')
        links=re.findall(r'\[[^\]]+\]\(([^)]+)\)',(REPO/report_path).read_text())
        for link in links:
            if '://' not in link:check(((REPO/report_path).parent/link).resolve().exists(),'Broken report link '+link)
        receipt=dict(passed=True,archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,
            payload_files=len(manifest),all_payload_hashes_verified=True,all_archive_CRC_passed=True,
            independent_extracted_recomputation_byte_identical=True,recomputed_hashes=recomputed,
            saved_report_relative_links_verified=len(links),
            dependencies=dependencies,execution_logs=logs,simulator_imported=False,scientific_runs=0)
        receipt.update(original_workspace_reads_forbidden=True,simulator_import_forbidden=True)
    save(OUT/'archive-validation.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('passed','archive_sha256','archive_bytes','payload_files')}))


if __name__=='__main__':main()
