"""One-time six-case main authorization, exact old-data reuse, no retries."""
import ast
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'results/c1-current-timing-2026-11-10'
EXPECTED='8b9d11bda03f952c96d64d5b2b58f92496ecdcc008e5c471a51463664320f12a'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def main():
    authorization=PROJECT/'results/c1-current-timing-main-2026-11-10/main-run-authorization.json'
    approval=read(authorization);summary=read(ROOT/'control-summary.json');validation=read(ROOT/'control-validation.json')
    assert sha(ROOT/'protocol.json')==EXPECTED==approval['protocol_sha256']
    assert approval['status']=='approved_six_main_conditions' and approval['main_new_run_cap']==6
    assert sha(ROOT/'control-summary.json')==approval['control_summary_sha256']==validation['control_summary_sha256']
    assert summary['all_control_gates_passed'] and validation['independent_control_audit_passed']
    assert not (ROOT/'main-execution-lock.json').exists() and not (ROOT/'main-runs').exists()
    shutil.copy2(authorization,ROOT/'approvals/main-run-authorization.json')
    runner=Path(__file__).with_name('timing_main_run.py')
    tree=ast.parse(runner.read_text())
    forbidden=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and (node.func.attr=='fcurrent' or node.func.attr.startswith('rates')):forbidden.append(node.func.attr)
    assert not forbidden
    kernels={}
    for variant in ['euler_direct','exp_direct']:
        base=ROOT/'proposed-variants'/variant
        kernels[variant]={str(p.relative_to(base)):sha(p) for p in base.rglob('*') if p.is_file()}
    reused={}
    for name,audit in read(ROOT/'reuse-audit.json').items():
        source=PROJECT/audit['source'];target=ROOT/'reused-main'/name;target.mkdir(parents=True)
        assert sha(source/'trace.npz')==audit['trace_sha256']
        assert sha(source/'record.json')==audit['record_sha256'] and sha(source/'spec.json')==audit['spec_sha256']
        for file in ['trace.npz','record.json','spec.json']:
            shutil.copy2(source/file,target/file);assert sha(source/file)==sha(target/file)
        reused[name]=dict(original_source=audit['source'],trace_sha256=audit['trace_sha256'],
            record_sha256=audit['record_sha256'],spec_sha256=audit['spec_sha256'],
            secondorder=0,secondorder_evidence='INFERRED same-version source/default; not present in historical record; accepted by main/B')
    write(ROOT/'reused-main-sources.json',reused)
    snapshot=ROOT/'execution-code';snapshot.mkdir(exist_ok=True);shutil.copy2(runner,snapshot/runner.name)
    lock=dict(protocol_sha256=EXPECTED,runner_sha256=sha(runner),batch_script_sha256=sha(Path(__file__)),
        main_authorization_sha256=sha(authorization),control_summary_sha256=sha(ROOT/'control-summary.json'),
        control_validation_sha256=sha(ROOT/'control-validation.json'),input_sha256=sha(ROOT/'presynaptic-input.npz'),
        generated_code_validation_sha256=sha(ROOT/'generated-code-validation-v2.json'),kernel_sha256=kernels,
        reused_sources_sha256=sha(ROOT/'reused-main-sources.json'),main_conditions_authorized=6,retries_allowed=0,
        observers=0,explicit_fcurrent_calls=0,full302_body_authorized=0)
    write(ROOT/'main-execution-lock.json',lock)
    jobs=[(v,dt) for v in ['euler_direct','exp_direct'] for dt in [.025,.0125,.00625]];statuses={}
    def launch(job):
        v,dt=job;name=f'{v}_dt{dt:g}'
        with (ROOT/f'main_{name}.log').open('w') as log:
            code=subprocess.run([sys.executable,str(runner),'--variant',v,'--dt',str(dt)],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1'}).returncode
        return name,dict(exit_code=code,complete=code==0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(launch,j) for j in jobs]):
            name,result=future.result();statuses[name]=result
            write(ROOT/'main-batch-status.json',dict(declared_new_conditions=6,status=statuses,retries=0,new_full302_body=0))
            print(name,result,flush=True)
    assert all(d['complete'] for d in statuses.values()), 'Failure preserved, no retry'


if __name__=='__main__':main()
