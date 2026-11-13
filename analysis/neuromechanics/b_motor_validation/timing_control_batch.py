"""Authorize and run exactly eight zero-step and four short V2 controls."""
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
VARIANTS=['euler_cached','exp_cached','euler_direct','exp_direct']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def main():
    approval=PROJECT/'results/c1-current-timing-main-2026-11-10/protocol-review.json'
    review=PROJECT/'results/c1-current-timing-review-2026-11-10/protocol-review-v2.json'
    a=read(approval);b=read(review);g=read(ROOT/'generated-code-validation-v2.json')
    assert sha(ROOT/'protocol.json')==EXPECTED==a['protocol_sha256']==b['selected_protocol_sha256']==g['protocol_sha256']
    assert a['status']=='approved_control_first' and a['small_controls_cap']==12 and a['observer_controls']==0
    assert not b['findings'] and g['passed']
    assert not (ROOT/'controls').exists() and not (ROOT/'control-execution-lock.json').exists(), 'Never retry controls in place'
    shutil.copy2(approval,ROOT/'approvals/main-control-authorization-v2.json')
    shutil.copy2(review,ROOT/'approvals/independent-control-review-v2.json')
    kernels={}
    for v in VARIANTS:
        base=ROOT/'frozen'/('original' if v.startswith('euler') else 'both') if v.endswith('cached') else ROOT/'proposed-variants'/v
        kernels[v]={str(p.relative_to(base)):sha(p) for p in base.rglob('*') if p.is_file()}
    runner=Path(__file__).with_name('timing_control_run.py')
    lock=dict(protocol_sha256=EXPECTED,scientific_stage='v2_controls_only',runner_sha256=sha(runner),
        batch_script_sha256=sha(Path(__file__)),generated_code_validation_sha256=sha(ROOT/'generated-code-validation-v2.json'),
        main_approval_sha256=sha(approval),independent_review_sha256=sha(review),kernel_sha256=kernels,
        zero_step_controls=8,short_controls=4,max_short_duration_ms=5,main_runs_authorized=0,
        full302_body_runs_authorized=0,observer_kernels=0,retries_allowed=0)
    write(ROOT/'control-execution-lock.json',lock)
    jobs=[('zero',v,voltage) for v in VARIANTS for voltage in [-20,-10]]+[('short',v,-10) for v in VARIANTS]
    statuses={}
    for kind,v,voltage in jobs:
        name=f'{kind}_{v}_v{voltage:g}'
        with (ROOT/f'{name}.log').open('w') as log:
            result=subprocess.run([sys.executable,str(runner),'--kind',kind,'--variant',v,'--voltage',str(voltage)],
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
        statuses[name]=dict(exit_code=result.returncode,complete=result.returncode==0)
        write(ROOT/'control-batch-status.json',dict(stage='controls_only',declared_conditions=12,status=statuses,main_runs=0))
        print(name,'complete' if result.returncode==0 else 'FAILED (retained, no retry)',flush=True)
    assert all(d['complete'] for d in statuses.values()), 'Control failure retained; main stage blocked'


if __name__=='__main__':main()
