"""Package fixed results; verify CRC/SHA and reanalyze outside the workspace.

Usage: joint_archive.py build|validate|complete
complete performs two passes so the final report can cite an already observed
offline payload check, while the final ZIP certificate remains external.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

PROJECT=Path(__file__).resolve().parents[3]
REL=Path('results/full302-joint-numerics-2026-11-12')
ROOT=PROJECT/REL
ASSETS=PROJECT/'docs/reports/assets/full302-joint-numerics-2026-11-12'
REPORT=PROJECT/'docs/reports/full302-joint-numerics-2026-11-12.md'
SCRIPTS=PROJECT/'analysis/neuromechanics/b_motor_validation'
ARCHIVE=ASSETS/'reproduction.zip'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d): p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def protection_check():
    """Rehash the actual original inputs, not only their new snapshot copies."""
    checks={}
    for name,d in read(ROOT/'sources.json').items():
        origin=d['source']
        source=ROOT/origin if origin.startswith('candidate-reference/') else PROJECT/origin
        assert source.is_file(),str(source)
        assert sha(source)==d['sha256'],str(source)
        checks[origin]=d['sha256']
    # Check the complete frozen parent execution snapshot, including old binary.
    parent=PROJECT/'results/b-motor-convergence-2026-11-09'
    for name,d in read(ROOT/'parent-evidence/inputs.json').items():
        assert sha(parent/name)==d['sha256'],str(parent/name)
        checks[str((parent/name).relative_to(PROJECT))]=d['sha256']
    result=dict(all_original_sources_unchanged=True,unique_original_files_checked=len(checks),
                source_sha256=checks,scientific_runs_added=0)
    write(ROOT/'baseline-protection.json',result)
    return result


def build():
    s=read(ROOT/'summary.json');v=read(ROOT/'validation.json')
    assert s['final_budget_complete'] and s['new_full_runs']==6
    assert v['independent_candidate_audit_passed'] or v['incomplete_or_rhythm_failures_retained']
    assert v['summary_sha256']==sha(ROOT/'summary.json')
    protection_check()
    ASSETS.mkdir(parents=True,exist_ok=True)
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and p!=ROOT/'archive-validation.json']
    files += [p for p in SCRIPTS.glob('joint_*.py') if not p.name.startswith('joint_review')]
    files += [REPORT]
    files=sorted(set(files))
    manifest={str(p.relative_to(PROJECT)):sha(p) for p in files}
    instructions='''# 联合数值修正：独立复算归档

本包包含六条新候选运行、六条明确标注的历史对照、冻结源代码／参数／身体程序、生成代码、协议、运行记录及中文报告。
没有将六条旧轨迹替代新运行。6个新条件的执行已经结束，完整性及失败情况以summary为准；预算用尽，不需要启动NEURON。

## 离线分析

在解压目录执行（环境要求：Python3.12、NumPy2.3.5、SciPy1.16.3）：

```
python analysis/neuromechanics/b_motor_validation/joint_analyze.py
python analysis/neuromechanics/b_motor_validation/joint_verify.py
```

上述步骤只读取本包数据，不导入NEURON、不编译、不启动科学仿真。最终summary及validation应与包内原件相同。
独立检查保留冻结的NumPy计算公式，不导入主分析器。旧对照只用于带标签的模型变化和误差迁移比。

MANIFEST.json列出其余全部有效载荷的SHA256。最终ZIP的SHA256和解压重算证书置于包外，避免自引用。
archive-payload-validation.json若存在，是最终报告生成前一轮实际完成的解压检查；最终证书另见外置archive-validation.json。

## 冻结运行材料

内置NEURON生成二进制和身体程序为本次macOS arm64环境产物。它们用于记录实际执行材料；跨平台重建需另行建立环境。
joint_prepare/build/run/batch是本次已执行操作的来源记录，不是离线复算入口；prepare还引用受保护的父研究资料。
独立目录中的6条spec已存在，运行器拒绝覆盖和重跑。此归档不授权额外科学运行。
'''
    payload=instructions.encode()
    manifest['REPRODUCE.md']=hashlib.sha256(payload).hexdigest()
    with zipfile.ZipFile(ARCHIVE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:
            assert not p.is_symlink(),str(p)
            z.write(p,str(p.relative_to(PROJECT)))
        z.writestr('REPRODUCE.md',payload)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(archive=str(ARCHIVE),payload_files=len(manifest),bytes=ARCHIVE.stat().st_size)))


def validate(final=False):
    expected={name:sha(ROOT/name) for name in ['summary.json','validation.json','independent-validation.json']}
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('MANIFEST.json'))
        assert set(z.namelist())==set(manifest)|{'MANIFEST.json'}
        with tempfile.TemporaryDirectory(prefix='worm-joint-offline-',dir='/private/tmp') as td:
            destination=Path(td).resolve()
            for name in z.namelist():
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
            z.extractall(destination)
            for name,digest in manifest.items():assert sha(destination/name)==digest,name
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            env.pop('PYTHONPATH',None)
            logs=[]
            # Enforce that reanalysis never opens the original project or imports
            # NEURON. Installed NumPy/SciPy and Python libraries remain available.
            bootstrap='''import sys, pathlib, runpy
original=pathlib.Path(sys.argv[2]).resolve()
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=pathlib.Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
        if p.is_relative_to(original): raise RuntimeError('Forbidden original-project read: '+str(p))
    if event=='import' and args[0].split('.')[0]=='neuron': raise RuntimeError('Forbidden NEURON import')
sys.addaudithook(guard)
script=sys.argv[1];sys.argv=[script];runpy.run_path(script,run_name='__main__')
'''
            for script in ['joint_analyze.py','joint_verify.py']:
                command=[sys.executable,'-c',bootstrap,str(destination/'analysis/neuromechanics/b_motor_validation'/script),str(PROJECT)]
                result=subprocess.run(command,cwd=destination,env=env,capture_output=True,text=True,timeout=120)
                logs.append(dict(script=script,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
                if result.returncode:
                    write(ROOT/'archive-failed-validation.json',dict(logs=logs));raise RuntimeError(result.stderr)
            observed={name:sha(destination/REL/name) for name in expected}
            assert observed==expected,(observed,expected)
            numerical=read(destination/REL/'summary.json')
    result=dict(zip_crc_passed=True,all_payload_sha256_passed=True,payload_files=len(manifest),
        archive_sha256=sha(ARCHIVE),archive_bytes=ARCHIVE.stat().st_size,
        summary_sha256=expected['summary.json'],offline_reanalysis_identical=True,
        offline_files_compared=expected,original_project_reads_forbidden=True,NEURON_import_forbidden=True,
        new_simulations=0,scientific_conditions=numerical['new_full_runs'],
        convergence_accepted=numerical['neural_step_convergence_accepted'],logs=logs)
    target=ROOT/('archive-validation.json' if final else 'archive-payload-validation.json')
    write(target,result)
    if final:
        write(ASSETS/'archive-validation.json',result)
        (ASSETS/'reproduction.zip.sha256').write_text(result['archive_sha256']+'  reproduction.zip\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['logs','offline_files_compared']}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','validate','complete'])
    action=parser.parse_args().action
    if action=='build':build()
    elif action=='validate':validate(final=True)
    else:
        build();validate()
        subprocess.run([sys.executable,str(SCRIPTS/'joint_report.py')],check=True)
        build();validate(final=True)


if __name__=='__main__':main()
