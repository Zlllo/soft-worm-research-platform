"""Self-contained archive, per-file hashes, CRC and independent recomputation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ROOT=Path(__file__).resolve().parents[2]
STAGE='results/full302-mechanism-stage-2026-11-07'
ASSETS='docs/reports/assets/full302-mechanisms-2026-11-07'


def main():
    os.chdir(ROOT)
    files={}
    def add(path,name=None):files[name or str(path)]=Path(path)
    for tree in ['results/full302-operating-range-2026-11-07',
        'results/body-phase-coordination-2026-11-07','results/body-isotropic-drag-2026-11-07',
        'results/full302-modworm-reference-2026-11-07',STAGE,ASSETS]:
        for p in sorted(Path(tree).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name not in (
                'reproduction.zip','reproduction.zip.sha256','archive-validation.json','archive-payload.json'):
                add(p)
    scripts=['probe_full302_operating_range.py','modworm_python_bridge.py','run_full302_modworm_reference.py',
        'probe_body_phase_coordination.py','validate_graded_synapse.py','build_direct_cnexp_variant.py',
        'build_isotropic_body_variant.py','plot_full302_mechanisms.py','summarize_full302_mechanisms.py',
        'archive_full302_mechanisms.py']
    for name in scripts:add(Path('analysis/neuromechanics')/name)
    for name in ['docs/research-mainline.md','docs/reports/full302-mechanisms-2026-11-07.md']:add(Path(name))
    recovered=Path('/tmp/worm-loop-evidence-40zogygz')
    for p in sorted((recovered/'src').rglob('*.py')):add(p,str(p.relative_to(recovered)))
    for p in sorted((recovered/'data/connectome').iterdir()):
        if p.is_file():add(p,str(p.relative_to(recovered)))
    base=Path('results/full-network-body-interface-2026-11-05/build')
    for name in ['worker','worm-muscle-worker.cc','build.json']:add(base/name)
    numerical=Path('results/full302-c1-dynamics-2026-11-05/kernels/arm64')
    for name in ['neuron_to_neuron_exc_syn.cpp','neuron_to_neuron_inh_syn.cpp']:
        add(numerical/name,'numerical-evidence/original-'+name)
    sundials=Path('results/neuromechanics-fastbody-2026-10-27/upstream/boyle2012/WormSim/Sundials')
    for p in sorted((sundials/'include').rglob('*.h')):add(p,'third_party/sundials/'+str(p.relative_to(sundials)))
    for name in ['libsundials_ida.a','libsundials_cvode.a','libsundials_nvecserial.a']:
        add(sundials/'lib'/name,'third_party/sundials/lib/'+name)
    add(Path(STAGE)/'licenses/Sundials-2.3.0-LICENSE.txt','third_party/sundials/LICENSE')
    manifest={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sorted(files.items())}
    archive=Path(ASSETS)/'reproduction.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,p in sorted(files.items()):z.write(p,name)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    target=Path(tempfile.mkdtemp(prefix='worm-mechanisms-archive-'))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('CRC failure')
        z.extractall(target)
    for name,digest in manifest.items():
        if hashlib.sha256((target/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    before=(target/STAGE/'summary.json').read_bytes()
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['OPENBLAS_NUM_THREADS']='1'
    subprocess.run([sys.executable,'analysis/neuromechanics/summarize_full302_mechanisms.py'],cwd=target,env=env,check=True)
    if before!=(target/STAGE/'summary.json').read_bytes():raise RuntimeError('archive-only recomputation differed')
    # zipfile does not restore Unix executable bits. Smoke-test the extracted
    # binaries and full302 runtime using only the archived source/data.
    for p in [target/'results/full-network-body-interface-2026-11-05/build/worker',
              target/'results/body-isotropic-drag-2026-11-07/build/worker']:
        p.chmod(0o755)
    smoke="""import sys,json,numpy as np
sys.path.insert(0,'src')
from thermotaxis.neuromechanics.full302_c1 import C1Network
from thermotaxis.neuromechanics.boyle_muscle_body import BoyleMuscleBody
from pathlib import Path
root=Path('results/full302-operating-range-2026-11-07')
network=C1Network(root/'kernels-direct-cnexp',json.loads((root/'connection-plan.json').read_text()))
network.reset();network.advance(1.)
assert len(network.cells)==397
assert len(network.chemical_connections)==4665
assert np.isfinite(network.state()['neuron_voltage_mV']).all()
for binary in ['results/full-network-body-interface-2026-11-05/build/worker','results/body-isotropic-drag-2026-11-07/build/worker']:
    with BoyleMuscleBody(binary) as body:
        body.reset();state=body.step(3,muscle_drive=np.zeros((48,2)))
        assert abs(state.time_s-.003)<1e-12 and np.isfinite(state.pose).all()
print('Extracted full302 network and both body binaries passed smoke check')
"""
    subprocess.run([sys.executable,'-c',smoke],cwd=target,env=env,check=True)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(digest+'  reproduction.zip\n')
    record=dict(archive=str(archive),archive_sha256=digest,archive_bytes=archive.stat().st_size,
        payload_files=len(manifest),crc_passed=True,all_payload_sha256_passed=True,
        independent_archive_only_summary_byte_identical=True,
        extracted_full302_and_both_native_bodies_smoke_passed=True,extracted_to=str(target))
    (Path(STAGE)/'archive-validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
