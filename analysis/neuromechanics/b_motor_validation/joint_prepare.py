"""Pin all inputs and original acceptance rules before a candidate run."""
from datetime import datetime,timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'results/full302-joint-numerics-2026-11-12'
PARENT=PROJECT/'results/b-motor-convergence-2026-11-09'
CANDIDATE=PROJECT/'results/c1-current-timing-2026-11-10/proposed-variants/exp_direct'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    import neuron,numpy,scipy
    assert (neuron.__version__,numpy.__version__,scipy.__version__)==('9.0.2','2.3.5','1.16.3')
    assert not (ROOT/'protocol.json').exists()
    assert {p.name for p in ROOT.iterdir()} <= {'environment','task-control.json'}
    ROOT.mkdir(exist_ok=True)
    parent_inputs=read(PARENT/'inputs.json');sources={}
    def copy(source,target,expected=None):
        if expected is not None:assert sha(source)==expected,str(source)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        assert sha(source)==sha(target)
        sources[str(target.relative_to(ROOT))]=dict(source=str(source.relative_to(PROJECT)),sha256=sha(target))
    for folder in ['runtime','kernels']:
        for path in sorted((PARENT/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                relative=str(path.relative_to(PARENT))
                assert relative in parent_inputs,relative
                copy(path,ROOT/relative,parent_inputs[relative]['sha256'])
    for name in ['selected-parameters.json','connection-plan.json','requirements.txt']:
        copy(PARENT/name,ROOT/name,parent_inputs[name]['sha256'])
    for name in ['protocol.json','protocol-lock.json','summary.json','archive-validation.json','inputs.json']:
        copy(PARENT/name,ROOT/'parent-evidence'/name)
    for mode in ['off','live']:
        for dt in [.025,.0125,.00625]:
            name=f'{mode}_dt{dt:g}';record=read(PARENT/name/'record.json')
            assert record['complete']
            for file in ['trace.npz','record.json','spec.json']:
                copy(PARENT/name/file,ROOT/'historical'/name/file,record['trace_sha256'] if file=='trace.npz' else None)
    with zipfile.ZipFile(PROJECT/'docs/reports/assets/b-motor-convergence-2026-11-09/reproduction.zip') as z:manifest=json.loads(z.read('MANIFEST.json'))
    for name,target in [('convergence_analyze.py','primary.py'),('convergence_verify.py','independent.py')]:
        source=PROJECT/'analysis/neuromechanics/b_motor_validation'/name
        copy(source,ROOT/'frozen-analysis'/target,manifest[str(source.relative_to(PROJECT))])
    copy(PROJECT/'docs/full302-joint-numerics-2026-11-12.md',ROOT/'context/handoff.md')
    verified=read(PROJECT/'results/c1-current-timing-2026-11-10/generated-code-validation-v2.json')['variants']['exp_direct']
    changes={
        'ca_boyle.mod':[("e_q' = rate_e_q","e_q' = (e_inf-e_q)/e_tau"),("f_q' = rate_f_q","f_q' = (f_inf-f_q)/f_tau"),
            ('fopen0 = e_fcond * f_fcond * h_fcond','fopen0 = (e_q ^ e_instances) * (f_q ^ f_instances) * h_fcond')],
        'k_fast.mod':[("p_q' = rate_p_q","p_q' = (p_inf-p_q)/p_tau"),("q_q' = rate_q_q","q_q' = (q_inf-q_q)/q_tau"),
            ('fopen0 = p_fcond * q_fcond','fopen0 = (p_q ^ p_instances) * (q_q ^ q_instances)')],
        'k_slow.mod':[("n_q' = rate_n_q","n_q' = (n_inf-n_q)/n_tau"),('fopen0 = n_fcond','fopen0 = n_q ^ n_instances')],
        'CaPool.mod':[("concentration' = rate_concentration","concentration' = (iCa/surfaceArea)*rho-(concentration-restingConc)/decayConstant")]}
    diff=[]
    for name,replacements in changes.items():
        baseline=(ROOT/'kernels'/name).read_text();expected=baseline
        for before,after in replacements:
            assert expected.count(before)==1;expected=expected.replace(before,after)
        assert expected==(CANDIDATE/name).read_text()
        copy(CANDIDATE/name,ROOT/'candidate-reference'/name,verified['source_hashes'][name])
        # Changing the new copy is intended; original files and snapshots stay fixed.
        (ROOT/'kernels'/name).write_text(expected)
        diff.extend(difflib.unified_diff(baseline.splitlines(True),expected.splitlines(True),fromfile='parent/'+name,tofile='joint_exp_direct/'+name))
    for name in ['ca_boyle.cpp','k_fast.cpp','k_slow.cpp','CaPool.cpp']:
        copy(CANDIDATE/'arm64'/name,ROOT/'candidate-reference/arm64'/name,verified['generated_hashes'][name])
    (ROOT/'candidate.patch').write_text(''.join(diff))
    # Preserve parent generated code/binary separately; new candidate will compile
    # into a clean directory, never overwrite the only copy of parent evidence.
    shutil.move(ROOT/'kernels/arm64',ROOT/'parent-kernel-generated')
    # Adjust manifest paths after this physical move; original SHA remains visible.
    for key in list(sources):
        if key.startswith('kernels/arm64/'):
            sources[key.replace('kernels/arm64/','parent-kernel-generated/',1)]=sources.pop(key)
    for name in changes:sources['kernels/'+name]=dict(source='candidate-reference/'+name,sha256=sha(ROOT/'kernels'/name))
    write(ROOT/'sources.json',sources)
    inherited=read(PARENT/'protocol.json')
    protocol=dict(date='2026-11-12',created_utc=datetime.now(timezone.utc).isoformat(),candidate='joint_exp_direct',
        objective='Test component improvement transfer to full302/95-muscle/body system, not tune parameters',
        population=dict(neurons=302,muscles=95,chemical=4665,gap_pairs=1192,
            shared_active_C1_scope='284 generic C1 neurons plus 95 muscles; 18 B engineering prototypes retain point-current model and zero old density gmax'),
        frozen_parent_protocol_sha256=sha(ROOT/'parent-evidence/protocol.json'),
        inherited_acceptance= 'Exact numerical values and definitions from 2026-11-09 protocol and byte-pinned analysis',
        acceptance=inherited['acceptance'],definitions=inherited['definitions'],
        duration_s=8,seed=13,feedback=['off','live'],steps_ms=[.025,.0125,.00625],window_s=[4,8],
        nmj_nS=.9,gap_nS=.1,feedback_gain_pA_per_strain=100,feedback_clip_pA=1,exchange_ms=1,recorded_ms=10,
        max_new_full_runs=6,initial_cases=4,
        refinement_rule='After off/live .025/.0125 ms, if ANY acceptance or inherited trace/rhythm validity criterion fails (including incomplete candidate run), run .00625 for BOTH modes. No retries or extra parameters/conditions.',
        reference='Finer adjacent member; also direct .025 vs .00625 when available',
        historical='Six old same-parameter traces copied for labeled model comparison only; none substitutes a new candidate run',
        protected=['all parameters/geometry/initialization code/topology/polarity/BMotorPoint/NMJ/h_fcond/rates/CaPool sync and clipping',
            'body binary/1ms exchange/order of body step then network advance','no extra fcurrent/rates or observer hooks'],
        allowed_changes=list(changes),numerical_edits=9,
        initial_current_note='INITIAL/STATE initial conditions are unchanged; native initialized currents may differ under direct readout and form part of the candidate total model change',
        safety_preflight='Source/generated-code matching, node/connection counts, seed initial states, actual default secondorder=0, zero-time neural/body clocks, no neuron/body advance; if critical mismatch do not start scientific conditions',
        failure_preservation='Frozen driver retains voltage-guard failures. Outer wrapper additionally serializes already-computed traceback-local rows after unexpected exception; no altered loop or retry.',
        comparison_limits=['neural dt refined only, body/exchange 1ms fixed','10ms state output / 1ms feedback; sampled phase resolution retained',
            'Do not attribute whole-model changes to muscle alone','h(cai)/NMJ cache/pool sync remain unresolved','Engineering B prototype is not physiological fit',
            'Propulsion still speed>=10um/s AND displacement>=20um, independently of numerical convergence'],
        source_manifest_sha256=sha(ROOT/'sources.json'),prepare_script_sha256=sha(Path(__file__)),
        runner_sha256=sha(Path(__file__).with_name('joint_run.py')),
        environment=dict(python=sys.version,executable=sys.executable,neuron=neuron.__version__,numpy=numpy.__version__,scipy=scipy.__version__))
    write(ROOT/'protocol.json',protocol);write(ROOT/'protocol-lock.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),locked_before_any_scientific_run=True))
    print(json.dumps(dict(protocol_sha256=sha(ROOT/'protocol.json'),source_files=len(sources),candidate_changes=4,scientific_runs=0)))


if __name__=='__main__':main()
