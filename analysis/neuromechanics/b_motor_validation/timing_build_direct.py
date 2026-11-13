"""V2: compile two direct kernels only; inspect generated executable bodies."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]/'results/c1-current-timing-2026-11-10'
EXPECTED='8b9d11bda03f952c96d64d5b2b58f92496ecdcc008e5c471a51463664320f12a'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def compact(t):return re.sub(r'\s+','',t)
def body(text,kind):
    pattern=r'static '+('int states' if kind=='states' else 'double _nrn_current')+r'\s*\([^;\n]*\)\s*\{'
    m=re.search(pattern,text);assert m,pattern
    start=text.index('{',m.start());depth=0
    for i in range(start,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[m.start():i+1]
    raise ValueError('unclosed function')


def main():
    assert sha(ROOT/'protocol.json')==EXPECTED==read(ROOT/'protocol-lock.json')['protocol_sha256']
    approval=read(ROOT/'approvals/main-protocol-review-v2.json');assert approval['protocol_sha256']==EXPECTED and approval['small_controls_cap']==12
    assert not (ROOT/'controls').exists()
    for name,digest in read(ROOT/'proposal-source-hashes.json').items():assert sha(ROOT/name)==digest
    changes={
        'ca_boyle':('fopen0 = e_fcond * f_fcond * h_fcond','fopen0 = (e_q ^ e_instances) * (f_q ^ f_instances) * h_fcond',['e','f']),
        'k_fast':('fopen0 = p_fcond * q_fcond','fopen0 = (p_q ^ p_instances) * (q_q ^ q_instances)',['p','q']),
        'k_slow':('fopen0 = n_fcond','fopen0 = n_q ^ n_instances',['n'])}
    evidence={}
    for parent,child in [('original','euler_direct'),('both','exp_direct')]:
        source=ROOT/'frozen'/parent;target=ROOT/'proposed-variants'/child
        for path in source.iterdir():
            if path.suffix not in ['.mod','.hoc']:continue
            expected=path.read_text()
            if path.stem in changes:
                before,after,_=changes[path.stem];expected=expected.replace(before,after)
            assert (target/path.name).read_text()==expected
        with (target/'v2-build.log').open('w') as log:
            subprocess.run([str(Path(sys.executable).parent/'nrnivmodl')],cwd=target,stdout=log,stderr=subprocess.STDOUT,check=True)
        functions={}
        for mech in ['ca_boyle','k_fast','k_slow','CaPool','neuron_to_neuron_exc_syn','neuron_to_neuron_inh_syn']:
            old=(source/'arm64'/f'{mech}.cpp').read_text();new=(target/'arm64'/f'{mech}.cpp').read_text()
            assert compact(body(old,'states'))==compact(body(new,'states')),(child,mech,'states changed')
            original=body(old,'current');direct=body(new,'current')
            if mech in changes:
                assert 'rates (' not in direct and 'audit_' not in direct
                for p in changes[mech][2]:
                    assert f'pow( {p}_q , {p}_instances )' in direct
                    assert f'{p}_fcond' not in direct
                if mech=='ca_boyle':assert 'h_fcond' in direct and 'h_q' not in direct
                old_without= re.sub(r'fopen0\s*=.*?;', 'fopen0=READOUT;',original)
                new_without= re.sub(r'fopen0\s*=.*?;', 'fopen0=READOUT;',direct)
                assert compact(old_without)==compact(new_without)
                functions[mech]=dict(parent_current=original,direct_current=direct,states=body(new,'states'))
            else:assert compact(original)==compact(direct)
        # Last current evaluation in the engine's generated per-mechanism wrapper.
        for mech in changes:
            text=(target/'arm64'/f'{mech}.cpp').read_text()
            assert text.index('_nrn_current(_threadargscomma_ _v + .001)')<text.index('_rhs = _nrn_current(_threadargscomma_ _v)')
        evidence[child]=dict(parent=parent,functions=functions,states_preserved=True,only_three_current_readouts_changed=True,
            NMJ_and_pool_current_and_states_preserved=True,
            source_hashes={p.name:sha(p) for p in target.iterdir() if p.suffix in ['.mod','.hoc']},
            generated_hashes={p.name:sha(p) for p in (target/'arm64').glob('*.cpp')},binary_sha256=sha(target/'arm64/libnrnmech.dylib'))
        print('compiled and verified',child,flush=True)
    output=dict(protocol_sha256=EXPECTED,passed=True,build_script_sha256=sha(Path(__file__)),
        new_kernel_builds=2,observer_builds=0,new_scientific_runs=0,variants=evidence)
    (ROOT/'generated-code-validation-v2.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
