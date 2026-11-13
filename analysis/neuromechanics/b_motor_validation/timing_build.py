"""Build six new kernels and statically verify physical/observer separation."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]/'results/c1-current-timing-2026-11-10'
VARIANTS=['euler_cached','exp_cached','euler_direct','exp_direct']
MECHS={'ca_boyle':['e','f'],'k_fast':['p','q'],'k_slow':['n']}
OBS_COMMON={'audit_calls':'','audit_time':'(ms)','audit_voltage':'(mV)','audit_reversal':'(mV)',
    'audit_gmax':'(S/cm2)','audit_fopen':'','audit_gion':'(S/cm2)','audit_current':'(mA/cm2)'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def compact(t):return re.sub(r'\s+','',t)
def function(text,pattern):
    m=re.search(pattern,text);assert m,pattern
    start=text.index('{',m.start());depth=0
    for i in range(start,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[m.start():i+1]
    raise ValueError('Unclosed function')
def state(path):return function(path.read_text(),r'static int states\s*\([^;\n]*\)\s*\{')
def current(path):return function(path.read_text(),r'static double _nrn_current\s*\([^;\n]*\)\s*\{')
def parent(v):return ROOT/'frozen'/('original' if v.startswith('euler') else 'both')
def plain(v):return parent(v) if v.endswith('cached') else ROOT/'proposed-variants'/v


def add_observer(text,mech):
    fields=dict(OBS_COMMON)
    for p in MECHS[mech]:fields.update({f'audit_state_{p}':'',f'audit_cache_{p}':''})
    if mech=='ca_boyle':fields.update(audit_hq='',audit_hfactor='',audit_cai='(mM)')
    text=text.replace('NEURON {','NEURON {\n    RANGE '+', '.join(fields),1)
    text=text.replace('ASSIGNED {','ASSIGNED {\n'+''.join(f'    {name} {unit}\n' for name,unit in fields.items()),1)
    text=text.replace('INITIAL {','INITIAL {\n    audit_calls = 0',1)
    before=['audit_calls = audit_calls + 1','audit_time = t','audit_voltage = v',
        'audit_reversal = '+('eca' if mech=='ca_boyle' else 'ek'),'audit_gmax = gmax']
    for p in MECHS[mech]:before.extend([f'audit_state_{p} = {p}_q',f'audit_cache_{p} = {p}_fcond'])
    if mech=='ca_boyle':before.extend(['audit_hq = h_q','audit_hfactor = h_fcond','audit_cai = cai'])
    needle='    conductanceScale = 1';assert text.count(needle)==1
    text=text.replace(needle,'\n'.join('    '+x for x in before)+'\n'+needle,1)
    name='i__'+mech;lines=text.splitlines(True)
    ix=next(i for i,line in enumerate(lines) if line.lstrip().startswith(name+' ='))
    lines[ix+1:ix+1]=['    audit_fopen = fopen\n','    audit_gion = gion\n',f'    audit_current = -{name}\n']
    return ''.join(lines),fields


def build():
    lock=read(ROOT/'protocol-lock.json');main=read(ROOT/'approvals/main-protocol-review.json');review=read(ROOT/'approvals/independent-protocol-review.json')
    assert sha(ROOT/'protocol.json')==lock['protocol_sha256']==main['protocol_sha256']==review['reviewed_protocol_sha256']
    assert main['status']=='approved_control_first' and review['protocol_acceptable_for_implementation']
    assert not (ROOT/'controls').exists(), 'No compile/source changes after control execution'
    evidence={};(ROOT/'observer-kernels').mkdir(exist_ok=True)
    for v in VARIANTS:
        base=plain(v)
        if v.endswith('direct'):
            with (base/'timing-build.log').open('w') as log:
                subprocess.run([str(Path(sys.executable).parent/'nrnivmodl')],cwd=base,stdout=log,stderr=subprocess.STDOUT,check=True)
        observer=ROOT/'observer-kernels'/v;observer.mkdir(exist_ok=False)
        all_fields={}
        for path in sorted(base.iterdir()):
            if path.suffix not in ['.hoc','.mod']:continue
            text=path.read_text()
            if path.stem in MECHS:text,all_fields[path.stem]=add_observer(text,path.stem)
            (observer/path.name).write_text(text)
        with (observer/'timing-build.log').open('w') as log:
            subprocess.run([str(Path(sys.executable).parent/'nrnivmodl')],cwd=observer,stdout=log,stderr=subprocess.STDOUT,check=True)
        checks={};functions={}
        for mech in ['ca_boyle','k_fast','k_slow','CaPool']:
            cpp=base/'arm64'/f'{mech}.cpp';ocpp=observer/'arm64'/f'{mech}.cpp';pcpp=parent(v)/'arm64'/f'{mech}.cpp'
            assert compact(state(cpp))==compact(state(pcpp))==compact(state(ocpp)),(v,mech,'states')
            if mech in MECHS:
                normal=current(cpp);observed=current(ocpp)
                stripped='\n'.join(line for line in observed.splitlines() if not line.strip().startswith('audit_'))
                assert compact(normal)==compact(stripped),(v,mech,'physical current arithmetic')
                assert observed.index('audit_voltage = v')<observed.index('fopen0 =')<observed.index('audit_current =')
                assert 'rates (' not in normal and 'rates (' not in observed
                assert 'audit_' not in state(ocpp)
                if mech=='ca_boyle':assert 'h_fcond' in normal and 'h_q' not in normal
                if v.endswith('direct'):
                    for p in MECHS[mech]:assert f'pow( {p}_q , {p}_instances )' in normal
                else:
                    for p in MECHS[mech]:assert f'{p}_fcond' in normal
                nrncur=function(ocpp.read_text(),r'static void nrn_cur\s*\([^;\n]*\)\s*\{')
                assert nrncur.index('_v + .001')<nrncur.index('_rhs = _nrn_current')
                functions[mech]=dict(plain_current=normal,observer_current=observed,nrn_cur=nrncur,states=state(cpp))
            checks[mech]=True
        # Untouched NMJ and CaPool MOD source and their physical generated functions.
        for mech in ['neuron_to_neuron_exc_syn','neuron_to_neuron_inh_syn','CaPool']:
            assert sha(base/f'{mech}.mod')==sha(parent(v)/f'{mech}.mod')==sha(observer/f'{mech}.mod')
        for mech in ['neuron_to_neuron_exc_syn','neuron_to_neuron_inh_syn']:
            for extractor in [current,state]:
                assert compact(extractor(base/'arm64'/f'{mech}.cpp'))==compact(extractor(parent(v)/'arm64'/f'{mech}.cpp'))==compact(extractor(observer/'arm64'/f'{mech}.cpp'))
        evidence[v]=dict(base=str(base.relative_to(ROOT)),observer=str(observer.relative_to(ROOT)),checks=checks,
            generated_functions=functions,observer_fields=all_fields,
            base_sha256={str(p.relative_to(base)):sha(p) for p in base.rglob('*') if p.is_file()},
            observer_sha256={str(p.relative_to(observer)):sha(p) for p in observer.rglob('*') if p.is_file()})
        print('verified',v,flush=True)
    write(ROOT/'generated-code-validation.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),
        build_script_sha256=sha(Path(__file__)),variants=evidence,new_kernel_builds=6,new_scientific_runs=0,
        scope='All observer writes are metadata; original physical arithmetic/states and protected NMJ/pool source verified'))


if __name__=='__main__':build()
