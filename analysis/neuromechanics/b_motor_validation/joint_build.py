"""Compile candidate, verify actual functions, and do a zero-step full setup."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'results/full302-joint-numerics-2026-11-12'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def compact(t):return re.sub(r'\s+','',t)
def function(text,name):
    kind='int states' if name=='states' else 'double _nrn_current'
    m=re.search(r'static '+kind+r'\s*\([^;\n]*\)\s*\{',text)
    if not m:return None
    start=text.index('{',m.start());depth=0
    for i in range(start,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[m.start():i+1]
    raise ValueError('unclosed function')


def main():
    import neuron,numpy as np,scipy
    assert (neuron.__version__,np.__version__,scipy.__version__)==('9.0.2','2.3.5','1.16.3')
    protocol=read(ROOT/'protocol.json');assert sha(ROOT/'protocol.json')==read(ROOT/'protocol-lock.json')['protocol_sha256']
    for name,d in read(ROOT/'sources.json').items():assert sha(ROOT/name)==d['sha256'],name
    assert not list(ROOT.glob('*_dt*/spec.json'))
    assert not (ROOT/'kernels/arm64').exists(), 'Compile once into clean candidate directory'
    write(ROOT/'source-manifest-semantics.json',dict(default_source_root='project',
        local_candidate_sources={f'kernels/{n}':f'candidate-reference/{n}' for n in protocol['allowed_changes']},
        local_candidate_source_root='results/full302-joint-numerics-2026-11-12',
        note='Source manifest bytes unchanged; four candidate source references are explicitly result-root relative'))
    with (ROOT/'build.log').open('w') as log:
        subprocess.run([str(Path(sys.executable).parent/'nrnivmodl')],cwd=ROOT/'kernels',stdout=log,stderr=subprocess.STDOUT,check=True)
    changed={'ca_boyle','k_fast','k_slow','CaPool'};checks={};bodies={}
    for mod in sorted((ROOT/'kernels').glob('*.mod')):
        name=mod.stem;new=(ROOT/'kernels/arm64'/f'{name}.cpp').read_text()
        parent=(ROOT/'parent-kernel-generated'/f'{name}.cpp').read_text()
        reference=(ROOT/'candidate-reference/arm64'/f'{name}.cpp').read_text() if name in changed else parent
        for kind in ['states','current']:
            actual=function(new,kind);expected=function(reference,kind)
            assert (actual is None)==(expected is None),(name,kind)
            if actual is not None:assert compact(actual)==compact(expected),(name,kind,'actual function mismatch')
        if name in changed:
            block=function(new,'states');assert 'exp(' in block
            if name!='CaPool':assert 'rates (' not in function(new,'current')
            if name=='ca_boyle':assert 'h_fcond' in function(new,'current') and 'h_q' not in function(new,'current')
            bodies[name]=dict(states=function(new,'states'),current=function(new,'current'))
        checks[name]=dict(matches='verified 11-10 exp_direct' if name in changed else 'frozen 11-09 parent',passed=True)
    generated=dict(protocol_sha256=sha(ROOT/'protocol.json'),passed=True,mechanism_checks=checks,
        candidate_functions=bodies,binary_sha256=sha(ROOT/'kernels/arm64/libnrnmech.dylib'),
        generated_sha256={p.name:sha(p) for p in (ROOT/'kernels/arm64').glob('*.cpp')},
        untouched_templates={p.name:sha(p) for p in (ROOT/'kernels').glob('*.hoc')},
        build_script_sha256=sha(Path(__file__)),scientific_runs=0)
    write(ROOT/'generated-code-validation.json',generated)
    wrapper=Path(__file__).with_name('joint_run.py')
    assert sha(wrapper)==protocol['runner_sha256']
    spec=importlib.util.spec_from_file_location('joint_wrapper',wrapper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    driver=module.driver_module();a=module.args_for('off',.025)
    plan=read(ROOT/'connection-plan.json');net=driver.C1Network(ROOT/'kernels',plan,dt_ms=.025)
    names,points,clamps,afferents,gaps,initial=driver.configure(net,a);net.reset()
    state=net.state();assert len(names)==len(points)==len(afferents)==18
    assert len(plan['neurons'])==302 and len(plan['muscles'])==95
    assert state['neuron_voltage_mV'].shape==(302,) and state['muscle_calcium_mM'].shape==(95,)
    assert net.h.t==0 and net.h.dt==.025 and net.h.secondorder==0
    with np.load(ROOT/'historical/off_dt0.025/trace.npz') as z:
        assert np.array_equal(state['neuron_voltage_mV'],z['neuron_voltage_mV'][0])
        assert np.array_equal(state['muscle_voltage_mV'],z['muscle_voltage_mV'][0])
        assert np.array_equal(state['muscle_calcium_mM'],z['muscle_calcium_mM'][0])
    for name in names:
        seg=net.cells[name].soma(.5)
        assert all(getattr(seg,m).gmax==0 for m in ['Leak','ca_boyle','k_fast','k_slow'])
    active=[n for n in plan['neurons'] if n not in names]
    assert len(active)==284 and all(net.cells[n].soma(.5).ca_boyle.gmax>0 for n in active)
    assert all(net.cells[n].soma(.5).ca_boyle.gmax>0 for n in plan['muscles'])
    nmjs=0
    for (pre,post),syn in net.chemical_connections.items():
        if pre in names and post in plan['muscles']:
            assert abs(syn.conductance-.0009)<1e-15 and syn.Vth==-30;nmjs+=1
    sums={n:0. for n in names}
    for row in gaps:sums[next(n for n in row['pair'] if n in names)]+=row['conductance_nS']
    assert all(abs(s-.1)<1e-12 for s in sums.values())
    w=driver.load_anatomical_body_wiring();weights,sides,centers=driver.field(w,names)
    assert weights.shape==(18,48) and len(w.muscle_names)==95
    with driver.BoyleMuscleBody('ignored') as body:
        body.reset();assert body.state.time_s==0 and body.state.microsteps==0 and body.dt_s==.001
        assert body.state.side_length.shape==(48,2) and body.state.pose.shape==(49,3)
        assert np.isfinite(body.state.pose).all() and np.isfinite(body.state.side_length).all()
        body_meta=dict(time_s=body.state.time_s,internal_time_s=body.state.internal_time_s,microsteps=body.state.microsteps,dt_s=body.dt_s)
    preflight=dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),actual_neurons=302,actual_muscles=95,actual_chemical=4665,actual_gap_pairs=1192,
        B_points=18,shared_active_C1_neurons=284,shared_active_C1_muscles=95,B_density_gmax_zero=True,
        B_NMJ_edges_checked=nmjs,AVB_total_per_B_nS=sums,B_seed_initial_voltage_mV=initial.tolist(),
        historical_initial_voltage_and_calcium_arrays_identical=True,secondorder=0,neural_time_ms=float(net.h.t),actual_dt_ms=float(net.h.dt),body=body_meta,
        neural_fadvance_calls=0,body_steps=0,scientific_runs=0,
        versions=dict(neuron=neuron.__version__,numpy=np.__version__,scipy=scipy.__version__),python=sys.version)
    write(ROOT/'preflight.json',preflight)
    hashes={}
    for folder in ['runtime','kernels']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:hashes[str(path.relative_to(ROOT))]=sha(path)
    for name in ['selected-parameters.json','connection-plan.json']:hashes[name]=sha(ROOT/name)
    write(ROOT/'execution-lock.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),runner_sha256=sha(wrapper),
        generated_validation_sha256=sha(ROOT/'generated-code-validation.json'),preflight_sha256=sha(ROOT/'preflight.json'),
        runtime_sha256=hashes,max_new_full_runs=6,retries=0,loop_driver_sha256=sha(ROOT/'runtime/driver.py'),
        environment_executable=sys.executable,versions=preflight['versions']))
    print(json.dumps(dict(generated_code_passed=True,zero_step_population_clock_preflight_passed=True,scientific_runs=0)))


if __name__=='__main__':main()
