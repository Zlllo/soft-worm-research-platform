"""Byte-pinned independent NumPy audit plus migration/source checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]/'results/full302-joint-numerics-2026-11-12'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def near(a,b):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-12)
def norm(x):return float(np.sqrt(np.sum(x*x)/x.size))
def transverse(z):
    xy=z['pose'][:,:,:2];r=xy-xy.mean(1,keepdims=True)
    angle=np.arctan2(xy[:,0,1]-xy[:,-1,1],xy[:,0,0]-xy[:,-1,0])
    return (-r[:,:,0]*np.sin(angle[:,None])+r[:,:,1]*np.cos(angle[:,None]))*1e6


def main():
    s=read(ROOT/'summary.json');p=read(ROOT/'protocol.json');lock=read(ROOT/'execution-lock.json')
    assert sha(ROOT/'protocol.json')==s['protocol_sha256']==lock['protocol_sha256']
    assert p['acceptance']==read(ROOT/'parent-evidence/protocol.json')['acceptance']
    assert p['definitions']==read(ROOT/'parent-evidence/protocol.json')['definitions']
    assert sha(ROOT/'runtime/driver.py')==lock['loop_driver_sha256']==read(ROOT/'parent-evidence/inputs.json')['runtime/driver.py']['sha256']
    for name,d in read(ROOT/'sources.json').items():assert sha(ROOT/name)==d['sha256']
    # Adapt only root and the hash index, not the independent numerical formulas.
    inputs={name:dict(sha256=digest) for name,digest in lock['runtime_sha256'].items()}
    if (ROOT/'inputs.json').exists():assert read(ROOT/'inputs.json')==inputs
    else:write(ROOT/'inputs.json',inputs)
    for name,d in inputs.items():assert sha(ROOT/name)==d['sha256']
    script=ROOT/'frozen-analysis/independent.py'
    assert sha(script)==read(ROOT/'sources.json')['frozen-analysis/independent.py']['sha256']
    valid=all(d['valid'] for d in s['metrics'].values())
    if valid:
        spec=importlib.util.spec_from_file_location('joint_frozen_independent_audit',script)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=ROOT;m.main()
        frozen=read(ROOT/'independent-validation.json');assert frozen['independent_numpy_audit_passed']
    else:
        # Missing data is never converted to passing numerical tests.
        assert not s['neural_step_convergence_accepted']
        for name,d in s['metrics'].items():
            r=read(ROOT/name/'record.json')
            if not d['complete']:assert not r['complete'] or (ROOT/name/'failure.json').exists()
        frozen=dict(independent_numpy_audit_passed=False,runs_checked=0,pair_gate_values_checked=0,
            reason='Numerical audit cannot accept incomplete runs or missing inherited rhythms; failures retained')
    reference=read(ROOT/'parent-evidence/summary.json');old_archive=read(ROOT/'parent-evidence/archive-validation.json')
    assert sha(ROOT/'parent-evidence/summary.json')==old_archive['summary_sha256']
    migration_checked=0;model_changes=0;feedback_samples=0
    for key,c in s['numerical_error_migration_ratios'].items():
        for gate,d in c.items():
            a=reference['comparisons'][key]['errors'][gate]['value'];b=s['comparisons'][key]['errors'][gate]['value']
            near(d['old_error'],a);near(d['candidate_error'],b)
            if a>1e-15:near(d['ratio'],b/a)
            else:assert d['ratio'] is None
            migration_checked+=1
    for name,d in s['metrics'].items():
        record=read(ROOT/name/'record.json');provenance=read(ROOT/name/'provenance.json')
        assert provenance['runner_sha256']==lock['runner_sha256'] and not provenance['loop_changed']
        assert provenance['extra_fcurrent_calls']==provenance['extra_rates_calls']==0
        if not d['complete']:continue
        z=dict(np.load(ROOT/name/'trace.npz',allow_pickle=False));old=dict(np.load(ROOT/'historical'/name/'trace.npz',allow_pickle=False))
        preflight=read(ROOT/'preflight.json')
        near(record['B_initial_voltage_mV'],preflight['B_seed_initial_voltage_mV'])
        assert np.array_equal(z['neuron_voltage_mV'][0],old['neuron_voltage_mV'][0])
        assert np.array_equal(z['muscle_calcium_mM'][0],old['muscle_calcium_mM'][0])
        strain=z['side_length']/z['side_length'][0]-1
        sensed=((strain[:,:,1]-strain[:,:,0])/2)@z['feedback_weights'].T*np.where(z['feedback_sides']==0,1.,-1.)
        expected=np.clip(100*sensed,-1,1) if d['feedback']=='live' else np.zeros_like(sensed)
        assert np.max(abs(expected-z['feedback_pA'][::10]))<1e-12;feedback_samples+=expected.size
        ix=[z['neuron_names'].tolist().index(n) for n in z['B_names']];late=np.arange(400,801)
        change=s['historical_model_changes'][name]
        near(norm(z['neuron_voltage_mV'][late][:,ix]-old['neuron_voltage_mV'][late][:,ix]),change['B_voltage_RMS_mV'])
        near(norm(z['neuron_voltage_mV'][late]-old['neuron_voltage_mV'][late]),change['all302_voltage_RMS_mV'])
        near(norm(z['muscle_calcium_mM'][late]-old['muscle_calcium_mM'][late]),change['muscle_calcium_RMS_mM'])
        near(norm(z['drive'][late]-old['drive'][late]),change['muscle_drive_RMS'])
        near(norm(transverse(z)[late]-transverse(old)[late]),change['body_frame_shape_RMS_um'])
        near((z['speed'][late].mean()-old['speed'][late].mean())*1e6,change['speed_mean_change_um_s'])
        new_centroid=z['pose'][:,:,:2].mean(1);old_centroid=old['pose'][:,:,:2].mean(1)
        own_displacement_difference=(np.linalg.norm(new_centroid[-1]-new_centroid[400])-
            np.linalg.norm(old_centroid[-1]-old_centroid[400]))*1e6
        near(own_displacement_difference,change['displacement_change_um'])
        assert d['propulsion_gate_passed']==bool(d['late_speed_um_s']>=10 and d['late_displacement_um']>=20)
        model_changes+=1
    attempts=len(list(ROOT.glob('*_dt*/spec.json')));assert attempts<=6 and attempts==s['new_full_runs']
    if s['fine_required'] is not None:
        required=any(not s['comparisons'][f'{mode}_0.025_vs_0.0125']['passed'] for mode in ['off','live'])
        assert required==s['fine_required']
        if attempts==6:assert required
    result=dict(independent_candidate_audit_passed=valid and frozen['independent_numpy_audit_passed'],
        incomplete_or_rhythm_failures_retained=not valid,runs_checked=frozen['runs_checked'],
        pair_gate_values_checked=frozen['pair_gate_values_checked'],migration_ratio_values_checked=migration_checked,
        historical_same_parameter_model_changes_checked=model_changes,sampled_feedback_values_reconstructed=feedback_samples,
        frozen_independent_script_sha256=sha(script),summary_sha256=sha(ROOT/'summary.json'),
        candidate_convergence_accepted=s['neural_step_convergence_accepted'],acceptance_unchanged=True,
        scientific_conditions=attempts,retries=0,analyzer_imported=False,NEURON_imported=False)
    write(ROOT/'validation.json',result);print(json.dumps(result))


if __name__=='__main__':main()
