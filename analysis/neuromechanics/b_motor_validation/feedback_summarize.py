"""Recompute branch-B checks and effects exclusively from saved raw arrays."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO/'results/b-motor-feedback-causality-2026-11-09'
CASES = ['s13_baseline', 's13_off_baseline', 's13_live_pert', 's13_replay_base_pert', 's13_off_pert',
         's13_replay_pert', 's29_baseline', 's29_live_pert', 's29_replay_base_pert', 's29_replay_pert']
WINDOWS = {'during':[2,2.3], 'recovery':[2.3,4], 'late':[4,8.000001], 'post':[2,8.000001]}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def rms(x): return float(np.sqrt(np.mean(x*x)))
def load(p):
    with np.load(p) as z: return {k:z[k].copy() for k in z.files}
def mask(t, a, b): return (t >= a-1e-8) & (t < b-1e-8)
def first_change(x, y, dt, atol=0):
    d=np.abs(x-y).reshape(len(x),-1).max(1)
    changed=np.flatnonzero(d>atol)
    return float(changed[0]*dt) if len(changed) else None

def phase(a, b, names):
    """a versus b; signed positive means a's peak is later (delay)."""
    result=[]
    ia=[a['neuron_names'].tolist().index(n) for n in names]
    ib=[b['neuron_names'].tolist().index(n) for n in names]
    for j,n in enumerate(names):
        ta,tb=a['time_s'],b['time_s']
        va,vb=a['neuron_voltage_mV'][:,ia[j]],b['neuron_voltage_mV'][:,ib[j]]
        sampling_ms=float(np.median(np.diff(ta))*1000)
        distance=round(250/sampling_ms)
        pa=find_peaks(va, prominence=3, distance=distance)[0]
        pb=find_peaks(vb, prominence=3, distance=distance)[0]
        aa=ta[pa][ta[pa]>2+1e-8]; bb=tb[pb][tb[pb]>2+1e-8]
        matched=len(aa)==len(bb) and len(aa)>=2
        r=dict(name=n, peak_times_a_s=aa.tolist(), peak_times_b_s=bb.tolist(), ordinal_match_valid=matched,
               late_amplitude_a_mV=float(np.ptp(va[ta>=4-1e-8])), late_amplitude_b_mV=float(np.ptp(vb[tb>=4-1e-8])))
        r['next_peak_latency_difference_ms']=float((aa[0]-bb[0])*1000) if len(aa) and len(bb) else None
        if matched:
            shifts=aa-bb
            period=np.median(np.diff(bb))
            r.update(peak_shift_ms=(shifts*1000).tolist(), peak_shift_cycles=(shifts/period).tolist(),
                     first_peak_shift_ms=float(shifts[0]*1000), late_mean_shift_ms=float(shifts[bb>=4].mean()*1000) if np.any(bb>=4) else None,
                     ambiguous_half_cycle=bool(np.any(abs(shifts)>.5*period)))
        result.append(r)
    valid=[r for r in result if r['ordinal_match_valid'] and not r['ambiguous_half_cycle']]
    return dict(per_cell=result, valid_matched_cells=len(valid), observation_resolution_ms=sampling_ms,
                median_first_peak_shift_ms=float(np.median([r['first_peak_shift_ms'] for r in valid])) if valid else None,
                median_late_peak_shift_ms=float(np.median([r['late_mean_shift_ms'] for r in valid if r['late_mean_shift_ms'] is not None])) if valid else None,
                max_abs_peak_shift_ms=max((max(abs(x) for x in r['peak_shift_ms']) for r in valid),default=None))

def compare(a,b):
    names=a['B_names'].tolist(); ix=[a['neuron_names'].tolist().index(n) for n in names]
    out={}
    for name,(start,end) in WINDOWS.items():
        m=mask(a['time_s'],start,end); f=mask(a['exchange_time_s'],start,end)
        dv=a['neuron_voltage_mV'][m][:,ix]-b['neuron_voltage_mV'][m][:,ix]
        dl=(a['side_length'][m]-b['side_length'][m])/b['side_length'][0]
        dc=a['feedback_pA'][f]-b['feedback_pA'][f]
        dp=a['pose'][m,:,:2]-b['pose'][m,:,:2]
        out[name]=dict(B_voltage_rms_mV=rms(dv),B_voltage_max_mV=float(abs(dv).max()),
            feedback_rms_pA=rms(dc),feedback_max_pA=float(abs(dc).max()),
            side_strain_difference_rms=rms(dl),pose_rms_um=rms(dp)*1e6,
            speed_difference_um_s=float((a['speed'][m]-b['speed'][m]).mean()*1e6))
    out['peak_timing']=phase(a,b,names)
    return out

def main():
    checks={}; records={}; traces={}; audits={}; metrics={}; inputs={}
    frozen=read(ROOT/'source-verification.json')
    oldprefix='results/b-motor-mechanisms-2026-11-07/'
    for original,digest in frozen['verified_archive_payload'].items():
        if original.startswith(oldprefix):
            relative=original[len(oldprefix):]
            relative=relative.replace('nmj9_live8/','s13_baseline/').replace('nmj9_off8/','s13_off_baseline/')
        elif original=='analysis/neuromechanics/run_full302_b_motor.py':
            relative='runtime/analysis/frozen_driver.py'
        else:
            relative='body/'+Path(original).name
        assert sha(ROOT/relative)==digest, relative
    for n in CASES:
        folder=ROOT/n; r=read(folder/'record.json'); z=load(folder/'trace.npz')
        assert r['complete'] and sha(folder/'trace.npz')==r['trace_sha256']
        assert r['neurons']==302 and r['muscles']==95 and r['chemical_connections']==4665 and r['gap_pairs']==1192
        assert z['neuron_voltage_mV'].shape==(801,302) and z['muscle_calcium_mM'].shape==(801,95)
        assert np.allclose(z['time_s'],np.arange(801)*.01,atol=1e-9,rtol=0)
        assert np.array_equal(z['exchange_time_s'],np.arange(8001)*.001)
        for k,v in z.items():
            if np.issubdtype(v.dtype,np.number): assert np.isfinite(v).all(),(n,k)
        expected=dict(model='prototype',duration=8.,dt_ms=.025,gap_nS=.1,nmj_nS=.9,feedback_gain=100,
                      feedback_clip=1,avb_on=True,calcium_scale=1,seed=29 if n.startswith('s29') else 13)
        assert all(r['spec'][k]==v for k,v in expected.items())
        if n in ['s13_baseline','s13_off_baseline']:
            source='nmj9_live8' if n=='s13_baseline' else 'nmj9_off8'
            key='results/b-motor-mechanisms-2026-11-07/'+source+'/trace.npz'
            assert r['trace_sha256']==frozen['verified_archive_payload'][key]
        else:
            assert r['protocol_sha256']==sha(ROOT/'protocol.json')
            assert r['driver_sha256']==sha(REPO/'analysis/neuromechanics/b_motor_validation/feedback_run.py')
            assert sha(folder/'exchange-audit.npz')==r['exchange_audit_sha256']
            q=load(folder/'exchange-audit.npz');audits[n]=q
            for k,v in q.items():
                assert np.isfinite(v).all(),(n,k)
            assert np.max(abs(q['body_time_s']-z['exchange_time_s']))<1e-9
            assert np.max(abs(q['neural_time_s']-z['exchange_time_s']))<1e-9
            assert np.max(q['body_internal_time_s']-q['body_time_s'])<=1e-10
            external=np.zeros((8001,48,2))
            if r['spec']['perturb']: external[2000:2300,:8,0]=.2
            assert np.array_equal(q['external_drive'],external)
            # Command comparison separates the identical additive intervention from neural drive.
            assert np.array_equal(q['command'][::10],np.clip(z['drive']+external[::10],-1,1))
            assert np.max(abs(q['command']))<1 # no clipping changes the .2 intervention
            st=(q['side_length']-q['side_length'][0])/q['side_length'][0]
            sensed=((st[:,:,1]-st[:,:,0])/2)@z['feedback_weights'].T*np.where(z['feedback_sides']==0,1.,-1.)
            candidate=np.clip(100*sensed,-1,1)
            assert np.max(abs(candidate-q['candidate_pA']))<1e-12
            if r['spec']['replay']:
                src=load(ROOT/r['spec']['replay']/'trace.npz')
                assert np.array_equal(z['feedback_pA'],src['feedback_pA'])
                assert np.array_equal(q['replay_index'],np.arange(8001))
                assert r['replay_provenance']['trace_sha256']==sha(ROOT/r['spec']['replay']/'trace.npz')
            elif r['spec']['feedback']=='live':
                assert np.max(abs(z['feedback_pA']-candidate))<1e-12
            else: assert np.all(z['feedback_pA']==0)
        lengths=z['side_length']; st=(lengths-lengths[0])/lengths[0]
        sensed=((st[:,:,1]-st[:,:,0])/2)@z['feedback_weights'].T*np.where(z['feedback_sides']==0,1.,-1.)
        assert np.max(abs(sensed-z['sensed_strain']))<1e-14
        if not r['spec']['replay'] and r['spec']['feedback']=='live':
            assert np.max(abs(z['feedback_pA'][::10]-np.clip(100*sensed,-1,1)))<1e-12
        records[n]=r;traces[n]=z
        inputs[n]=dict(trace_sha256=sha(folder/'trace.npz'), record_sha256=sha(folder/'record.json'))
        t=z['time_s']; late=t>=4-1e-8; bidx=[z['neuron_names'].tolist().index(b) for b in z['B_names']]
        v=z['neuron_voltage_mV'][late][:,bidx]; tt=t[late]; freqs=[]; counts=[]
        for j in range(18):
            p=find_peaks(v[:,j],prominence=3,distance=25)[0];counts.append(len(p))
            freqs.append(float(1/np.diff(tt[p]).mean()) if len(p)>1 else None)
        displacement=float(np.linalg.norm(z['pose'][-1,:,:2].mean(0)-z['pose'][np.flatnonzero(late)[0],:,:2].mean(0))*1e6)
        speed=float(z['speed'][late].mean()*1e6)
        metrics[n]=dict(late_mean_signed_speed_um_s=speed,late_centroid_displacement_um=displacement,
            late_B_frequency_Hz=freqs,late_B_peak_count=counts,late_B_mean_voltage_sd_mV=float(v.std(0).mean()),
            late_B_mean_peak_to_peak_mV=float(np.ptp(v,axis=0).mean()), maximum_feedback_pA=float(abs(z['feedback_pA']).max()),
            late_drive_sd=float(z['drive'][late].std(0).mean()),late_calcium_sd_mM=float(z['muscle_calcium_mM'][late].std(0).mean()),
            late_body_lateral_span_um=float(np.ptp(z['pose'][late,:,1],axis=0).max()*1e6),
            propulsion_threshold_passed=speed>=10 and displacement>=20)
    checks.update(all_ten_traces_complete_hashed_finite_and_population_valid=True, frozen_parameters_and_clocks_valid=True,
                  sealed_seed13_baseline_trace_and_runtime_hashes_valid=True, actual_body_sensory_map_recomputed=True,
                  exchange_clocks_causal_no_body_lookahead=True, replay_sources_and_all_8001_indices_valid=True,
                  external_perturbation_exact_300_ticks_no_command_clipping=True)
    pre_details={}
    state_keys=['neuron_voltage_mV','muscle_voltage_mV','muscle_calcium_mM','pose','drive','speed','body_muscle','side_length','sensed_strain','B_gate']
    for seed in [13,29]:
        base=traces[f's{seed}_baseline']; pre=base['time_s']<=2+1e-8
        for suffix in ['live_pert','replay_base_pert','replay_pert']:
            n=f's{seed}_{suffix}'; z=traces[n]
            exact={k:bool(np.array_equal(z[k][pre],base[k][pre])) for k in state_keys}
            assert all(exact.values()),(n,exact)
            assert np.array_equal(z['feedback_pA'][:2001],base['feedback_pA'][:2001])
            pre_details[n]=exact
        fine_live=audits[f's{seed}_live_pert'];fine_frozen=audits[f's{seed}_replay_base_pert']
        assert all(np.array_equal(fine_live[k][:2001],fine_frozen[k][:2001])
                   for k in fine_live if k!='replay_index')
        live=traces[f's{seed}_live_pert']; replay=traces[f's{seed}_replay_pert']
        exact={k:bool(np.array_equal(live[k],replay[k])) for k in live}
        assert all(exact.values()),exact
        assert all(np.array_equal(audits[f's{seed}_live_pert'][k],audits[f's{seed}_replay_pert'][k])
                   for k in audits[f's{seed}_live_pert'] if k!='replay_index')
    offbase=traces['s13_off_baseline']; pre=offbase['time_s']<=2+1e-8
    assert all(np.array_equal(offbase[k][pre],traces['s13_off_pert'][k][pre]) for k in state_keys)
    checks['pre_intervention_states_bitwise_identical_to_matched_baseline']=True
    checks['perturbed_source_replay_bitwise_reproduces_all_live_neural_and_body_states_both_seeds']=True
    differences={};onsets={}
    pairs=[('s13_live_pert','s13_baseline'),('s13_replay_base_pert','s13_baseline'),('s13_off_pert','s13_off_baseline'),
           ('s13_live_pert','s13_replay_base_pert'),('s13_live_pert','s13_off_pert'),
           ('s29_live_pert','s29_baseline'),('s29_replay_base_pert','s29_baseline'),('s29_live_pert','s29_replay_base_pert')]
    for an,bn in pairs:
        differences[an+'__minus__'+bn]=compare(traces[an],traces[bn])
    for seed in [13,29]:
        live=audits[f's{seed}_live_pert']; replay=audits[f's{seed}_replay_base_pert']
        zi=traces[f's{seed}_live_pert']; zr=traces[f's{seed}_replay_base_pert']
        onsets[str(seed)]=dict(body_length_s=first_change(live['side_length'],replay['side_length'],.001),
            applied_feedback_s=first_change(zi['feedback_pA'],zr['feedback_pA'],.001),
            B_voltage_s=first_change(live['B_voltage_mV'],replay['B_voltage_mV'],.001),
            candidate_vs_frozen_s=first_change(live['candidate_pA'],zr['feedback_pA'],.001))
        onsets[str(seed)]['body_length_comparison']='live versus same-perturbation frozen-current replay; this is feedback consequence, not initial external deformation onset'
        base_for_onset=traces[f's{seed}_baseline']
        onsets[str(seed)]['perturbed_body_vs_unperturbed_first_saved_s']=first_change(zi['side_length'],base_for_onset['side_length'],.01)
        # Live versus frozen replay has exactly equal neural history; causally the current divergence precedes B voltage.
        c=onsets[str(seed)]['applied_feedback_s'];v=onsets[str(seed)]['B_voltage_s']
        assert c>=2.001-1e-12 and v>=c+.001-1e-12
        # Physical intervention under frozen current cannot alter neural state in this model.
        base=traces[f's{seed}_baseline'];rp=traces[f's{seed}_replay_base_pert']
        assert np.array_equal(base['neuron_voltage_mV'],rp['neuron_voltage_mV'])
        assert np.array_equal(base['muscle_calcium_mM'],rp['muscle_calcium_mM'])
        # The only body->network input is the 18 afferent currents. Exact fixed-current
        # shared-time identity permits this dense neural reference without another run.
        dense_ref=audits[f's{seed}_replay_base_pert']
        assert np.array_equal(dense_ref['B_voltage_mV'][::10],base['neuron_voltage_mV'][:,
            [base['neuron_names'].tolist().index(n) for n in base['B_names']]])
        names=base['B_names']
        def dense_trace(q):
            return dict(time_s=np.arange(8001)*.001, neuron_voltage_mV=q['B_voltage_mV'],
                        neuron_names=names, B_names=names)
        dense_phase=phase(dense_trace(live),dense_trace(dense_ref),names.tolist())
        dense_phase['reference']='perturbed body with frozen unperturbed current; exact neural equivalence to baseline verified at all shared times; frozen driver has no other body->network route'
        for key in [f's{seed}_live_pert__minus__s{seed}_baseline', f's{seed}_live_pert__minus__s{seed}_replay_base_pert']:
            differences[key]['dense_peak_timing']=dense_phase
        f=zi['exchange_time_s']>=2
        differences[f's{seed}_live_pert__minus__s{seed}_replay_base_pert']['dense_post_B_rms_mV']=rms(live['B_voltage_mV'][f]-dense_ref['B_voltage_mV'][f])
    checks['current_divergence_precedes_neural_response']=True
    checks['frozen_baseline_current_blocks_all_perturbation_to_neural_effects']=True
    # Off is a separately matched arm, rather than a misleading all-arms pre-equality assertion.
    assert np.array_equal(offbase['neuron_voltage_mV'],traces['s13_off_pert']['neuron_voltage_mV'])
    checks['off_feedback_blocks_perturbation_to_neural_effects']=True
    assert sum('exchange_audit_sha256' in r for r in records.values())==8
    output=dict(protocol_sha256=sha(ROOT/'protocol.json'),new_full_runs=8,reused_old_baselines=2,
        checks=checks,raw_inputs=inputs,pre_intervention_details=pre_details,causal_onsets=onsets,
        metrics=metrics,comparisons=differences,
        interpretation=dict(body_to_current_to_B_causal_path_verified=True,exact_replay_implementation_verified=True,
            single_intervention_peak_timing_only=True,full_phase_response_curve=False,entrainment_demonstrated=False,
            autonomous_propulsion_demonstrated=False,engineering_prototype_not_physiological_fit=True),
        numerical_comparison='See numerical-crosscheck.json; generated separately if branch A is available.')
    (ROOT/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(checks=len(checks),passed=sum(checks.values()),causal_onsets=onsets)))

if __name__=='__main__':main()
