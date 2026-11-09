"""Archive-only analysis of the paused B-cell mechanism stage."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


ROOT=Path('results/b-motor-mechanisms-2026-11-07')
CASES=['prototype_off','prototype_live','prototype_gap0','prototype_off8','prototype_live8',
       'calcium0_8','generic_C1_8','nmj9_off8','nmj9_live8']


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    checks={};metrics={}
    plan=read(ROOT/'connection-plan.json');allowed={tuple(pair) for pair in plan['gap_pairs']}
    assert len(plan['neurons'])==302 and len(plan['muscles'])==95 and len(plan['chemical'])==4665
    for name in CASES:
        folder=ROOT/name;d=read(folder/'record.json')
        assert d['complete'] and sha(folder/'trace.npz')==d['trace_sha256']
        with np.load(folder/'trace.npz') as z:
            n=z['neuron_names'].tolist();bn=z['B_names'].tolist();indices=[n.index(x) for x in bn]
            t=z['time_s'];late=t>=d['spec']['duration']/2-1e-8
            assert z['neuron_voltage_mV'].shape==(len(t),302) and z['muscle_voltage_mV'].shape==(len(t),95)
            assert len(bn)==18 and d['gap_pairs']==1192
            assert abs(t[-1]-d['spec']['duration'])<1e-8
            for k in z.files:
                if np.issubdtype(z[k].dtype,np.number):assert np.isfinite(z[k]).all(),(name,k)
            assert z['neuron_voltage_mV'].min()>=-100 and z['neuron_voltage_mV'].max()<=60
            if 'external_drive' in z:assert np.all(z['external_drive']==0)
            assert d['max_gap_pair_current_imbalance_nA']<1e-12
            totals={b:0. for b in bn}
            for pair in d['modified_AVB_gap_pairs']:
                assert tuple(pair['pair']) in allowed
                b=next(x for x in pair['pair'] if x in bn);totals[b]+=pair['conductance_nS']
            assert all(abs(value-d['spec']['gap_nS'])<1e-12 for value in totals.values())
            v=z['neuron_voltage_mV'][late][:,indices];tt=t[late];peaks=[];freq=[]
            for j in range(18):
                pp,_=find_peaks(v[:,j],prominence=3);peaks.append(len(pp))
                freq.append(float(1/np.diff(tt[pp]).mean()) if len(pp)>1 else None)
            dorsal=v[:,[i for i,b in enumerate(bn) if b.startswith('DB')]].mean(1)
            ventral=v[:,[i for i,b in enumerate(bn) if b.startswith('VB')]].mean(1)
            corr=float(np.corrcoef(dorsal,ventral)[0,1]) if min(dorsal.std(),ventral.std())>1e-5 else None
            max_feedback_error=0.
            if d['spec']['feedback']=='live':
                lengths=z['side_length'];strain=(lengths-lengths[0])/lengths[0]
                bending=(strain[:,:,1]-strain[:,:,0])/2
                expected=(bending@z['feedback_weights'].T)*np.where(z['feedback_sides']==0,1.,-1.)
                assert np.max(abs(expected-z['sensed_strain']))<1e-14
                current=np.clip(d['spec']['feedback_gain']*expected,-d['spec']['feedback_clip'],d['spec']['feedback_clip'])
                max_feedback_error=float(np.max(abs(current-z['feedback_pA'][::10])))
                assert max_feedback_error<1e-12
            metrics[name]=dict(duration_s=d['spec']['duration'],B_peak_counts=peaks,B_frequency_Hz=freq,
                B_mean_sd_mV=float(v.std(0).mean()),B_voltage_range_mV=[float(v.min()),float(v.max())],
                dorsal_ventral_voltage_correlation=corr,late_drive_sd=d['late_drive_sd'],maximum_drive=d['maximum_drive'],
                late_speed_um_s=d['late_mean_speed_um_s'],late_displacement_um=d['late_centroid_displacement_um'],
                body_lateral_amplitude_um=float(np.ptp(z['pose'][late,:,1],axis=0).max()*1e6),
                maximum_feedback_pA=d['maximum_feedback_pA'],feedback_reconstruction_error_pA=max_feedback_error)
    checks['all_nine_full302_traces_complete_hashed_and_finite']=True
    checks['population_shapes_and_end_times']=True
    checks['all_neuron_voltages_inside_declared_guard']=True
    checks['all_gap_currents_balanced']=True
    checks['AVB_gap_masks_and_total_conductance_per_B_match_design']=True
    checks['no_external_muscle_waveform_in_full302_trials']=True
    checks['actual_body_strain_feedback_reconstructed']=True
    component=[read(ROOT/f'component-dt{x}/record.json') for x in [.025,.0125]]
    checks['independent_neuron_scipy_voltage_agreement']=all(d['max_voltage_error_mV']<.02 for d in component)
    checks['component_error_decreases_with_half_dt']=component[1]['max_voltage_error_mV']<.6*component[0]['max_voltage_error_mV']
    checks['corrected_nonlinear_capacitance_3pF']=all(abs(d['neuron_capacitance_pF']-3)<1e-12 for d in component)
    transfers={};source_v=[]
    for g in [.09,.3,.9]:
        folder=ROOT/f'nmj-{g}nS';transfers[str(g)]=read(folder/'record.json')
        with np.load(folder/'trace.npz') as z:source_v.append(z['B_voltage_mV'].copy())
    checks['isolated_NMJ_trials_have_identical_presynaptic_waveform']=all(np.array_equal(v,source_v[0]) for v in source_v)
    bif=read(ROOT/'bifurcation/results.json')
    hypothesis=dict(all18_B_oscillate_in_intact_full302=sum(x>=5 for x in metrics['prototype_off8']['B_peak_counts'])==18,
        removing_AVB_gap_eliminates_observed_rhythm=max(metrics['prototype_gap0']['B_peak_counts'])==0,
        removing_active_calcium_eliminates_observed_rhythm=max(metrics['calcium0_8']['B_peak_counts'])==0,
        matched_generic_C1_rhythm_count=sum(x>=5 for x in metrics['generic_C1_8']['B_peak_counts']),
        gap_point_one_fixed_point_unstable=not next(x for x in bif if x['gap_nS']==.1)['fixed_points'][0]['stable'],
        stronger_gap_point_one_two_five_returns_to_rest=next(x for x in bif if x['gap_nS']==.125)['peaks']==0,
        propulsion_gate_passed={n:m['late_speed_um_s']>=10 and m['late_displacement_um']>=20 for n,m in metrics.items() if m['duration_s']==8})
    calibration=[read(ROOT/name/'calibration.json') for name in ['isolated','narrow-gates','lower-calcium']]
    summary=dict(status='stage_paused_at_user_request',checks=checks,full302_runs=len(metrics),metrics=metrics,
        component_comparison=component,isolated_candidates=sum(len(x['results']) for x in calibration),
        isolated_failed_candidates=sum(not r['passed'] for x in calibration for r in x['results']),
        isolated_NMJ=transfers,NMJ_drive_sd_gain_10x_conductance=transfers['0.9']['late_drive_sd']/transfers['0.09']['late_drive_sd'],
        bifurcation=bif,hypothesis_results=hypothesis,
        not_verified=['Exact Xu SI parameter reproduction','Cell-specific biological calibration','Full-body half-step refinement for new B branch',
            'Body perturbation vs frozen feedback replay','Mechanical phase entrainment or feedback propulsion benefit'],
        boundary='Engineering model mechanisms, not biological thermotaxis or measured B-cell physiology')
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(checks_passed=sum(checks.values()),checks_total=len(checks),hypothesis=hypothesis)))
    assert all(checks.values())


if __name__=='__main__':main()
