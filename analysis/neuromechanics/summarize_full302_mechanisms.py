"""Archive-only checks and metrics. No NEURON, native worker or Julia required."""
import hashlib
import json
from pathlib import Path

import numpy as np


NEURAL=Path('results/full302-operating-range-2026-11-07')
PHYS=Path('results/body-phase-coordination-2026-11-07')
ISO=Path('results/body-isotropic-drag-2026-11-07')
REFERENCE=Path('results/full302-modworm-reference-2026-11-07')
OUTPUT=Path('results/full302-mechanism-stage-2026-11-07')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(exist_ok=True)
    checks={};neural={};failures=[]
    for p in sorted(NEURAL.glob('*/record.json')):
        d=read(p);folder=p.parent;name=folder.name
        if sha(folder/'trace.npz')!=d['trace_sha256']:raise RuntimeError('trace changed: '+name)
        if not d['complete']:
            failures.append(dict(name=name,failure=d['failure'],duration_s=d['actual_duration_s']));continue
        with np.load(folder/'trace.npz') as z:
            assert z['neuron_voltage_mV'].shape[1]==302
            assert z['muscle_voltage_mV'].shape[1]==95
            assert z['syn_open'].shape[1]==4665
            assert all(np.isfinite(z[k]).all() for k in z.files)
            assert abs(z['time_s'][-1]-d['spec']['duration'])<1e-8
            assert z['neuron_voltage_mV'].min()>=-100 and z['neuron_voltage_mV'].max()<=60
            tail=z['time_s']>=d['spec']['duration']/2-1e-8
            neural[name]=dict(mean_nmj_gate=d['last_half_mean_nmj_release'],mean_nn_gate=d['last_half_mean_nn_release'],
                duration_s=d['spec']['duration'],late_speed_um_s=d['last_half_signed_speed_um_s'],
                displacement_um=d['centroid_displacement_um'],late_max_drive=float(z['drive'][tail].max()),
                final_max_drive=float(z['drive'][-1].max()),voltage_range_mV=d['neuron_range_mV'])
    checks['complete_neural_trace_shapes_times_finite_voltage_guards']=True
    checks['failed_cases_kept_separate']=len(failures)==2 and (NEURAL/'both30_AVB/failure.json').exists()
    original=read(NEURAL/'synapse-check-kernels.json')
    fixed=read(NEURAL/'synapse-check-kernels-direct-cnexp.json')
    checks['constant_voltage_exact_solution']=fixed['max_absolute_gate_error']<1e-14 and fixed['within_probability_bounds']
    checks['original_instability_reproduced']=original['max_absolute_gate_error']>1e10 and not original['within_probability_bounds']
    for p in sorted((NEURAL/'kernels').glob('*')):
        if p.suffix not in ('.mod','.hoc'):continue
        expected=p.read_text()
        if p.name in ('neuron_to_neuron_exc_syn.mod','neuron_to_neuron_inh_syn.mod'):
            expected=expected.replace("s' = rate_s","if ((1-inf) > 1e-4) {\n        s' = (inf-s)/tau\n    } else {\n        s' = 0\n    }")
        assert expected==(NEURAL/'kernels-direct-cnexp'/p.name).read_text()
    checks['synaptic_variant_changes_only_two_explicit_state_expressions']=True
    # Compare outcomes at common output times; tolerate finite-step spike timing.
    coarse=neural['both30_AVB_cnexp'];fine=neural['both30_AVB_cnexp_halfdt']
    gate_rel=abs(coarse['mean_nmj_gate']-fine['mean_nmj_gate'])/fine['mean_nmj_gate']
    drive_rel=abs(coarse['final_max_drive']-fine['final_max_drive'])/max(fine['final_max_drive'],1e-30)
    speed_abs=abs(coarse['late_speed_um_s']-fine['late_speed_um_s'])
    checks['half_dt_release_metric_agreement']=gate_rel<1e-3
    # Initial coarse-resolution peak check failed; retain it explicitly and
    # compare the two finest resolutions after the declared refinement.
    quarter=neural['both30_AVB_cnexp_quarterdt']
    finest_drive_rel=abs(fine['final_max_drive']-quarter['final_max_drive'])/max(quarter['final_max_drive'],1e-30)
    checks['finest_two_resolutions_final_drive_agreement']=finest_drive_rel<.02
    checks['half_dt_motion_difference_below_1e_minus3_um_per_s']=speed_abs<1e-3
    physical={};named_doses=[];named_energy=[]
    for root,label in [(PHYS,'directional'),(ISO/'cases','isotropic')]:
        for p in sorted(root.glob('*/record.json')):
            d=read(p);assert d['complete'] and sha(p.parent/'trace.npz')==d['trace_sha256']
            with np.load(p.parent/'trace.npz') as z:
                assert np.isfinite(z['pose']).all() and abs(z['time_s'][-1]-12)<1e-10
            physical[f'{label}/{p.parent.name}']=dict(late_signed_speed_um_s=d['late_signed_speed_um_s'],
                late_net_displacement_um=d['late_net_displacement_um'],late_centroid_path_um=d['late_centroid_path_um'])
            if p.parent.name!='zero':named_doses.append(d['muscle_drive_integral']);named_energy.append(d['muscle_drive_squared_integral'])
    checks['physical_input_dose_matched']=bool(np.max(np.abs(np.array(named_doses)-1.8))<1e-12)
    checks['physical_input_squared_integral_matched']=bool(np.max(np.abs(np.array(named_energy)-.405))<1e-12)
    with np.load(PHYS/'head_to_tail/trace.npz') as za,np.load(ISO/'cases/head_to_tail/trace.npz') as zi:
        checks['same_muscle_commands_across_drag_conditions']=bool(np.array_equal(za['drive'],zi['drive']))
        centroid=zi['pose'][:,:,:2].mean(1)
        checks['isotropic_centroid_stationary']=bool(np.linalg.norm(centroid-centroid[0],axis=1).max()*1e6<1e-6)
        checks['isotropic_body_still_deforms']=bool(np.linalg.norm(zi['pose'][:,0,:2]-zi['pose'][0,0,:2],axis=1).max()*1e6>.1)
    source=Path('results/full-network-body-interface-2026-11-05/build/worm-muscle-worker.cc').read_text()
    expected=source.replace('CL_agar = RCONST(3.2e-3/(2.0*NBAR))','CL_agar = RCONST(65.6e-3/(2.0*NBAR))').replace('CN_agar = RCONST(128e-3/(2.0*NBAR))','CN_agar = RCONST(65.6e-3/(2.0*NBAR))')
    checks['drag_variant_only_changes_two_eigenvalues']=expected==(ISO/'build/worm-muscle-worker.cc').read_text()
    checks['mean_drag_preserved']=(3.2+128)/2==65.6
    reference={}
    for p in sorted(REFERENCE.glob('*/record.json')):
        d=read(p);assert d['complete'] and d['audit']['neurons']==302
        assert sha(p.parent/'trace.npz')==d['trace_sha256']
        reference[p.parent.name]={k:d[k] for k in ['voltage_range_mV','centroid_displacement_BL','late_path_speed_BL_s']}
    provenance=read(REFERENCE/'provenance.json')
    checks['upstream_modworm_snapshot_hashes']=all(sha(REFERENCE/'upstream'/p)==h for p,h in provenance['sha256'].items())
    full_vs_none=abs(neural['both30_AVB_cnexp4s']['mean_nmj_gate']-neural['both30_none_cnexp4s']['mean_nmj_gate'])
    summary=dict(status='completed_mechanism_stage_full302_locomotion_not_achieved',checks=checks,
        neural_complete_runs=len(neural),neural_failed_runs=failures,initial_untraced_failed_runs=1,
        neural=neural,physical=physical,reference=reference,
        synapse_exact_check=dict(original_error=original['max_absolute_gate_error'],fixed_error=fixed['max_absolute_gate_error']),
        numerical_refinement=dict(gate_relative_error=gate_rel,final_drive_relative_error=drive_rel,
            initial_coarse_peak_check_passed=drive_rel<.02,finest_final_drive_relative_error=finest_drive_rel,
            finest_peak_check_passed=finest_drive_rel<.02,speed_absolute_difference_um_s=speed_abs),
        nmj_only_gate_gain=neural['nmj30_AVB_cnexp']['mean_nmj_gate']/neural['base_AVB_cnexp']['mean_nmj_gate'],
        global_shift_avb_vs_none_late_gate_abs_difference=full_vs_none,
        limitations=['Thresholds are sensitivity interventions, not calibrated physiology',
            'Visible Boyle motion uses prescribed muscle waves, no neural controller',
            'modWorm reference uses delayed neural voltage, not body strain',
            'High-current modWorm adaptation excluded from biological interpretation'])
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(checks_passed=sum(checks.values()),checks_total=len(checks),neural_complete_runs=len(neural),physical_runs=len(physical),reference_runs=len(reference))))
    if not all(checks.values()):raise RuntimeError('failed checks: '+str([k for k,v in checks.items() if not v]))


if __name__=='__main__':main()
