"""Read-only comparison of B effects with branch-A unperturbed dt differences."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'results/b-motor-feedback-causality-2026-11-09'
A=REPO/'results/b-motor-convergence-2026-11-09'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rms(x):return float(np.sqrt(np.mean(x*x)))
def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}

def main():
    source=A if (A/'summary.json').exists() else ROOT/'numerical-reference'
    out=dict(branch_A_summary_available=(A/'summary.json').exists(),
        numerical_reference_source='live branch A' if source==A else 'B-owned frozen branch A snapshot',
        limitation='A varies neural dt on unperturbed trajectories only, with fixed 1ms exchange. Its differences are diagnostic comparison scales, not certified error bounds for the perturbed branch; B itself has no dt refinement.')
    if (source/'summary.json').exists():
        summary_bytes=(source/'summary.json').read_bytes()
        summary=json.loads(summary_bytes)
        fine=next((d for d in ['0.00625','0.0125'] if (source/f'live_dt{d}/record.json').exists()),None)
        out.update(branch_A_summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),branch_A_new_runs=summary['new_full_runs'],
                   branch_A_convergence_accepted=summary['neural_step_convergence_accepted'],
                   branch_A_complete=summary['new_full_runs']==6 or summary.get('fine_required') is False)
        if fine:
            cases=['live_dt0.025',f'live_dt{fine}'];data=[];sources={}
            for case in cases:
                p=source/case/'trace.npz';r=read(source/case/'record.json')
                assert sha(p)==r['trace_sha256'] and r['complete']
                data.append(load(p));sources[case]=dict(trace_sha256=sha(p),record_sha256=sha(source/case/'record.json'))
                if source==A:
                    target=ROOT/'numerical-reference'/case;target.mkdir(parents=True,exist_ok=True)
                    for name in ['trace.npz','record.json']:shutil.copy2(source/case/name,target/name)
            co,fi=data;names=co['neuron_names'].tolist();ix=[names.index(n) for n in co['B_names']]
            baseline=load(ROOT/'s13_baseline/trace.npz')
            assert all(np.array_equal(co[k],baseline[k]) for k in baseline if np.issubdtype(baseline[k].dtype,np.number))
            effect=read(ROOT/'summary.json')['comparisons']['s13_live_pert__minus__s13_replay_base_pert']
            out.update(coarse_dt_ms=.025,reference_dt_ms=float(fine),source_traces=sources,baseline_numeric_arrays_exactly_equal=True,windows={})
            for window,(lo,hi) in {'during':(2,2.3),'recovery':(2.3,4),'late':(4,8.000001),'post':(2,8.000001)}.items():
                m=(co['time_s']>=lo-1e-8)&(co['time_s']<hi-1e-8)
                f=(co['exchange_time_s']>=lo-1e-8)&(co['exchange_time_s']<hi-1e-8)
                dv=rms(co['neuron_voltage_mV'][m][:,ix]-fi['neuron_voltage_mV'][m][:,ix])
                dc=rms(co['feedback_pA'][f]-fi['feedback_pA'][f])
                ds=abs(float((co['speed'][m]-fi['speed'][m]).mean()*1e6))
                e=effect[window]
                out['windows'][window]=dict(B_voltage_dt_RMS_mV=dv,B_effect_RMS_mV=e['B_voltage_rms_mV'],
                    B_effect_to_dt_difference=e['B_voltage_rms_mV']/dv if dv else None,
                    feedback_dt_RMS_pA=dc,feedback_effect_RMS_pA=e['feedback_rms_pA'],
                    feedback_effect_to_dt_difference=e['feedback_rms_pA']/dc if dc else None,
                    mean_speed_dt_difference_um_s=ds,mean_speed_effect_um_s=e['speed_difference_um_s'],
                    speed_effect_to_dt_difference=abs(e['speed_difference_um_s'])/ds if ds else None)
            key=f'live_0.025_vs_{fine}'
            if key in summary['comparisons']:
                out['A_peak_timing_diagnostic']=summary['comparisons'][key]['details']['B_bidirectional_peak_offset_max_s']
                timing=effect['dense_peak_timing']['per_cell'];offset=out['A_peak_timing_diagnostic']
                out['phase_cell_crosscheck']=[]
                for j,p in enumerate(timing):
                    if p['ordinal_match_valid'] and not p['ambiguous_half_cycle'] and p['late_mean_shift_ms'] is not None:
                        scale=max(offset[j]*1000,10.)
                        out['phase_cell_crosscheck'].append(dict(name=p['name'],late_effect_shift_ms=p['late_mean_shift_ms'],
                            A_max_peak_offset_ms=offset[j]*1000,comparison_floor_ms=10.,
                            effect_to_dt_peak_difference_with_sampling_floor=abs(p['late_mean_shift_ms'])/scale))
            # Preserve the exact A summary examined rather than depending on its future state.
            (ROOT/'branch-A-summary-readonly-snapshot.json').write_bytes(summary_bytes)
            (ROOT/'numerical-reference/summary.json').write_bytes(summary_bytes)
    (ROOT/'numerical-crosscheck.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':main()
