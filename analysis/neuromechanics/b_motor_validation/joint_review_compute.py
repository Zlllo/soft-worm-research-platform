"""Independent saved-data review; never constructs a NEURON/body runtime.

All acceptance definitions come from the frozen 2026-11-09 protocol. No A
analysis helper is imported. Run with --parent for a historical cross-check.
"""
import argparse
import hashlib
import json
import importlib.metadata
import os
from pathlib import Path
import sys
import numpy as np
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/full302-joint-numerics-review-2026-11-12'
A = REPO / 'results/full302-joint-numerics-2026-11-12'
OLD = REPO / 'results/b-motor-convergence-2026-11-09'
STEPS = (.025, .0125, .00625)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def save(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def rms(x, axis=None): return np.sqrt(np.mean(np.asarray(x)**2, axis=axis))
def name(mode, dt): return f'{mode}_dt{dt:g}'
def check(condition, message):
    if not condition: raise AssertionError(message)


def load_run(root, run):
    directory = root / run
    record = read(directory/'record.json')
    check(record['complete'], f'{run}: incomplete scientific run')
    trace_sha = digest(directory/'trace.npz')
    check(trace_sha == record['trace_sha256'], f'{run}: trace SHA')
    with np.load(directory/'trace.npz', allow_pickle=False) as raw:
        z = {k:raw[k] for k in raw.files}
    spec = record['spec']
    check(read(directory/'spec.json')==spec, f'{run}: spec/record agreement')
    check(run==name(spec['feedback'],spec['dt_ms']), f'{run}: directory/condition mapping')
    required = dict(model='prototype', replay=None, gap_nS=.1, feedback_gain=100,
        feedback_clip=1, duration=8., seed=13, avb_on=True, calcium_scale=1,
        perturb=False, nmj_nS=.9)
    check(all(spec.get(k)==v for k,v in required.items()), f'{run}: frozen parameters')
    check(spec['dt_ms'] in STEPS and spec['feedback'] in ('off','live'), f'{run}: dt/mode')
    plan = read(root/'connection-plan.json')
    check(z['neuron_names'].tolist()==plan['neurons'], f'{run}: neuron ordering')
    bnames = z['B_names'].tolist()
    expected_b = [n for n in plan['neurons'] if n.startswith(('DB','VB')) and n[2:].isdigit()]
    check(bnames==expected_b and len(bnames)==18, f'{run}: B population/order')
    check((record['neurons'],record['muscles'],record['chemical_connections'],record['gap_pairs']) ==
        (302,95,4665,1192), f'{run}: topology counts')
    for k,v in z.items():
        if np.issubdtype(v.dtype,np.number): check(np.isfinite(v).all(), f'{run}: nonfinite {k}')
    shapes = dict(neuron_voltage_mV=(801,302), muscle_voltage_mV=(801,95),
        muscle_calcium_mM=(801,95), pose=(801,49,3), drive=(801,48,2),
        side_length=(801,48,2), feedback_pA=(8001,18), sensed_strain=(801,18))
    # Body pose can contain additional coordinates in historical serialization.
    shapes.pop('pose')
    for k,s in shapes.items(): check(z[k].shape==s, f'{run}: {k} shape {z[k].shape}')
    check(z['pose'].shape[:2]==(801,49) and z['pose'].shape[2]>=2, f'{run}: pose shape')
    tick = np.arange(801)*.01
    drift = float(np.max(np.abs(z['time_s']-tick)))
    check(drift<1e-8, f'{run}: neural/record clock')
    check(np.array_equal(z['exchange_time_s'],np.arange(8001)*.001), f'{run}: exchange clock')
    check(np.all(z['external_drive']==0), f'{run}: external drive')
    check(np.all(z['muscle_calcium_mM']>=0), f'{run}: negative recorded calcium')
    check(np.all(z['drive']>=0) and np.all(z['drive']<=.3), f'{run}: drive range')
    check(record['max_gap_pair_current_imbalance_nA']<1e-12, f'{run}: gap balance')
    check(z['neuron_voltage_mV'].min()>=-100 and z['neuron_voltage_mV'].max()<=60, f'{run}: voltage guard')
    z['B'] = z['neuron_voltage_mV'][:,[plan['neurons'].index(n) for n in bnames]]
    expected_initial=-60+np.random.default_rng(13).normal(0,.2,18)
    check(np.array_equal(z['B'][0],expected_initial), f'{run}: seed13 actual initial B voltage')
    z['strain'] = (z['side_length']-z['side_length'][0])/z['side_length'][0]
    bend = (z['strain'][:,:,1]-z['strain'][:,:,0])*.5
    sensed = bend @ z['feedback_weights'].T * np.where(z['feedback_sides']==0,1.,-1.)
    sensed_error = float(np.max(abs(sensed-z['sensed_strain'])))
    expected_current = np.clip(100*sensed,-1,1) if spec['feedback']=='live' else np.zeros_like(sensed)
    current_error = float(np.max(abs(expected_current-z['feedback_pA'][::10])))
    check(sensed_error<1e-12 and current_error<1e-12, f'{run}: shape-feedback reconstruction')
    # Independently reconstruct the declared rank-bin muscle projection.
    calcium=np.maximum(z['muscle_calcium_mM'],0)
    activation=.3*calcium/(calcium+2.5e-7)
    projected=np.zeros((801,48,2))
    for j,muscle in enumerate(plan['muscles']):
        side=0 if muscle[1]=='D' else 1
        segment=2*(int(muscle[3:])-1)
        projected[:,segment:segment+2,side] += .5*activation[:,j,None]
    drive_error=float(abs(projected-z['drive']).max())
    check(drive_error<1e-12, f'{run}: calcium-to-drive reconstruction')
    xy = z['pose'][:,:,:2]*1e6
    centered = xy-xy.mean(axis=1,keepdims=True)
    headward = xy[:,0]-xy[:,-1]
    headward /= np.linalg.norm(headward,axis=1,keepdims=True)
    z['body_y_um'] = centered[:,:,1]*headward[:,None,0]-centered[:,:,0]*headward[:,None,1]
    z['late'] = np.arange(801)>=400
    late = z['late']; v = z['B'][late]
    peaks = [find_peaks(v[:,j], prominence=3)[0] for j in range(18)]
    counts = [len(p) for p in peaks]
    check(all(n>=5 for n in counts), f'{run}: lost B rhythm {counts}')
    # Retain actual neural clock for frequency, canonical sample ticks for phase.
    freq = [float(1/np.diff(z['time_s'][late][p]).mean()) for p in peaks]
    speed = float(np.mean(z['speed'][late])*1e6)
    displacement = float(np.linalg.norm(xy[-1].mean(0)-xy[400].mean(0)))
    check(abs(speed-record['late_mean_speed_um_s'])<1e-10, f'{run}: speed record')
    check(abs(displacement-record['late_centroid_displacement_um'])<1e-10, f'{run}: displacement record')
    provenance = read(directory/'provenance.json')
    check(provenance['protocol_sha256']==digest(root/'protocol.json'), f'{run}: protocol provenance')
    if 'inputs_sha256' in provenance:
        check(provenance['inputs_sha256']==digest(root/'inputs.json'), f'{run}: input provenance')
    else:
        check(provenance['execution_lock_sha256']==digest(root/'execution-lock.json'), f'{run}: execution provenance')
        lock=read(root/'execution-lock.json')
        check(provenance['runner_sha256']==lock['runner_sha256'], f'{run}: runner provenance')
        check(provenance['driver_sha256']==digest(root/'runtime/driver.py'), f'{run}: driver provenance')
        check(provenance['secondorder']==0, f'{run}: secondorder')
    m = dict(feedback=spec['feedback'],dt_ms=spec['dt_ms'],B_names=bnames,muscle_names=plan['muscles'],
        B_peak_counts=counts,B_frequency_Hz=freq,
        B_amplitude_mV=np.ptp(np.percentile(v,[5,95],axis=0),axis=0).tolist(),
        B_mean_mV=v.mean(0).tolist(),B_std_mV=v.std(0).tolist(),
        calcium_mean_mM=z['muscle_calcium_mM'][late].mean(0).tolist(),
        calcium_std_mM=z['muscle_calcium_mM'][late].std(0).tolist(),
        body_lateral_amplitude_um=float(np.ptp(z['body_y_um'][late],axis=0).max()),
        side_strain_RMS=float(rms(z['strain'][late])),late_speed_um_s=speed,
        late_speed_RMS_um_s=float(rms(z['speed'][late])*1e6),late_displacement_um=displacement,
        feedback_max_pA=float(abs(z['feedback_pA']).max()),
        propulsion_gate_passed=bool(speed>=10 and displacement>=20),trace_sha256=trace_sha,
        max_neural_clock_drift_s=drift,sensed_reconstruction_max_error=sensed_error,
        feedback_reconstruction_max_error_pA=current_error,drive_reconstruction_max_error=drive_error,
        provenance=provenance)
    return z,m


def compare(x,y,mx,my,acceptance):
    late = np.arange(801)>=400; vx=x['B'][late]; vy=y['B'][late]
    errors={};details={}
    def gate(k,value,limit):
        errors[k]=dict(value=float(value),limit=float(limit),passed=bool(value<=limit+1e-12))
    fx=np.array(mx['B_frequency_Hz']);fy=np.array(my['B_frequency_Hz'])
    ax=np.array(mx['B_amplitude_mV']);ay=np.array(my['B_amplitude_mV'])
    freq=abs(fx-fy)/fy; amp=abs(ax-ay)/ay; voltage=rms(vx-vy,axis=0)
    gate('B_frequency',freq.max(),acceptance['B_frequency_max_relative'])
    gate('B_amplitude',amp.max(),acceptance['B_q95_q05_amplitude_max_relative'])
    gate('B_mean',abs(vx.mean(0)-vy.mean(0)).max(),acceptance['B_mean_max_abs_mV'])
    gate('B_trajectory',voltage.max(),acceptance['B_unshifted_RMS_max_mV'])
    phase=[];peak_counts=[]
    ticks=np.arange(801)*.01
    for j in range(18):
        px=find_peaks(x['B'][350:,j],prominence=3)[0]+350
        py=find_peaks(y['B'][350:,j],prominence=3)[0]+350
        px_late=px[px>=400];py_late=py[py>=400]
        check(len(px_late)>0 and len(py_late)>0, 'No phase peaks')
        distance=abs(ticks[px_late,None]-ticks[py][None,:]).min(axis=1)
        reverse=abs(ticks[py_late,None]-ticks[px][None,:]).min(axis=1)
        phase.append(float(max(distance.max(),reverse.max())))
        peak_counts.append([len(px_late),len(py_late)])
    gate('B_peak_phase',max(phase),acceptance['B_peak_time_max_abs_s'])
    details.update(B_frequency_relative=freq.tolist(),B_amplitude_relative=amp.tolist(),
        B_unshifted_RMS_mV=voltage.tolist(),B_bidirectional_peak_offset_max_s=phase,
        phase_peak_counts=peak_counts,phase_clock='Canonical 10 ms ticks; no phase fitting or shifting',
        all302_unshifted_RMS_mV=float(rms(x['neuron_voltage_mV'][late]-y['neuron_voltage_mV'][late])))
    for field,label,floor in [('muscle_calcium_mM','calcium',acceptance['calcium_scale_floor_mM']),
                              ('drive','drive',acceptance['drive_scale_floor'])]:
        u=x[field][late].reshape(401,-1);v=y[field][late].reshape(401,-1)
        scale=np.maximum(np.ptp(np.percentile(v,[5,95],axis=0),axis=0),floor)
        sx=np.stack([u.mean(0),u.std(0),np.percentile(u,5,axis=0),np.percentile(u,95,axis=0)])
        sy=np.stack([v.mean(0),v.std(0),np.percentile(v,5,axis=0),np.percentile(v,95,axis=0)])
        stats=abs(sx-sy)/scale;trajectory=rms(u-v,axis=0)/scale
        gate(label+'_statistics',stats.max(),acceptance[label+'_stats_and_trace_scaled_error_max'])
        gate(label+'_trajectory',trajectory.max(),acceptance[label+'_stats_and_trace_scaled_error_max'])
        details[label]=dict(scale=scale.tolist(),scaled_statistics=stats.tolist(),
            scaled_RMS_per_channel=trajectory.tolist(),absolute_RMS=float(rms(u-v)),
            worst_trajectory_channel=int(trajectory.argmax()),worst_statistics_stat_channel=[int(i) for i in np.unravel_index(stats.argmax(),stats.shape)])
    gate('body_shape_trajectory',rms(x['body_y_um'][late]-y['body_y_um'][late]),acceptance['body_shape_RMS_max_um'])
    gate('body_shape_amplitude',abs(mx['body_lateral_amplitude_um']-my['body_lateral_amplitude_um']),
        max(acceptance['body_lateral_amplitude_abs_um'],acceptance['body_lateral_amplitude_relative']*my['body_lateral_amplitude_um']))
    gate('side_strain_trajectory',rms(x['strain'][late]-y['strain'][late]),
        max(acceptance['side_strain_RMS_abs'],acceptance['side_strain_RMS_relative']*rms(y['strain'][late])))
    gate('mean_speed',abs(mx['late_speed_um_s']-my['late_speed_um_s']),acceptance['speed_mean_abs_difference_um_s'])
    gate('speed_trajectory',rms(x['speed'][late]-y['speed'][late])*1e6,acceptance['speed_trace_RMS_um_s'])
    gate('centroid_displacement',abs(mx['late_displacement_um']-my['late_displacement_um']),acceptance['centroid_displacement_abs_difference_um'])
    gate('feedback_trajectory',rms(x['feedback_pA'][4000:]-y['feedback_pA'][4000:]),
        max(acceptance['feedback_RMS_abs_pA'],acceptance['feedback_RMS_relative']*rms(y['feedback_pA'][4000:])))
    return dict(passed=all(v['passed'] for v in errors.values()),
        failed_gates=[k for k,v in errors.items() if not v['passed']],errors=errors,details=details)


def verify_inputs(root):
    records={}
    manifest=root/('sources.json' if (root/'sources.json').exists() else 'inputs.json')
    for path,item in read(manifest).items():
        actual=digest(root/path);check(actual==item['sha256'],f'Input SHA mismatch: {path}')
        records[path]=actual
    protocol=read(root/'protocol.json')
    lock=read(root/'protocol-lock.json')
    check(lock['protocol_sha256']==digest(root/'protocol.json'),'Protocol lock')
    if (root/'execution-lock.json').exists():
        execution=read(root/'execution-lock.json')
        check(execution['protocol_sha256']==digest(root/'protocol.json'),'Execution protocol lock')
        for path,expected in execution['runtime_sha256'].items():
            check(digest(root/path)==expected,f'Execution input: {path}')
            records[path]=expected
    return records,protocol


def compute(root):
    inputs,p=verify_inputs(root)
    traces={};metrics={};invalid={};attempted=[]
    for mode in ('off','live'):
        for dt in STEPS:
            run=name(mode,dt)
            if (root/run/'record.json').exists():
                attempted.append(run)
                try:traces[run],metrics[run]=load_run(root,run)
                except (AssertionError,KeyError,ValueError) as error:
                    invalid[run]=dict(error=repr(error),record=read(root/run/'record.json'),
                        record_sha256=digest(root/run/'record.json'),
                        trace_sha256=digest(root/run/'trace.npz') if (root/run/'trace.npz').exists() else None)
    pairs={}
    for mode in ('off','live'):
        for coarse,fine in ((.025,.0125),(.0125,.00625),(.025,.00625)):
            a,b=name(mode,coarse),name(mode,fine)
            if a in traces and b in traces:
                pairs[f'{mode}_{coarse:g}_vs_{fine:g}']=compare(traces[a],traces[b],metrics[a],metrics[b],p['acceptance'])
    initial=[pairs.get(f'{mode}_0.025_vs_0.0125') for mode in ('off','live')]
    initial_names=[name(mode,dt) for mode in ('off','live') for dt in (.025,.0125)]
    initial_complete=all(n in attempted for n in initial_names)
    need_fine=(any(n in invalid for n in initial_names) or any(not k or not k['passed'] for k in initial)) if initial_complete else None
    complete=(len(attempted)==6 if need_fine else len(attempted)==4) if need_fine is not None else False
    final_pair=(.0125,.00625) if need_fine else (.025,.0125)
    final=[pairs.get(f'{mode}_{final_pair[0]:g}_vs_{final_pair[1]:g}') for mode in ('off','live')]
    accepted=bool(not invalid and all(q and q['passed'] for q in final)) if complete else None
    return dict(source_root=str(root.relative_to(REPO)),protocol_sha256=digest(root/'protocol.json'),
        review_calculation_sha256=digest(Path(__file__)),
        review_environment=dict(python=sys.version,numpy=np.__version__,scipy=importlib.metadata.version('scipy'),
            OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),simulator_imported=False),
        inputs_verified=inputs,new_full_runs=len(attempted),valid_full_runs=len(metrics),
        invalid_or_incomplete_runs=invalid,attempted_conditions=attempted,fine_required=need_fine,
        adaptive_batch_complete=complete,latest_pair_ms=list(final_pair),
        neural_step_convergence_accepted=accepted,metrics=metrics,comparisons=pairs,
        scope='Only neural dt refined; 1 ms exchange/body and IDA settings frozen'),traces


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',action='store_true');args=parser.parse_args()
    result,traces=compute(OLD if args.parent else A)
    target=OUT/('parent-independent.json' if args.parent else 'independent-results.json')
    save(target,result)
    if not args.parent and result['new_full_runs']==4 and result['fine_required'] is not None:
        snapshot=OUT/'initial-independent-results.json'
        if not snapshot.exists():save(snapshot,result)
        save(OUT/'independent-refinement-decision.json',dict(protocol_sha256=result['protocol_sha256'],
            fine_required=result['fine_required'],initial_independent_sha256=digest(snapshot),
            failed_gates={k:v['failed_gates'] for k,v in result['comparisons'].items()},
            invalid_or_incomplete_runs=result['invalid_or_incomplete_runs'],
            scientific_runs_started_by_B=0))
    print(json.dumps({k:result[k] for k in ['new_full_runs','fine_required','adaptive_batch_complete','neural_step_convergence_accepted']}))


if __name__=='__main__': main()
