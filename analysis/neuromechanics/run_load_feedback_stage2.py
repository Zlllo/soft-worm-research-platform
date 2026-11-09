"""Paired drag-pulse experiment with actual or prerecorded stretch feedback."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from build_boyle_load_worker import build
from thermotaxis.neuromechanics.boyle_body import BoyleBody
from thermotaxis.neuromechanics.boyle_output import curvature,side_lengths


REPO=Path(__file__).resolve().parents[2]
DEFAULT_SOURCE=Path('results/neuromechanics-fastbody-2026-10-27/upstream/boyle2012')
DEFAULT_BASE=Path('results/neuromechanics-stepper-2026-10-27/build-v2')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(root,source,base):
    if root.exists():raise FileExistsError('preserve existing run')
    sources=['analysis/neuromechanics/build_boyle_load_worker.py','analysis/neuromechanics/run_load_feedback_stage2.py',
        'src/thermotaxis/neuromechanics/boyle_body.py','src/thermotaxis/neuromechanics/boyle_output.py','tests/test_boyle_load_worker.py']
    inputs=[base/'worm-worker.cc',base/'build.json',source/'WormSim/Model/worm.cc']
    inputs+=[source/'WormSim/Sundials/lib'/f'libsundials_{s}.a' for s in ['ida','cvode','nvecserial']]
    design=dict(status='declared_before_load_runs',source=str(source),base_build=str(base),
        duration_s=20.,sample_dt_s=.02,dt_s=.001,check_dt_s=.0005,drive=.675,head_bias=0.,
        factors=[1.,1.5,2.],feedback=['live','replay'],phases=8,pulse_duration_s=2.,affected_rods=list(range(13)),
        onset_rule='first upward head-curvature crossing after 5 s plus j/8 of baseline period, rounded to 1ms',
        period_rule='median interval of interpolated upward zero crossings after 3 s',
        numerical_check='phase0 factor2, nominal/live/replay at half dt; same declared physical onset',
        metrics=dict(recovery_lag_tolerance_rad=math.pi/18,recovery_speed_tolerance_fraction=.1,
                     recovery_hold_periods=1,phase_evaluation='offline analytic signal, excluded last baseline period; never sent to controller'),
        scope='simplified forward motor system; equal scalar scaling of anterior longitudinal/transverse effective drag; no thermotaxis controller',
        source_sha256={s:digest(REPO/s) for s in sources},inputs_sha256={str(s):digest(REPO/s) for s in inputs})
    write(root/'design.json',design)


def check(design):
    for name,sha in {**design['source_sha256'],**design['inputs_sha256']}.items():
        if digest(REPO/name)!=sha:raise RuntimeError('changed source/input: '+name)


def simulate(root,name,design,*,dt=None,factor=1.,onset=6.,replay=None,record=None):
    out=root/name
    if (out/'run.json').exists():
        r=json.loads((out/'run.json').read_text())
        if digest(out/'trajectory.npz')!=r['trajectory_sha256']:raise RuntimeError('changed completed trajectory')
        return out
    if out.exists():raise FileExistsError('incomplete run preserved: '+str(out))
    started=time.monotonic()
    binary=build(REPO/design['source'],REPO/design['base_build'],out,factor=factor,onset=onset,
        duration=design['pulse_duration_s'],dt=dt or design['dt_s'],replay=replay,record=record,
        trace_interval=design['sample_dt_s'])
    build_wall=time.monotonic()-started;started=time.monotonic()
    with BoyleBody(binary) as body:
        states=[body.reset()]
        for _ in range(round(design['duration_s']/design['sample_dt_s'])):
            states.append(body.advance(design['sample_dt_s'],drive=design['drive'],head_bias=design['head_bias']))
    wall=time.monotonic()-started
    t=np.array([s.time_s for s in states]);pose=np.stack([s.pose for s in states])
    trace=np.fromfile(out/'feedback-trace.bin').reshape(-1,194)
    np.testing.assert_allclose(trace[:,0],t[1:],atol=1e-12,rtol=0)
    actual=trace[:,1:97].reshape(-1,48,2);used=trace[:,97:193].reshape(-1,48,2)
    geometry_error=float(np.max(abs(actual-side_lengths(pose[1:]))))
    if geometry_error>1e-17:raise RuntimeError('trace reports a nonphysical current geometry')
    np.savez_compressed(out/'trajectory.npz',time_s=t,pose=pose,
        neural=np.stack([s.neural for s in states]),muscle=np.stack([s.muscle for s in states]),
        stretch=np.stack([s.stretch for s in states]),velocity=np.stack([s.velocity for s in states]),
        actual_side=np.concatenate([side_lengths(pose[:1]),actual]),
        used_side=np.concatenate([side_lengths(pose[:1]),used]),
        load=np.r_[1.,trace[:,-1]],internal_time_s=np.array([s.internal_time_s for s in states]))
    ahead=max(s.internal_time_s-s.time_s for s in states)
    if ahead>1e-10:raise RuntimeError('solver passed controller clock')
    write(out/'run.json',dict(name=name,factor=factor,onset_s=onset,dt_s=dt or design['dt_s'],
        feedback='replay' if replay is not None else 'live',wall_s=wall,build_wall_s=build_wall,
        max_geometry_error_m=geometry_error,max_solver_ahead_s=ahead,
        trajectory_sha256=digest(out/'trajectory.npz'),complete=True))
    print('Completed physical run',name,flush=True)
    return out


def baseline_clock(path):
    with np.load(path) as z:
        t=z['time_s'];signal=curvature(z['pose'])[:,:5].mean(1)
    ids=np.flatnonzero((signal[:-1]<=0)&(signal[1:]>0)&(t[:-1]>=3))
    cross=t[ids]-signal[ids]*(t[ids+1]-t[ids])/(signal[ids+1]-signal[ids])
    if len(cross)<3:raise RuntimeError('insufficient normal gait crossings')
    period=float(np.median(np.diff(cross)))
    first=float(cross[cross>=5][0])
    return dict(period_s=period,reference_upward_crossing_s=first,crossings_s=cross.tolist(),
                onsets_s=[round((first+j*period/8)/.001)*.001 for j in range(8)])


def run(root,design):
    reference=root/'reference-1ms.bin'
    baseline=simulate(root,'nominal-live',design,record=reference)
    clock=baseline_clock(baseline/'trajectory.npz');write(root/'baseline-clock.json',clock)
    nominal_replay=simulate(root,'nominal-replay',design,replay=reference)
    with np.load(baseline/'trajectory.npz') as a,np.load(nominal_replay/'trajectory.npz') as b:
        for name in ['pose','neural','muscle','stretch','actual_side','used_side']:
            np.testing.assert_array_equal(a[name],b[name])
    records=[]
    for phase,onset in enumerate(clock['onsets_s']):
        for factor in design['factors']:
            for feedback in design['feedback']:
                if factor==1.:
                    folder=baseline if feedback=='live' else nominal_replay
                else:
                    folder=simulate(root,f'phase-{phase:02d}-load-{factor:g}-{feedback}',design,
                        factor=factor,onset=onset,replay=reference if feedback=='replay' else None)
                records.append(dict(phase=phase,onset_s=onset,factor=factor,feedback=feedback,path=str(folder.relative_to(root))))
    write(root/'conditions.json',dict(rows=records,nominal_replay_identity=True,complete=True))
    half=design['check_dt_s'];half_ref=root/'reference-halfms.bin'
    half_baseline=simulate(root,'half-dt-nominal',design,dt=half,record=half_ref)
    for feedback in design['feedback']:
        simulate(root,'half-dt-load-2-'+feedback,design,dt=half,factor=2.,onset=clock['onsets_s'][0],
                 replay=half_ref if feedback=='replay' else None)
    check(design)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run'])
    p.add_argument('--root',type=Path,default=Path('results/body-load-stage2-2026-11-03'))
    p.add_argument('--source',type=Path,default=DEFAULT_SOURCE);p.add_argument('--base-build',type=Path,default=DEFAULT_BASE)
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.root,a.source,a.base_build)
    else:
        d=json.loads((a.root/'design.json').read_text());check(d);run(a.root,d)
