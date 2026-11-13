"""Open-loop, linearly resampled replay of frozen 302-network muscle calcium.

No neurons are simulated. Run from the project root with PYTHONPATH=src.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from thermotaxis.neuromechanics.anatomical_body import PlanarMuscleProjection, load_anatomical_body_wiring
from thermotaxis.neuromechanics.boyle_muscle_body import BoyleMuscleBody


ROOT=Path('results/network-output-replay-2026-11-13')
SOURCE=Path('results/full302-joint-numerics-2026-11-12')
PARENT=Path('results/calcium-body-transfer-2026-11-13')
K=2.5e-7


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def interpolation(source_time,calcium):
    u=np.asarray(source_time)
    assert np.all(u>=-1e-12) and np.all(u<=8+1e-12)
    x=np.clip(u/.01,0,800)
    index=np.minimum(np.floor(x).astype(int),799)
    fraction=x-index
    return calcium[index]*(1-fraction[...,None])+calcium[index+1]*fraction[...,None]


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    w=load_anatomical_body_wiring()
    projection=PlanarMuscleProjection.rank_bins(w.muscle_names)
    source_audits=[]
    for mode in ['off','live']:
        path=SOURCE/(mode+'_dt0.00625')/'trace.npz'
        z=np.load(path)
        canonical=np.arange(801)*.01
        assert len(z['time_s'])==801 and np.max(abs(z['time_s']-canonical))<1e-8
        c=z['muscle_calcium_mM']
        assert c.shape==(801,95) and np.isfinite(c).all() and c.min()>=0
        expected=projection.to_segments(.3*c/(c+K))
        np.testing.assert_allclose(expected,z['drive'],atol=2e-15,rtol=0)
        (root/'inputs').mkdir(exist_ok=True)
        np.savez_compressed(root/'inputs'/f'{mode}.npz',source_actual_time_s=z['time_s'],
                            time_s=canonical,calcium_mM=c,original_drive=z['drive'],
                            muscle_names=np.array(w.muscle_names),original_pose=z['pose'],
                            original_speed=z['speed'])
        source_audits.append(dict(mode=mode,path=str(path),sha256=sha(path),
                                  time_rounding_max_s=float(abs(z['time_s']-canonical).max()),
                                  drive_reconstruction_max=float(abs(expected-z['drive']).max())))
    for name,base_name in [('normal','normal'),('fine','normal_fine')]:
        folder=root/'build'/name
        folder.mkdir(parents=True)
        for filename in ['worker','worm-muscle-worker.cc','build.json','source.patch']:
            shutil.copy2(PARENT/'build'/base_name/filename,folder/filename)
        meta=json.loads((folder/'build.json').read_text())
        assert sha(folder/'worker')==meta['binary_sha256']
        assert sha(folder/'worm-muscle-worker.cc')==meta['generated_sha256']
    specs=[]
    for mode in ['off','live']:
        for label,amplitude,scale in [('original',1.,1.),('calcium_x2',2.,1.),
                                       ('calcium_x4',4.,1.),('time_x4',1.,4.)]:
            specs.append(dict(name=mode+'_'+label,mode=mode,amplitude=amplitude,time_scale=scale,
                              duration_s=8*scale,dt_s=.001,worker='normal'))
    for parent in ['live_original','live_time_x4']:
        base=next(s for s in specs if s['name']==parent)
        specs.append(dict(base,name=parent+'_fine',paired_with=parent,dt_s=.0005,worker='fine'))
    protocol=dict(date='2026-11-13',scope='Linearly resampled open-loop counterfactual of a frozen, not-system-converged network snapshot. No new neurons or learning.',
                  interpolation='Linear in calcium on a canonical 0..8s 10ms grid; original floating timestamp deviation <1e-8s. Interpolate then scale calcium then saturate then fixed projection.',
                  canonical_grid_normalization='Only timestamp roundoff removed, no signal samples changed.',
                  model=dict(calcium_half_mM=K,drive_cap=.3,muscle_tau_s=.1,drag='original agar anisotropy40',
                             projection='unchanged rank bins from 95 named muscles to 48 dorsal/ventral pairs'),
                  clock='At each left-end physical tick t, source u=t/time_scale; physical interval solves first, then command updates muscle filter. No feedback to replayed calcium.',
                  source_window_s=[4.,8.],physical_windows='[4*time_scale,8*time_scale]',
                  save_clock='0.01*time_scale physical seconds; all cases have 801 corresponding source-time samples.',
                  moments='Per-unit-physical-time means of c,c^2,a,a^2. Delivered tick quadrature, analytic continuous piecewise-linear calcium moments, and 32-point Gauss activation moments reported separately; 16-point cross-check. Stretching time multiplies total dose by4, not its time average.',
                  numerical=dict(speed_abs_um_s=.002,speed_rel=.02,
                                 dynamic_curvature_rel=.03,dynamic_curvature_abs_per_mm=.001,
                                 metric='Central39 curvature values: dynamic RMS removes each spatial point temporal mean; static RMS reported separately. No phase alignment.',
                                 finite_and_causal=True),
                  hypotheses=['Calcium amplitude and time dilation may change replayed movement differently.',
                              'Neither intervention is assumed to restore autonomous network movement.'],
                  specs=specs,budget=dict(main=8,half_step=2,new_neurons=0,training=0),
                  no_tuning='All declared cases retained; no response-based extra runs or parameter changes.',
                  parent_approval='Root internal review before main execution: eight main plus two preselected half-step pairs approved.')
    dump(root/'protocol.json',protocol)
    dump(root/'source-audit.json',source_audits)
    files=[Path(__file__).resolve().relative_to(Path.cwd()),
           Path('analysis/neuromechanics/network_output_replay/analyze.py')]
    files += [Path('src/thermotaxis/neuromechanics')/name for name in [
        'boyle_body.py','boyle_muscle_body.py','boyle_output.py','anatomical_body.py','muscle_mapping.py']]
    files += sorted((root/'inputs').glob('*.npz'))+sorted((root/'build').glob('*/*'))
    dump(root/'source-lock.json',{str(p):sha(p) for p in files})
    dump(root/'protocol-lock.json',dict(protocol_sha256=sha(root/'protocol.json'),source_lock_sha256=sha(root/'source-lock.json')))
    print('Prepared and locked 8 main + 2 half-step cases',flush=True)


def verify(root):
    lock=json.loads((root/'protocol-lock.json').read_text())
    assert sha(root/'protocol.json')==lock['protocol_sha256']
    assert sha(root/'source-lock.json')==lock['source_lock_sha256']
    for p,digest in json.loads((root/'source-lock.json').read_text()).items():assert sha(p)==digest,p


def run_case(root,spec):
    out=root/'cases'/spec['name'];out.mkdir(parents=True,exist_ok=False)
    dump(out/'spec.json',spec)
    source=np.load(root/'inputs'/f'{spec["mode"]}.npz')
    projection=PlanarMuscleProjection.rank_bins(tuple(source['muscle_names']))
    count=round(spec['duration_s']/spec['dt_s'])
    ticks=np.arange(count+1)*spec['dt_s']
    ca=interpolation(ticks/spec['time_scale'],source['calcium_mM'])*spec['amplitude']
    activation=.3*ca/(ca+K)
    commands=projection.to_segments(activation)
    every=round(.01*spec['time_scale']/spec['dt_s'])
    rows=[];started=time.monotonic()
    with BoyleMuscleBody(root/'build'/spec['worker']/'worker') as body:
        body.reset();assert body.dt_s==spec['dt_s']
        for tick in range(count+1):
            if tick%every==0:
                state=body.state
                rows.append(dict(time_s=ticks[tick],source_time_s=ticks[tick]/spec['time_scale'],
                                 pose=state.pose.copy(),velocity=state.velocity.copy(),speed=state.signed_speed_m_s,
                                 muscle=state.muscle.copy(),calcium_mM=ca[tick],activation=activation[tick],
                                 command=commands[tick],internal_time_s=state.internal_time_s))
            if tick==count:break
            body.step(1,muscle_drive=commands[tick])
    arrays={k:np.array([r[k] for r in rows]) for k in rows[0]}
    selected=(ticks[:-1]>=4*spec['time_scale']-1e-12)
    arrays['delivered_tick_moments']=np.stack([ca[:-1][selected].mean(0),
        (ca[:-1][selected]**2).mean(0),activation[:-1][selected].mean(0),
        (activation[:-1][selected]**2).mean(0)])
    np.savez_compressed(out/'trace.npz',**arrays)
    dump(out/'record.json',dict(complete=True,trace_sha256=sha(out/'trace.npz'),
                               protocol_sha256=sha(root/'protocol.json'),wall_s=time.monotonic()-started,
                               physical_ticks=count,neuron_runs=0))
    print(spec['name'],'complete',round(time.monotonic()-started,2),'s',flush=True)


def run(root):
    verify(root);protocol=json.loads((root/'protocol.json').read_text());failures=[]
    for spec in protocol['specs']:
        try:run_case(root,spec)
        except Exception as exc:
            failure=dict(name=spec['name'],error=repr(exc));failures.append(failure)
            dump(root/'cases'/spec['name']/'failure.json',failure)
            print('FAILED',failure,flush=True)
    verify(root);dump(root/'execution.json',dict(cases=len(protocol['specs']),failures=failures))
    if failures:raise RuntimeError(failures)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','verify'])
    parser.add_argument('--root',type=Path,default=ROOT)
    a=parser.parse_args();{'prepare':prepare,'run':run,'verify':verify}[a.action](a.root)
