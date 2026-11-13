"""Frozen, open-loop calcium-to-body mechanism assay; no neuron simulation.

Run from the repository root with PYTHONPATH=src. The calcium waveform is an
explicit engineering input. Its concentration scale is NOT animal-calibrated.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from thermotaxis.neuromechanics.anatomical_body import (
    PlanarMuscleProjection, load_anatomical_body_wiring,
)
from thermotaxis.neuromechanics.boyle_muscle_body import BoyleMuscleBody


ROOT = Path('results/calcium-body-transfer-2026-11-13')
BASE = Path('results/full-network-body-interface-2026-11-05/build')
SUNDIALS = Path('results/neuromechanics-fastbody-2026-10-27/upstream/boyle2012/WormSim/Sundials')
CA_HALF_MM = 2.5e-7


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def build_variant(output, *, dt=.001, tau=.1, drag=1.):
    """Change only declared clock, muscle filter and/or both agar drag axes."""
    original = (BASE / 'worm-muscle-worker.cc').read_text()
    meta = json.loads((BASE / 'build.json').read_text())
    assert sha(BASE / 'worm-muscle-worker.cc') == meta['generated_sha256']
    assert sha(BASE / 'worker') == meta['binary_sha256']
    replacements = {
        '#define DELTAT 0.001': f'#define DELTAT {dt:.17g}',
        'realtype T_muscle = RCONST(0.1);': f'realtype T_muscle = RCONST({tau:.17g});',
        'CL_agar = RCONST(3.2e-3/(2.0*NBAR))': f'CL_agar = RCONST({3.2e-3*drag:.17g}/(2.0*NBAR))',
        'CN_agar = RCONST(128e-3/(2.0*NBAR))': f'CN_agar = RCONST({128e-3*drag:.17g}/(2.0*NBAR))',
    }
    generated = original
    for old, new in replacements.items():
        assert generated.count(old) == 1, old
        generated = generated.replace(old, new)
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'worm-muscle-worker.cc'
    source.write_text(generated)
    (output / 'source.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), generated.splitlines(True), fromfile='frozen-base', tofile='assay-variant')))
    command = ['g++', '-O3', '-Wno-deprecated', '-I', str((SUNDIALS/'include').resolve()),
               str(source.resolve()), '-L', str((SUNDIALS/'lib').resolve()), '-lsundials_ida',
               '-lsundials_cvode', '-lsundials_nvecserial', '-lm', '-o', str((output/'worker').resolve())]
    with (output/'build.log').open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    meta.update(dt_s=dt, muscle_tau_s=tau, drag_scale=drag, build_command=command,
                base_generated_sha256=meta['generated_sha256'], generated_sha256=sha(source),
                binary_sha256=sha(output/'worker'))
    write_json(output/'build.json', meta)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    variants = dict(normal=dict(), fast_muscle=dict(tau=.025), heavy_drag=dict(drag=4.),
                    normal_fine=dict(dt=.0005), fast_muscle_fine=dict(dt=.0005, tau=.025),
                    heavy_drag_fine=dict(dt=.0005, drag=4.))
    for name, args in variants.items():
        build_variant(root/'build'/name, **args)
    specs = []
    for ratio in [.25, 1., 4.]:
        for frequency in [.5, 1., 2.]:
            specs.append(dict(name=f'wave_r{ratio:g}_f{frequency:g}', variant='normal',
                              ratio=ratio, frequency=frequency, pattern='travel'))
    for frequency in [.5, 2.]:
        for pattern in ['reverse', 'synchronous']:
            specs.append(dict(name=f'{pattern}_r1_f{frequency:g}', variant='normal', ratio=1.,
                              frequency=frequency, pattern=pattern))
        for variant in ['fast_muscle', 'heavy_drag']:
            specs.append(dict(name=f'{variant}_r1_f{frequency:g}', variant=variant, ratio=1.,
                              frequency=frequency, pattern='travel'))
    specs.append(dict(name='zero', variant='normal', ratio=0., frequency=.5, pattern='travel'))
    fine_names = ['wave_r1_f0.5', 'wave_r1_f2', 'fast_muscle_r1_f2', 'heavy_drag_r1_f2']
    for name in fine_names:
        base = next(x for x in specs if x['name'] == name)
        specs.append(dict(base, name=name+'_fine', variant=base['variant']+'_fine', paired_with=name))
    design = dict(date='2026-11-13', declared_before_execution=True, duration_s=20.,
                  metric_window_s=[8., 20.], save_dt_s=.01, waves_per_body=1.5,
                  calcium_half_mM=CA_HALF_MM, drive_cap=.3, specs=specs,
                  scope='Prescribed 95-muscle calcium, fixed planar projection, 48x2 effective body channels. No new neurons or learning.',
                  calcium_interpretation='c(t)=K*r*(1+side*sin(2*pi*f*t-direction*phase))/2; r is a model scale, not an animal dose.',
                  coupling='Physical interval first; left-end command then updates existing Euler muscle filter. No physical-state feedback to commands.',
                  hypotheses=['At matched calcium waveform statistics, phase order changes translation.',
                              'Frequency affects muscle-filter gain and mechanical curvature response.',
                              'Changing filter tau or both environmental drag axes isolates their roles.'],
                  numerical_checks=dict(speed_abs_um_s=.2, speed_relative=.02, curvature_relative=.03,
                                        filter_gain_abs_error=.002, zero_motion_um=1e-6,
                                        matching_moments_relative=1e-10, finite_required=True),
                  no_selection='All declared conditions reported. Failed numerical checks stay failed; no parameter tuning.',
                  run_budget=dict(main=18, half_step=4, new_full302=0, training=0))
    write_json(root/'protocol.json', design)
    files = [Path(__file__), BASE/'build.json', BASE/'worm-muscle-worker.cc', BASE/'worker',
             Path('src/thermotaxis/neuromechanics/boyle_body.py'),
             Path('src/thermotaxis/neuromechanics/boyle_muscle_body.py'),
             Path('src/thermotaxis/neuromechanics/anatomical_body.py')]
    files += sorted((root/'build').glob('*/*'))
    # Frozen network snapshots are diagnostic references, never new runs.
    for mode in ['off', 'live']:
        files += [Path(f'results/full302-joint-numerics-2026-11-12/{mode}_dt0.00625/trace.npz')]
    write_json(root/'source-lock.json', {str(p): sha(p) for p in files if p.is_file()})
    write_json(root/'protocol-lock.json', dict(protocol_sha256=sha(root/'protocol.json'),
                                             source_lock_sha256=sha(root/'source-lock.json')))
    print(f'Prepared {len(specs)} cases; source and protocol locked.', flush=True)


def verify_locks(root):
    lock = json.loads((root/'protocol-lock.json').read_text())
    assert sha(root/'protocol.json') == lock['protocol_sha256']
    assert sha(root/'source-lock.json') == lock['source_lock_sha256']
    for name, expected in json.loads((root/'source-lock.json').read_text()).items():
        assert sha(name) == expected, name


def signal(time_s, spec, names, waves=1.5):
    phase = np.array([(int(name[3:])-.5)/24 for name in names])*2*np.pi*waves
    if spec['pattern'] == 'synchronous':
        phase[:] = 0
    direction = -1 if spec['pattern'] == 'reverse' else 1
    side = np.array([1 if name[1] == 'D' else -1 for name in names])
    calcium = CA_HALF_MM*spec['ratio']*(1+side*np.sin(2*np.pi*spec['frequency']*time_s-direction*phase))/2
    drive = .3*calcium/(calcium+CA_HALF_MM)
    return calcium, drive


def run_case(root, design, spec):
    output = root/'cases'/spec['name']
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'spec.json', spec)
    names = load_anatomical_body_wiring().muscle_names
    projection = PlanarMuscleProjection.rank_bins(names)
    rows = []
    moments = np.zeros((4, 95))
    start = time.monotonic()
    with BoyleMuscleBody(root/'build'/spec['variant']/'worker') as body:
        body.reset()
        dt = body.dt_s
        count = round(design['duration_s']/dt)
        save_every = round(design['save_dt_s']/dt)
        for tick in range(count+1):
            t = tick*dt
            ca, activation = signal(t, spec, names, design['waves_per_body'])
            command = projection.to_segments(activation)
            if tick % save_every == 0:
                s = body.state
                rows.append(dict(time_s=t, pose=s.pose.copy(), velocity=s.velocity.copy(),
                                 speed=s.signed_speed_m_s, muscle=s.muscle.copy(), command=command,
                                 calcium_mM=ca, activation=activation,
                                 internal_time_s=s.internal_time_s))
            if tick == count:
                break
            if t >= design['metric_window_s'][0]-1e-12:
                moments += np.stack([ca, ca*ca, activation, activation*activation])*dt
            body.step(1, muscle_drive=command)
    arrays = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    arrays['input_moments'] = moments
    arrays['muscle_names'] = np.array(names)
    np.savez_compressed(output/'trace.npz', **arrays)
    write_json(output/'record.json', dict(complete=True, wall_s=time.monotonic()-start,
                                        dt_s=dt, trace_sha256=sha(output/'trace.npz'),
                                        protocol_sha256=sha(root/'protocol.json')))
    late = arrays['time_s'] >= 8
    print(spec['name'], 'speed_um_s', round(float(arrays['speed'][late].mean()*1e6), 6),
          'wall_s', round(time.monotonic()-start, 2), flush=True)


def run(root):
    verify_locks(root)
    design = json.loads((root/'protocol.json').read_text())
    failures = []
    for spec in design['specs']:
        output = root/'cases'/spec['name']
        if output.exists():
            raise FileExistsError(f'Refusing to replace existing case: {output}')
        try:
            run_case(root, design, spec)
        except Exception as exc:
            failures.append(dict(name=spec['name'], error=repr(exc)))
            write_json(output/'failure.json', failures[-1])
            print('FAILED', spec['name'], repr(exc), flush=True)
    verify_locks(root)
    write_json(root/'execution.json', dict(cases=len(design['specs']), failures=failures))
    if failures:
        raise RuntimeError(f'{len(failures)} cases failed; records retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'verify'])
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'verify': verify_locks}[args.action](args.root)
