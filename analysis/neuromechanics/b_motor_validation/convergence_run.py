"""Isolated, source-pinned full302 convergence runs. No shared files are edited."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / 'results/b-motor-convergence-2026-11-09'
OLD = PROJECT / 'results/b-motor-mechanisms-2026-11-07'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, d):
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def prepare():
    assert not (ROOT/'protocol.json').exists(), 'Never overwrite the preregistration'
    ROOT.mkdir(exist_ok=True)
    inputs = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        inputs[str(target.relative_to(ROOT))] = dict(source=str(source.relative_to(PROJECT)), sha256=sha(source))
    for folder in ['runtime/src', 'runtime/data', 'kernels']:
        for source in sorted((OLD/folder).rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts:
                copy(source, ROOT/source.relative_to(OLD))
    for name in ['selected-parameters.json', 'connection-plan.json', 'requirements.txt']:
        copy(OLD/name, ROOT/name)
    driver = PROJECT/'analysis/neuromechanics/run_full302_b_motor.py'
    frozen = json.loads((OLD/'final-sources.json').read_text())
    assert sha(driver) == frozen[str(driver.relative_to(PROJECT))]
    copy(driver, ROOT/'runtime/driver.py')
    for name in ['worker', 'build.json']:
        copy(PROJECT/'results/full-network-body-interface-2026-11-05/build'/name, ROOT/'runtime/body'/name)
    assert sha(ROOT/'runtime/body/worker') == json.loads((ROOT/'runtime/body/build.json').read_text())['binary_sha256']
    write(ROOT/'inputs.json', inputs)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='302 neurons / 95 muscles, engineering B prototype; neural dt only; no tuning',
        baseline='Two fresh 0.025 ms runs; historical results used only for an independent consistency check',
        duration_s=8, seed=13, feedback=['off', 'live'], nmj_nS=.9, gap_nS=.1,
        feedback_gain_pA_per_strain=100, feedback_clip_pA=1, exchange_ms=1, recorded_ms=10,
        steps_ms=[.025, .0125, .00625], max_new_full_runs=6,
        refinement_rule='Run coarse and middle for both feedback modes. If ANY declared comparison fails in either mode, run fine for both. No extra runs.',
        window_s=[4, 8], reference='Finer member of each adjacent pair; also report coarse vs fine if available',
        acceptance=dict(
            B_frequency_max_relative=.01, B_q95_q05_amplitude_max_relative=.05,
            B_mean_max_abs_mV=.2, B_unshifted_RMS_max_mV=.5,
            B_peak_time_max_abs_s=.01,
            calcium_stats_and_trace_scaled_error_max=.05, calcium_scale_floor_mM=1e-9,
            drive_stats_and_trace_scaled_error_max=.05, drive_scale_floor=.001,
            body_shape_RMS_max_um=.1, body_lateral_amplitude_relative=.05, body_lateral_amplitude_abs_um=.05,
            side_strain_RMS_abs=1e-4, side_strain_RMS_relative=.05,
            speed_mean_abs_difference_um_s=.01, speed_trace_RMS_um_s=.02,
            centroid_displacement_abs_difference_um=.05,
            feedback_RMS_abs_pA=.005, feedback_RMS_relative=.05),
        definitions=dict(
            B_frequency='find_peaks prominence 3 mV, no distance filter, reciprocal mean period; all 18 cells separately',
            B_phase='Match nearest prominent peak in 3.5..8 s to every 4..8 s peak in both directions; report count separately; no phase alignment for acceptance',
            calcium_drive='Per channel mean, std, q05 and q95 differences divided by max(reference q95-q05, scale floor); unshifted RMS divided by same scale; max across channels and statistics',
            body_shape='Transverse position after subtracting centroid and rotating head-tail axis onto x; all 49 nodes; RMS of unshifted differences',
            body_lateral_amplitude='max-node temporal peak-to-peak body-frame transverse position, 4..8 s; mixed abs/relative criterion',
            strain='(side_length - initial side_length)/initial side_length, 48x2',
            speed='Instantaneous signed centroid velocity along actual head-tail axis; compare absolute differences near zero',
            displacement='Euclidean centroid displacement from 4 to 8 s',
            mixed='Pass if difference <= max(absolute floor, relative tolerance * reference magnitude)',
            overall='Every listed gate must pass for both modes. Distribution statistics do not override trajectory or phase failures.',
            limitation='Only neural integration refined; exchange/body 1 ms and IDA settings fixed. A passing pair is operational tolerance evidence, not proof of asymptotic order.'),
        wrapper_sha256=sha(Path(__file__)), inputs_sha256=sha(ROOT/'inputs.json'))
    write(ROOT/'protocol.json', protocol)
    write(ROOT/'protocol-lock.json', dict(protocol_sha256=sha(ROOT/'protocol.json'), locked_before_any_run=True))
    print(json.dumps(dict(prepared=True, source_files=len(inputs), protocol_sha256=sha(ROOT/'protocol.json'))))


def run(feedback, dt):
    lock=json.loads((ROOT/'protocol-lock.json').read_text())
    assert sha(ROOT/'protocol.json') == lock['protocol_sha256']
    inputs=json.loads((ROOT/'inputs.json').read_text())
    for name,d in inputs.items():
        assert sha(ROOT/name) == d['sha256'], name
    assert dt in [.025, .0125, .00625]
    sys.path.insert(0, str(ROOT/'runtime/src'))
    spec=importlib.util.spec_from_file_location('convergence_frozen_driver', ROOT/'runtime/driver.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    body_class=driver.BoyleMuscleBody
    class PinnedBody(body_class):
        def __init__(self, ignored_binary):
            super().__init__(ROOT/'runtime/body/worker')
    driver.BoyleMuscleBody=PinnedBody
    driver.ROOT=ROOT
    name=f'{feedback}_dt{dt:g}'
    args=argparse.Namespace(name=name, model='prototype', feedback=feedback, replay=None,
        gap_nS=.1, feedback_gain=100, feedback_clip=1, duration=8., dt_ms=dt,
        seed=13, avb_on=True, calcium_scale=1, perturb=False, nmj_nS=.9)
    os.chdir(PROJECT)
    driver.run(args)
    import neuron, numpy, scipy
    write(ROOT/name/'provenance.json', dict(protocol_sha256=sha(ROOT/'protocol.json'),
        inputs_sha256=sha(ROOT/'inputs.json'), wrapper_sha256=sha(Path(__file__)),
        python=sys.version, executable=sys.executable,
        versions=dict(neuron=neuron.__version__,numpy=numpy.__version__,scipy=scipy.__version__),
        OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),
        path_only_overrides=['ROOT', 'BoyleMuscleBody binary location'],
        started_inputs_verified=True))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare', action='store_true')
    p.add_argument('--feedback', choices=['off','live']);p.add_argument('--dt', type=float)
    a=p.parse_args()
    if a.prepare: prepare()
    else: run(a.feedback,a.dt)
