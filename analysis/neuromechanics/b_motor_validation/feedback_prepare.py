"""Freeze branch-B protocol and verify old inputs against the sealed archive."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

REPO = Path(__file__).resolve().parents[3]
OLD = REPO / 'results/b-motor-mechanisms-2026-11-07'
ROOT = REPO / 'results/b-motor-feedback-causality-2026-11-09'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ROOT.mkdir(exist_ok=False)
    archive = REPO / 'docs/reports/assets/b-motor-mechanisms-2026-11-07/reproduction.zip'
    validation = json.loads((OLD / 'archive-validation.json').read_text())
    assert sha(archive) == validation['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('MANIFEST.json'))
        selected = [name for name in manifest if name.startswith(str(OLD.relative_to(REPO)) + '/runtime/')
                    or name.startswith(str(OLD.relative_to(REPO)) + '/kernels/')]
        selected += [str((OLD / name).relative_to(REPO)) for name in ['connection-plan.json', 'selected-parameters.json', 'requirements.txt']]
        selected += ['analysis/neuromechanics/run_full302_b_motor.py']
        selected += ['results/full-network-body-interface-2026-11-05/build/' + n for n in ['worker', 'build.json', 'worm-muscle-worker.cc']]
        for case in ['nmj9_live8', 'nmj9_off8']:
            selected += [str((OLD / case / n).relative_to(REPO)) for n in ['record.json', 'spec.json', 'trace.npz']]
        verified = {}
        for name in selected:
            p = REPO / name
            assert sha(p) == manifest[name], name
            verified[name] = manifest[name]
        # Copy only sealed inputs; no legacy files are changed or compiled.
        for sub in ['runtime', 'kernels']:
            shutil.copytree(OLD / sub, ROOT / sub, ignore=shutil.ignore_patterns('__pycache__'))
        for n in ['connection-plan.json', 'selected-parameters.json', 'requirements.txt']:
            shutil.copy2(OLD / n, ROOT / n)
        (ROOT / 'runtime/analysis').mkdir()
        shutil.copy2(REPO / 'analysis/neuromechanics/run_full302_b_motor.py', ROOT / 'runtime/analysis/frozen_driver.py')
        (ROOT / 'body').mkdir()
        for n in ['worker', 'build.json', 'worm-muscle-worker.cc']:
            shutil.copy2(REPO / 'results/full-network-body-interface-2026-11-05/build' / n, ROOT / 'body' / n)
        for oldcase, newcase in [('nmj9_live8', 's13_baseline'), ('nmj9_off8', 's13_off_baseline')]:
            shutil.copytree(OLD / oldcase, ROOT / newcase)
    protocol = dict(frozen_before_new_runs=True, duration_s=8, neural_dt_ms=.025, exchange_ms=1,
        seeds=[13,29], new_full_run_budget=8, model='original engineering Ca/K prototype',
        neurons=302, muscles=95, nmj_nS=.9, gap_nS=.1, feedback_gain_pA_per_strain=100, feedback_clip_pA=1,
        intervention=dict(start_tick=2000, end_tick_exclusive=2300, segments=list(range(8)), side=0, extra_drive=.2,
                          interpretation='external physical intervention; not autonomous locomotion'),
        conditions=['s13_baseline (sealed legacy reuse)', 's13_live_pert', 's13_replay_base_pert', 's13_off_pert',
                    's13_replay_pert', 's29_baseline', 's29_live_pert', 's29_replay_base_pert', 's29_replay_pert'],
        update_order='at tick k: accepted body/neural state -> sensed current or replay[k] -> neural muscle drive + external[k] -> body step -> neural step; affects interval k to k+1',
        replay_rule='source hash, seed, names, fixed parameters and exact 8001-point exchange grid must match; source[k] only; no future state lookup',
        pre_intervention_checks='live vs baseline and both replay conditions identical through state at tick2000; off vs its sealed off baseline; applied external commands identical',
        analysis_windows_s=dict(pre=[0,2], during=[2,2.3], recovery=[2.3,4], late=[4,8]),
        phase_method='positive voltage peaks prominence3mV, distance0.25s; ordinal matched post-2s peaks; shift divided by same-cell baseline local period; separately amplitude and peak-count changes',
        acceptance=dict(replay_same_source='all shared physical/neural arrays bitwise identical',
                        causal='state unchanged at tick2000; body/sensory differences no earlier than tick2001; B response no earlier than tick2002',
                        phase='report peak timing only for matched counts, separately first and late peaks; no full PRC or entrainment claim',
                        propulsion='late signed mean >=10um/s and displacement >=20um; external-perturbation motion excluded from autonomous claim'),
        limits=['no physiological fit', 'no training or optimization', 'convergence assessed independently by branch A'])
    (ROOT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (ROOT / 'source-verification.json').write_text(json.dumps(dict(archive_sha256=sha(archive), verified_archive_payload=verified), indent=2) + '\n')
    print(json.dumps(dict(verified_files=len(verified), protocol_sha256=sha(ROOT/'protocol.json'), root=str(ROOT))))

if __name__ == '__main__':
    main()
