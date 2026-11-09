"""Branch-B runner: frozen dynamics with per-exchange causal audit data."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'results/b-motor-feedback-causality-2026-11-09'
spec = importlib.util.spec_from_file_location('frozen_b_motor', ROOT/'runtime/analysis/frozen_driver.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.ROOT = ROOT

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main(a):
    # These assertions prevent a CLI option from silently altering the frozen protocol.
    assert a.seed in [13, 29] and a.duration == 8 and a.dt_ms == .025 and a.nmj_nS == .9
    out = ROOT/a.name
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    (out/'spec.json').write_text(json.dumps(vars(a), indent=2)+'\n')
    plan = json.loads((ROOT/'connection-plan.json').read_text())
    network = driver.C1Network(ROOT/'kernels', plan, dt_ms=a.dt_ms)
    names, points, clamps, afferents, gaps, initial = driver.configure(network, a)
    network.reset()
    w = driver.load_anatomical_body_wiring()
    projection = driver.PlanarMuscleProjection.rank_bins(w.muscle_names)
    weights, sides, centers = driver.field(w, names)
    bidx = [plan['neurons'].index(n) for n in names]
    replay = None
    replay_metadata = None
    if a.replay:
        src = ROOT/a.replay
        record = json.loads((src/'record.json').read_text())
        assert record['complete'] and sha(src/'trace.npz') == record['trace_sha256']
        for k in ['duration', 'dt_ms', 'seed', 'model', 'gap_nS', 'nmj_nS', 'feedback_gain', 'feedback_clip', 'avb_on', 'calcium_scale']:
            assert record['spec'][k] == vars(a)[k], k
        with np.load(src/'trace.npz') as z:
            assert z['B_names'].tolist() == list(names)
            assert np.array_equal(z['exchange_time_s'], np.arange(8001)*.001)
            replay = z['feedback_pA'].copy()
        assert replay.shape == (8001, 18)
        replay_metadata = dict(source=a.replay, trace_sha256=sha(src/'trace.npz'), indices='0..8000, exactly current tick', source_perturb=record['spec']['perturb'])
    rows = []
    fine = []
    feedback = []
    max_gap = 0.
    with driver.BoyleMuscleBody(ROOT/'body/worker') as body:
        body.reset()
        rest = body.state.side_length.copy()
        for tick in range(8001):
            t = tick/1000
            s = network.state()
            strain = (body.state.side_length-rest)/rest
            bend = (strain[:,1]-strain[:,0])/2
            sensed = np.where(sides == 0, 1., -1.)*(weights@bend)
            candidate = np.clip(a.feedback_gain*sensed, -a.feedback_clip, a.feedback_clip)
            actual = replay[tick] if replay is not None else candidate if a.feedback == 'live' else np.zeros(18)
            for clamp, value in zip(afferents, actual):
                clamp.amp = float(value)/1000
            feedback.append(actual.copy())
            calcium = np.maximum(s['muscle_calcium_mM'], 0)
            drive = projection.to_segments(.3*calcium/(calcium+2.5e-7))
            external = np.zeros((48,2))
            if a.perturb and 2000 <= tick < 2300:
                external[:8,0] = .2
            command = np.clip(drive+external, -1, 1)
            assert -100 <= s['neuron_voltage_mV'].min() and s['neuron_voltage_mV'].max() <= 60
            assert abs(body.state.time_s-t) < 1e-9 and abs(s['time_s']-t) < 1e-9
            assert body.state.microsteps == tick and body.state.internal_time_s <= t+1e-10
            fine.append(dict(B_voltage_mV=s['neuron_voltage_mV'][bidx], side_length=body.state.side_length.copy(),
                candidate_pA=candidate, command=command, external_drive=external, body_time_s=body.state.time_s,
                neural_time_s=s['time_s'], body_internal_time_s=body.state.internal_time_s,
                replay_index=tick if replay is not None else -1))
            if tick % 10 == 0:
                rows.append(dict(**s, pose=body.state.pose.copy(), drive=drive.copy(), external_drive=external.copy(),
                    speed=body.state.signed_speed_m_s, body_muscle=body.state.muscle.copy(), side_length=body.state.side_length.copy(),
                    sensed_strain=sensed, B_gate=np.array([p.n for p in points])))
                max_gap = max(max_gap, max(abs(float(x.i+y.i)) for x,y in network.gap_connections))
            if tick == 8000:
                break
            body.step(1, muscle_drive=command)
            network.advance(1.)
            if (tick+1) % 1000 == 0:
                print(a.name, (tick+1)/1000, 's', flush=True)
    values = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    values.update(exchange_time_s=np.arange(8001)*.001, feedback_pA=np.array(feedback), B_names=np.array(names),
                  neuron_names=np.array(w.neuron_names), feedback_weights=weights, feedback_sides=sides, nominal_centers=centers)
    np.savez_compressed(out/'trace.npz', **values)
    np.savez_compressed(out/'exchange-audit.npz', **{k:np.array([r[k] for r in fine]) for k in fine[0]})
    late = values['time_s'] >= 4-1e-8
    record = dict(spec=vars(a), complete=True, neurons=302, muscles=95, chemical_connections=len(network.chemical_connections),
        gap_pairs=len(network.gap_connections), actual_duration_s=float(values['time_s'][-1]), B_names=names,
        B_initial_voltage_mV=initial.tolist(), modified_AVB_gap_pairs=gaps, max_gap_pair_current_imbalance_nA=max_gap,
        late_mean_speed_um_s=float(values['speed'][late].mean()*1e6),
        late_centroid_displacement_um=float(np.linalg.norm(values['pose'][-1,:,:2].mean(0)-values['pose'][np.flatnonzero(late)[0],:,:2].mean(0))*1e6),
        trace_sha256=sha(out/'trace.npz'), exchange_audit_sha256=sha(out/'exchange-audit.npz'),
        replay_provenance=replay_metadata, protocol_sha256=sha(ROOT/'protocol.json'), wall_s=time.monotonic()-started,
        driver_sha256=sha(Path(__file__)), frozen_driver_sha256=sha(ROOT/'runtime/analysis/frozen_driver.py'))
    (out/'record.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['complete','wall_s','late_mean_speed_um_s','trace_sha256']}), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--feedback', choices=['live','off'], default='live')
    p.add_argument('--perturb', action='store_true')
    p.add_argument('--replay')
    p.set_defaults(model='prototype', duration=8., dt_ms=.025, gap_nS=.1, nmj_nS=.9,
                   feedback_gain=100, feedback_clip=1, avb_on=True, calcium_scale=1)
    main(p.parse_args())
