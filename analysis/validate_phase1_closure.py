"""Independent recomputation and actual-core replay of phase-1 closure artifacts."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from thermotaxis.contracts import SensorReading
from thermotaxis.phase1 import TwoTimescaleTurnController, run_phase1_experiment
from thermotaxis.phase1_config import MemoryControllerConfig, Phase1ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/reports/assets/phase1-closure-2026-10-04'


def finite_kernel_coefficient(d, h):
    hs, v, b = d['sensor_interval_s'], d['speed_m_s'], d['error_slope_K_m']
    r = 2*math.exp(-d['baseline_rate_s']*h)-1
    moments = []
    for tau in [d['fast_tau_s'], d['slow_tau_s']]:
        q = math.exp(-hs/tau)
        moments.append(-v*b*hs*sum(q**(j*round(h/hs)+k)*r**j
                       for j in range(5000) for k in range(1, round(h/hs)+1)))
    return 2*v*h*math.exp(-d['baseline_rate_s']*h)/(1-r)*(moments[1]-moments[0])


def sampling_checks():
    path = ROOT/'docs/reports/assets/phase1-sampling-2026-10-04/sampling-response.json'
    data = json.loads(path.read_text()); d = data['design']; replay = []
    assert data['script_sha256'] == hashlib.sha256((ROOT/'analysis/phase1_sampling_response.py').read_bytes()).hexdigest()
    for group in data['groups']:
        values = [r['drift_m_s'] for r in group['records']]
        mean = statistics.mean(values); se = statistics.stdev(values)/math.sqrt(len(values))
        assert abs(mean-group['mean_drift_m_s']) < 1e-18
        assert abs(se-group['standard_error_m_s']) < 1e-18
        assert np.allclose(group['normal95_m_s'], [mean-1.96*se, mean+1.96*se], atol=1e-18, rtol=0)
        assert abs(group['coefficient_m_K']-finite_kernel_coefficient(d, group['control_interval_s'])) < 1e-15
        if group['gain_K_s'] != .1:
            continue
        record = group['records'][0]; h = group['control_interval_s']; hs = d['sensor_interval_s']
        rng = np.random.default_rng(record['seed'])
        direction = 1 if rng.random() < .5 else -1
        controller = TwoTimescaleTurnController(MemoryControllerConfig(.5, 5., .1, .01, .5, .1, 0.), 293.15, 0)
        controller._rng = rng
        error, displacement, turns = 1000., 0., 0
        controller.observe(SensorReading(293.15+error, 0.))
        for n in range(round((d['burn_s']+d['measurement_s'])/h)):
            if n >= round(d['burn_s']/h):
                displacement += direction*d['speed_m_s']*h
            for j in range(1, round(h/hs)+1):
                error += direction*d['speed_m_s']*d['error_slope_K_m']*hs
                controller.observe(SensorReading(293.15+error, n*h+j*hs))
            if controller.should_turn(h):
                direction *= -1; turns += 1
        assert error > 0
        assert turns == record['turn_count']
        assert abs(displacement/d['measurement_s']-record['drift_m_s']) < 1e-15
        replay.append(dict(h_s=h, seed=record['seed'], turn_count=turns,
                           full_measurement_drift_m_s=displacement/d['measurement_s']))
    return dict(groups_checked=len(data['groups']), trajectories_checked=sum(len(g['records']) for g in data['groups']),
                core_full_length_replays=replay)


def independent_blocks(result, cfg, edges, bins):
    """General overlap integration, splitting wall segments without core helpers."""
    out = []; L = cfg.domain.x_extent_m; dt = cfg.time.physics_dt_s; v = cfg.point_model.speed_m_s
    G = cfg.temperature.gradient_K_per_m[0]; ref = cfg.temperature.reference_K
    target = cfg.temperature.preferred_K; half = cfg.temperature.comfort_half_width_K
    lower, upper = sorted(((target-half-ref)/G, (target+half-ref)/G))
    bin_edges = np.linspace(0., L, bins+1)
    for t0, t1 in zip(edges[:-1], edges[1:]):
        hist = np.zeros(bins); comfort = energy = 0.
        for i in range(round(t0/dt), round(t1/dt)):
            row, endrow = result['trajectory'][i:i+2]
            x, y, s = row['position_m'], endrow['position_m'], row['direction']
            unfolded = x+s*v*dt
            if unfolded > L:
                segments = [(x, L), (L, y)]
            elif unfolded < 0:
                segments = [(x, 0.), (0., y)]
            else:
                segments = [(x, y)]
            for start, end in segments:
                lo, hi = sorted((start, end)); duration = (hi-lo)/v
                hist += np.maximum(0., np.minimum(hi, bin_edges[1:])-np.maximum(lo, bin_edges[:-1]))/v
                comfort += max(0., min(hi, upper)-max(lo, lower))/v
                e0, e1 = ref+G*start-target, ref+G*end-target
                energy += duration*(e0*e0+e0*e1+e1*e1)/3
        out.append(dict(occupation=(hist/(t1-t0)).tolist(), residence=comfort/(t1-t0),
                        rms_K=math.sqrt(energy/(t1-t0))))
    return out


def boundary_checks():
    folder = ROOT/'results/phase1-boundary-stationarity-2026-10-04'
    records = [json.loads(line) for line in (folder/'trial-blocks.jsonl').read_text().splitlines()]
    design = json.loads((folder/'design.json').read_text())
    summaries = json.loads((folder/'summary.json').read_text())
    manifest = json.loads((folder/'manifest.json').read_text())
    for path, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected
    assert len(records) == 192
    replays = []
    for summary in summaries:
        L = summary['length_m']; selected = [r for r in records if r['length_m'] == L]
        for group in ['response', 'no_response']:
            hist = np.array([[b['occupation'] for b in r[group]] for r in selected])
            res = np.array([[b['residence'] for b in r[group]] for r in selected])
            rms = np.array([[b['rms_K'] for b in r[group]] for r in selected])
            assert np.max(np.abs(hist.sum(axis=2)-1)) < 1e-9
            entry = summary['groups'][group]
            assert np.max(np.abs(hist.mean(axis=0)-entry['mean_occupation_by_block'])) < 1e-15
            assert np.max(np.abs(res.mean(axis=0)-[x['mean'] for x in entry['residence_by_block']])) < 1e-15
            assert np.max(np.abs(rms.mean(axis=0)-[x['mean'] for x in entry['rms_K_by_block']])) < 1e-15
            for j in range(2):
                expected = np.abs(hist[:,j+1].mean(axis=0)-hist[:,j].mean(axis=0)).sum()/2
                assert abs(expected-entry['adjacent_block'][j]['TV']) < 1e-15
            for j in range(3):
                cold = [r[group][j] for r in selected if r['initial_offset_m'] < 0]
                hot = [r[group][j] for r in selected if r['initial_offset_m'] > 0]
                expected = np.abs(np.mean([b['occupation'] for b in cold], axis=0)-np.mean([b['occupation'] for b in hot], axis=0)).sum()/2
                assert abs(expected-entry['initial_side'][j]['TV']) < 1e-15
        for offset in design['center_relative_initial_offsets_m']:
            item = next(r for r in selected if r['initial_offset_m'] == offset and r['replicate'] == 0)
            for group, gain in [('response', 4.), ('no_response', 0.)]:
                cfg = Phase1ExperimentConfig.from_dict(item['config'])
                assert abs(cfg.point_model.initial_x_m-L/2-offset) < 1e-15
                cfg = replace(cfg, thermotaxis_controller=replace(cfg.thermotaxis_controller, response_gain_per_K_s=gain))
                result = run_phase1_experiment(cfg)
                actual = independent_blocks(result, cfg, design['late_block_edges_s'], design['histogram_bins'])
                for a, b in zip(actual, item[group]):
                    assert np.max(np.abs(np.array(a['occupation'])-b['occupation'])) < 1e-10
                    assert abs(a['residence']-b['residence']) < 1e-10
                    assert abs(a['rms_K']-b['rms_K']) < 1e-10
                replays.append(dict(length_m=L, offset_m=offset, group=group, replicate=0))
    scalar_count, tv_count = bootstrap_checks(records, summaries, design)
    return dict(pairs_checked=len(records), blocks_checked=len(records)*2*3,
                scalar_bootstrap_intervals_recomputed=scalar_count,
                tv_resampling_ranges_recomputed=tv_count, independent_full_core_replays=replays)


def bootstrap_checks(records, summaries, design):
    """Recompute intervals in saved RNG order; whole trajectories stay intact."""
    rng = np.random.default_rng(design['bootstrap_seed'])
    scalar_count = tv_count = 0

    def scalar(values, saved):
        nonlocal scalar_count
        values = np.asarray(values)
        ids = rng.integers(0, len(values), (design['bootstrap_resamples'], len(values)))
        interval = np.percentile(np.mean(values[ids], axis=1), [2.5, 97.5])
        assert np.allclose(interval, saved['bootstrap95'], atol=1e-14, rtol=0)
        scalar_count += 1

    for summary in summaries:
        rows = [r for r in records if r['length_m'] == summary['length_m']]
        assert len(rows) == 64
        assert all(r['initial_offset_m'] < 0 for r in rows[:32])
        assert all(r['initial_offset_m'] > 0 for r in rows[32:])
        for group in ['response', 'no_response']:
            entry = summary['groups'][group]
            hist = np.array([[b['occupation'] for b in r[group]] for r in rows])
            res = np.array([[b['residence'] for b in r[group]] for r in rows])
            rms = np.array([[b['rms_K'] for b in r[group]] for r in rows])
            for j in range(3): scalar(res[:,j], entry['residence_by_block'][j])
            for j in range(3): scalar(rms[:,j], entry['rms_K_by_block'][j])
            ids = rng.integers(0, 64, (4000, 64))
            for j in range(2):
                differences = hist[:,j+1]-hist[:,j]
                norms = np.abs(np.mean(differences[ids], axis=1)).sum(axis=1)/2
                assert np.allclose(np.percentile(norms, [2.5, 97.5]), entry['adjacent_block'][j]['TV_bootstrap95'], atol=1e-14, rtol=0)
                tv_count += 1
                scalar(res[:,j+1]-res[:,j], entry['adjacent_block'][j]['residence_difference'])
            cold_ids = rng.integers(0, 32, (4000, 32))
            hot_ids = rng.integers(0, 32, (4000, 32))
            for j in range(3):
                differences = hist[:32,j][cold_ids].mean(axis=1)-hist[32:,j][hot_ids].mean(axis=1)
                norms = np.abs(differences).sum(axis=1)/2
                assert np.allclose(np.percentile(norms, [2.5, 97.5]), entry['initial_side'][j]['TV_bootstrap95'], atol=1e-14, rtol=0)
                means = res[:32,j][cold_ids].mean(axis=1)-res[32:,j][hot_ids].mean(axis=1)
                assert np.allclose(np.percentile(means, [2.5, 97.5]), entry['initial_side'][j]['cold_minus_hot_residence']['bootstrap95'], atol=1e-14, rtol=0)
                tv_count += 1; scalar_count += 1
        for j in range(3):
            scalar([r['response'][j]['residence']-r['no_response'][j]['residence'] for r in rows], summary['paired_residence_difference_by_block'][j])
    return scalar_count, tv_count


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = dict(sampling=sampling_checks(), boundary=boundary_checks(),
                   validation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'independent-recomputation.json').write_text(json.dumps(results, indent=2)+'\n')
    print('Validated sampling records and actual-core replays; boundary blocks and 12 full-core replays.')


if __name__ == '__main__':
    main()
