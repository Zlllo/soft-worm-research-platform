"""Continuous constant-error-slope surrogate, separate from the finite-domain core.

Exact exponential memory between Poisson-thinning proposals; no time step,
noise, deadband, clipping, absolute-error cusp or boundary. This checks only
the local weak-response coefficient derived in docs/phase1-memory-scaling.md.
Run: python3 analysis/phase1_weak_response.py
"""
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reports/assets/phase1-theory-2026-10-04'
DESIGN = dict(speed_m_s=.0002, error_slope_K_m=100., baseline_rate_s=.1,
              fast_tau_s=.5, slow_tau_s=5., gains_K_s=[0., .025, .05, .1],
              replicates=64, burn_s=500., measurement_s=30000., seed_base=740000)


def trajectory(gain, seed, design=DESIGN):
    v, b, rate = design['speed_m_s'], design['error_slope_K_m'], design['baseline_rate_s']
    tf, ts = design['fast_tau_s'], design['slow_tau_s']
    # |m_i-e| <= v|b|tau_i from bounded input slope, initialized at zero.
    envelope = rate + gain*v*abs(b)*(tf+ts)
    rng = random.Random(seed)
    direction = 1 if rng.random() < .5 else -1
    zf = zs = 0.
    elapsed = displacement = 0.
    burn, finish = design['burn_s'], design['burn_s']+design['measurement_s']
    proposals = accepted = 0
    while elapsed < finish:
        dt = rng.expovariate(envelope)
        # Split burn endpoint exactly without discarding the proposal event.
        end = min(elapsed+dt, finish)
        measured_dt = max(0., end-max(elapsed, burn))
        displacement += v*direction*measured_dt
        span = end-elapsed
        af, ass = -v*b*direction*tf, -v*b*direction*ts
        zf = af+(zf-af)*math.exp(-span/tf)
        zs = ass+(zs-ass)*math.exp(-span/ts)
        elapsed = end
        if elapsed >= finish:
            break
        actual_rate = rate-gain*(zs-zf)
        if not 0. <= actual_rate <= envelope+1e-14:
            raise ValueError('thinning bound violated: reduce gain for unbounded linear response')
        proposals += 1
        if rng.random() < actual_rate/envelope:
            direction *= -1
            accepted += 1
    return dict(seed=seed, drift_m_s=displacement/design['measurement_s'],
                proposals=proposals, accepted=accepted)


def predicted_coefficient(design=DESIGN):
    v, b, rate = design['speed_m_s'], design['error_slope_K_m'], design['baseline_rate_s']
    tf, ts = design['fast_tau_s'], design['slow_tau_s']
    return -v*v*b*(ts-tf)/(rate*(1+2*rate*ts)*(1+2*rate*tf))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Save the fixed design before drawing any trajectories.
    (OUT/'weak-response-design.json').write_text(json.dumps(DESIGN, indent=2)+'\n')
    groups = []
    for index, gain in enumerate(DESIGN['gains_K_s']):
        records = [trajectory(gain, DESIGN['seed_base']+index*1000+i)
                   for i in range(DESIGN['replicates'])]
        values = [r['drift_m_s'] for r in records]
        mean, se = statistics.mean(values), statistics.stdev(values)/math.sqrt(len(values))
        theory = gain*predicted_coefficient()
        groups.append(dict(gain_K_s=gain, predicted_first_order_m_s=theory,
                           mean_drift_m_s=mean, standard_error_m_s=se,
                           normal95_m_s=[mean-1.96*se, mean+1.96*se],
                           discrepancy_in_standard_errors=(mean-theory)/se,
                           records=records))
        print(f'gain={gain}: mean={mean:.8g} m/s, SE={se:.3g}, first order={theory:.8g}', flush=True)
    result = dict(classification='continuous constant-slope weak-response surrogate; not finite-domain core',
                  created_utc=datetime.now(timezone.utc).isoformat(), design=DESIGN,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  theoretical_coefficient_m_K=predicted_coefficient(), groups=groups,
                  limits=['Normal interval across independent trajectory means, 64 replicates.',
                          'No paired inference across gains; distinct seed blocks.',
                          'Constant slope extends the local linear error; it is not global absolute temperature error.',
                          'Agreement at small gains does not validate strong-response reference parameters.'])
    (OUT/'weak-response.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
