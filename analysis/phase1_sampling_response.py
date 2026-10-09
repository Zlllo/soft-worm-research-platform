"""Finite sampling/control local weak response; no global domain, cusp or noise.

Run from repository: PYTHONPATH=src python3 analysis/phase1_sampling_response.py
The lag update exactly collapses sensor updates within a fixed-direction control
interval. Core calls are independently compared in contract_gate(); simulation
uses numpy PCG64 random streams and the core's Bernoulli probability convention.
"""
from pathlib import Path
import hashlib
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from thermotaxis.phase1 import TwoTimescaleTurnController
from thermotaxis.phase1_config import MemoryControllerConfig
from thermotaxis.contracts import SensorReading

CONFIG = ROOT/'configs/phase1_sampling_response.json'
OUT = ROOT/'docs/reports/assets/phase1-sampling-2026-10-04'

def factors(d, h):
    hs=d['sensor_interval_s']
    q=np.exp(-hs/np.array([d['fast_tau_s'], d['slow_tau_s']]))
    Q=q**round(h/hs)
    B=hs*q*(1-Q)/(1-q)
    return Q,B

def coefficient(d,h):
    Q,B=factors(d,h)
    exp=math.exp(-d['baseline_rate_s']*h)
    r=2*exp-1
    return -2*d['speed_m_s']**2*d['error_slope_K_m']*h*exp/(1-r)*(B[1]/(1-Q[1]*r)-B[0]/(1-Q[0]*r))

def continuous_coefficient(d):
    tf,ts,rate=d['fast_tau_s'],d['slow_tau_s'],d['baseline_rate_s']
    return -d['speed_m_s']**2*d['error_slope_K_m']*(ts-tf)/(rate*(1+2*rate*ts)*(1+2*rate*tf))

def contract_gate(d):
    """Calls real observe/should_turn against lag+identical Python RNG draws.

    Temperature offset 1000 K keeps |T-T*| on its positive branch over 200 s.
    No mutation of controller memories or any core module.
    """
    records=[]
    for h in d['control_intervals_s']:
        for gain in d['gains_K_s']:
            cfg=MemoryControllerConfig(d['fast_tau_s'],d['slow_tau_s'],d['baseline_rate_s'],0.,1.,gain,0.)
            seed=960000+round(h*1000)+round(gain*100)
            controller=TwoTimescaleTurnController(cfg,0.,seed)
            rng=random.Random(seed)
            e=1000.; z=np.zeros(2); direction=1
            controller.observe(SensorReading(e,0.))
            Q,B=factors(d,h); maxdiff=0.; count=0
            for n in range(1,round(200/h)+1):
                before=direction
                for j in range(1,round(h/d['sensor_interval_s'])+1):
                    e+=d['speed_m_s']*d['error_slope_K_m']*before*d['sensor_interval_s']
                    state=controller.observe(SensorReading(e,(n-1)*h+j*d['sensor_interval_s']))
                z=Q*z-d['speed_m_s']*d['error_slope_K_m']*before*B
                expected_rate=d['baseline_rate_s']-gain*(z[1]-z[0])
                maxdiff=max(maxdiff,abs((state.slow_error_K-state.fast_error_K)-(z[1]-z[0])),abs(state.turn_rate_per_s-expected_rate))
                actual=controller.should_turn(h)
                expected=rng.random() < -math.expm1(-expected_rate*h)
                assert actual==expected
                if actual: direction*=-1
                count+=actual
            assert e>0 and maxdiff<1e-9
            records.append(dict(control_interval_s=h,gain_K_s=gain,core_turn_count=count,max_absolute_difference=maxdiff))
    return records

def trajectories(d,h,gain,seeds):
    count=round((d['burn_s']+d['measurement_s'])/h)
    burn=round(d['burn_s']/h)
    draws=np.empty((count,len(seeds)))
    direction=np.empty(len(seeds))
    for j,seed in enumerate(seeds):
        rng=np.random.default_rng(seed)
        direction[j]=1 if rng.random()<.5 else -1
        draws[:,j]=rng.random(count)
    zf=np.zeros(len(seeds)); zs=zf.copy(); displacement=zf.copy(); turns=np.zeros(len(seeds),dtype=int)
    Q,B=factors(d,h); vb=d['speed_m_s']*d['error_slope_K_m']
    minrate=math.inf; maxrate=-math.inf
    for n in range(count):
        if n>=burn: displacement+=d['speed_m_s']*h*direction
        zf=Q[0]*zf-vb*B[0]*direction
        zs=Q[1]*zs-vb*B[1]*direction
        rate=d['baseline_rate_s']-gain*(zs-zf)
        minrate=min(minrate,float(rate.min())); maxrate=max(maxrate,float(rate.max()))
        turn=draws[n] < -np.expm1(-rate*h)
        direction[turn]*=-1
        turns+=turn
    assert minrate>d['min_rate_s'] and maxrate<d['max_rate_s']
    return [dict(seed=int(seed),drift_m_s=float(value/d['measurement_s']),turn_count=int(t)) for seed,value,t in zip(seeds,displacement,turns)], [minrate,maxrate]

def main():
    d=json.loads(CONFIG.read_text()); OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'design.json').write_text(json.dumps(d,indent=2)+'\n')
    gate=contract_gate(d)
    (OUT/'core-contract-gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    # |z_i| <= v|b| B_i/(1-Q_i), so the rate always stays within these bounds.
    theoretical_bounds={}
    groups=[]
    for hi,h in enumerate(d['control_intervals_s']):
        Q,B=factors(d,h)
        amplitude=d['speed_m_s']*abs(d['error_slope_K_m'])*float(sum(B/(1-Q)))
        theoretical_bounds[str(h)]=[d['baseline_rate_s']-max(d['gains_K_s'])*amplitude,d['baseline_rate_s']+max(d['gains_K_s'])*amplitude]
        assert theoretical_bounds[str(h)][0]>0 and theoretical_bounds[str(h)][1]<1
        for gi,gain in enumerate(d['gains_K_s']):
            seeds=[d['seed_base']+(hi*len(d['gains_K_s'])+gi)*d['seed_block_stride']+i for i in range(d['replicates'])]
            records,bounds=trajectories(d,h,gain,seeds)
            values=[r['drift_m_s'] for r in records]
            mean=statistics.mean(values); se=statistics.stdev(values)/math.sqrt(len(values))
            prediction=gain*coefficient(d,h)
            group=dict(control_interval_s=h,gain_K_s=gain,coefficient_m_K=coefficient(d,h),first_order_drift_m_s=prediction,mean_drift_m_s=mean,standard_error_m_s=se,normal95_m_s=[mean-1.96*se,mean+1.96*se],discrepancy_standard_errors=(mean-prediction)/se,observed_rate_range_s=bounds,records=records)
            groups.append(group)
            print(f'h={h}, g={gain}: mean={mean:.8g}, prediction={prediction:.8g}, z={group["discrepancy_standard_errors"]:.3f}',flush=True)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),classification=d['classification'],design=d,continuous_coefficient_m_K=continuous_coefficient(d),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),core_sha256=hashlib.sha256((ROOT/'src/thermotaxis/phase1.py').read_bytes()).hexdigest(),design_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),core_contract_gate=gate,theoretical_rate_bounds_s=theoretical_bounds,groups=groups)
    (OUT/'sampling-response.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(result)

def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
    for ax,h in zip(axs,result['design']['control_intervals_s']):
        rows=[g for g in result['groups'] if g['control_interval_s']==h]
        gains=[g['gain_K_s'] for g in rows]
        ax.errorbar(gains,[g['mean_drift_m_s']*1000 for g in rows],yerr=[1.96*g['standard_error_m_s']*1000 for g in rows],fmt='o',capsize=4,label='Independent trajectories (95%)')
        ax.plot(gains,[g['first_order_drift_m_s']*1000 for g in rows],label='Finite-clock first order')
        ax.plot(gains,[g*result['continuous_coefficient_m_K']*1000 for g in gains],linestyle='--',label='Continuous first order')
        ax.set(title=f'Control interval {h:g} s',xlabel='Gain (1 / K / s)'); ax.axhline(0,color='grey',lw=.5)
    axs[0].set_ylabel('Local drift (mm / s)'); axs[-1].legend(fontsize=7)
    fig.suptitle('Constant-slope local surrogate; sensor interval 0.1 s')
    fig.tight_layout()
    for suffix in ['png','svg']: fig.savefig(OUT/f'sampling-response.{suffix}',dpi=180)
    plt.close(fig)

if __name__=='__main__': main()
