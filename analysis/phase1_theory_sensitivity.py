"""Fixed control-clock/window sensitivity with an exact discrete zero-gain baseline."""
from __future__ import annotations
import copy, hashlib, json, math, platform, subprocess, time
from dataclasses import replace
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thermotaxis.phase1 import run_phase1_experiment
from thermotaxis.phase1_config import Phase1ExperimentConfig
from statistics import NormalDist

def wilson(k,n):
    z=NormalDist().inv_cdf(.975); p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]

ROOT=Path(__file__).resolve().parents[1]
DESIGN=json.loads((ROOT/'configs/phase1_theory_sensitivity.json').read_text())
OUT=ROOT/'results/phase1-theory-sensitivity-2026-10-04'
REF=json.loads((ROOT/DESIGN['reference_config']).read_text())
METRICS=['reached','restricted_first_entry_s','comfort_residence_fraction','rms_temperature_error_K','minimum_rate_fraction','maximum_rate_fraction','deadband_fraction']
def dump(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def config(dt,duration,replicate=0):
    raw=copy.deepcopy(REF)
    raw['name']=f'theory-clock{dt}-window{duration}'
    raw['time'].update(physics_dt_s=DESIGN['physics_dt_s'],sensor_dt_s=DESIGN['sensor_dt_s'],control_dt_s=dt,duration_s=duration)
    for k,b in DESIGN['seed_bases'].items(): raw['seeds'][k]=b+replicate
    return Phase1ExperimentConfig.from_dict(raw)
def extract(result,cfg):
    m=result['metrics']; stride=round(cfg.time.control_dt_s/cfg.time.physics_dt_s)
    rows=result['trajectory'][stride:-1:stride]
    rates=np.array([r['turn_rate_per_s'] for r in rows])
    assert len(rates)==round(cfg.time.duration_s/cfg.time.control_dt_s)-1
    return dict(zip(METRICS,[float(m['entered_comfort_region']),float(m['first_entry_time_s']) if m['entered_comfort_region'] else cfg.time.duration_s,m['comfort_residence_fraction'],m['rms_temperature_error_K'],float(np.mean(rates==cfg.thermotaxis_controller.min_turn_rate_per_s)),float(np.mean(rates==cfg.thermotaxis_controller.max_turn_rate_per_s)),float(np.mean([abs(r['improvement_K'])<=cfg.thermotaxis_controller.response_deadband_K for r in rows]))])) | {'turn_count':m['turn_count'],'effective_control_decisions':len(rates)}
def summarize(records,rng):
    arrays={g:np.array([[r[g][m] for m in METRICS] for r in records]) for g in ['response','no_response']}
    n=len(records); delta=arrays['response']-arrays['no_response']
    boot=delta[rng.integers(0,n,size=(DESIGN['bootstrap_resamples'],n))].mean(axis=1)
    return {'groups':{g:{'means':dict(zip(METRICS,a.mean(axis=0).tolist())),'reached_wilson95':wilson(int(a[:,0].sum()),n)} for g,a in arrays.items()},'paired_difference':{m:{'mean':float(delta[:,j].mean()),'bootstrap95':np.quantile(boot[:,j],[.025,.975]).tolist()} for j,m in enumerate(METRICS)}}
def discrete_survival(dt,horizon,x0=.003,a=.0125,v=.0002,rate=.1):
    """Exact probability propagation on reachable lattice before first hit.

    Columns are post-control +/- velocities, rows x=n*v*dt. Reflection at zero
    gives outward velocity before Bernoulli flipping. Crossing absorbs before
    control; event time includes fractional final segment.
    """
    dx=v*dt; ratio=a/dx; size=int(math.ceil(ratio-1e-10)); xindex=round(x0/dx)
    if abs(xindex*dx-x0)>1e-12: raise ValueError('Initial state off lattice')
    p=-math.expm1(-rate*dt); state=np.zeros((size,2));state[xindex]=.5
    events=[]; surv=[1.]; times=[0.]; rmst=0.; reached=0.
    fraction=ratio-(size-1)
    for k in range(1,round(horizon/dt)+1):
        absorbed=float(state[-1,0]); event_time=(k-1+fraction)*dt
        # Integral survival: each hit occurs inside its final movement interval.
        rmst+=dt*(1-reached)-absorbed*dt*(1-fraction)
        reached+=absorbed
        moved=np.zeros_like(state)
        moved[1:,0]+=state[:-1,0]
        moved[1,0]+=state[0,1] # Immediate reflection of inward state at zero.
        moved[:-1,1]+=state[1:,1]
        moved[0,0]+=moved[0,1];moved[0,1]=0. # Arrival at wall reflects.
        state=np.column_stack(((1-p)*moved[:,0]+p*moved[:,1],p*moved[:,0]+(1-p)*moved[:,1]))
        if absorbed:
            times.append(event_time);surv.append(float(state.sum()))
            events.append([event_time,absorbed])
        if abs(state.sum()+reached-1)>2e-11: raise AssertionError('Probability not conserved')
    times.append(horizon);surv.append(float(state.sum()))
    return {'time_s':times,'survival_probability':surv,'arrival_probability':reached,'rmst_s':rmst,'event_masses':events,'turn_probability':p,'lattice_spacing_m':dx,'transient_states':2*size}
def main():
    OUT.mkdir(parents=True,exist_ok=True);dump(OUT/'design.json',DESIGN)
    start=time.monotonic();rng=np.random.default_rng(DESIGN['bootstrap_seed']);summaries={};allrows=[]
    # Reconfirm acceleration against default step for both extreme control clocks.
    gate=[]
    for dt in [.1,1.]:
        cfg=config(dt,90.,999)
        for gain in [0.,cfg.thermotaxis_controller.response_gain_per_K_s]:
            cfg=replace(cfg,thermotaxis_controller=replace(cfg.thermotaxis_controller,response_gain_per_K_s=gain))
            outputs=[run_phase1_experiment(replace(cfg,time=replace(cfg.time,physics_dt_s=h)),record_trajectory=False)['metrics'] for h in [.01,.1]]
            diff=max(abs(float(outputs[0][m])-float(outputs[1][m])) for m in ['first_entry_time_s','comfort_residence_fraction','rms_temperature_error_K','turn_count'] if outputs[0][m] is not None)
            assert outputs[0]['entered_comfort_region']==outputs[1]['entered_comfort_region'] and diff<1e-8
            gate.append({'control_dt_s':dt,'gain':gain,'max_metric_difference':diff})
    dump(OUT/'numerical-gate.json',gate)
    for dt in DESIGN['control_dt_s']:
        for window in DESIGN['duration_s']:
            name=f'clock-{dt}-window-{window}';records=[];cfg=config(dt,window)
            for i in range(DESIGN['paired_replicates_per_condition']):
                c=config(dt,window,i);pair={'replicate':i,'seeds':{k:b+i for k,b in DESIGN['seed_bases'].items()}}
                for g,gain in [('response',c.thermotaxis_controller.response_gain_per_K_s),('no_response',0.)]:
                    cg=replace(c,thermotaxis_controller=replace(c.thermotaxis_controller,response_gain_per_K_s=gain))
                    pair[g]=extract(run_phase1_experiment(cg),cg)
                records.append(pair)
            folder=OUT/name;folder.mkdir(exist_ok=True);dump(folder/'config.resolved.json',cfg.to_dict());dump(folder/'paired-records.json',records)
            summary=summarize(records,rng);summary.update(control_dt_s=dt,duration_s=window,replicates=len(records),zero_gain_exact=discrete_survival(dt,window))
            dump(folder/'statistics.json',summary);summaries[name]=summary
            allrows.extend({'condition':name,**r} for r in records)
            print(f'{name}: {len(records)} pairs, elapsed {time.monotonic()-start:.1f}s',flush=True)
    dump(OUT/'summary.json',summaries)
    # Window-prefix identity check from complete aggregate metrics of all trials.
    for dt in DESIGN['control_dt_s']:
        for i in range(DESIGN['paired_replicates_per_condition']):
            rows=[next(r for r in allrows if r['condition']==f'clock-{dt}-window-{w}' and r['replicate']==i) for w in DESIGN['duration_s']]
            for g in ['response','no_response']:
                entries=[r[g]['restricted_first_entry_s'] for r in rows]
                assert abs(entries[0]-min(entries[-1],90.))<1e-8 and abs(entries[1]-min(entries[-1],180.))<1e-8
    dump(OUT/'manifest.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'core_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src/thermotaxis').glob('*.py'))},'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'python':platform.python_version(),'design':DESIGN,'elapsed_s':time.monotonic()-start,'prefix_first_entry_check':'all 512 seed/clock combinations passed','command':'PYTHONPATH=src MPLCONFIGDIR=/private/tmp/softworm-mpl python3 analysis/phase1_theory_sensitivity.py'})
    plot(summaries)
def plot(summaries):
    fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,metric,title in zip(axs,METRICS[1:4],['Paired restricted entry-time difference','Paired comfort residence difference','Paired RMS error difference']):
        for w,color in zip(DESIGN['duration_s'],['#0072B2','#D55E00','#009E73']):
            ds=[summaries[f'clock-{dt}-window-{w}']['paired_difference'][metric] for dt in DESIGN['control_dt_s']]
            means=np.array([d['mean'] for d in ds]);low=np.array([d['bootstrap95'][0] for d in ds]);high=np.array([d['bootstrap95'][1] for d in ds])
            ax.errorbar(DESIGN['control_dt_s'],means,yerr=[means-low,high-means],marker='o',color=color,label=f'{w:g} s window',capsize=3)
        ax.axhline(0,color='gray',linewidth=.7);ax.set(xlabel='Control interval (s)',ylabel={'restricted_first_entry_s':'Response minus control (s)','comfort_residence_fraction':'Response minus control (fraction)','rms_temperature_error_K':'Response minus control (K)'}[metric],title=title);ax.legend(fontsize=8)
    for ext in ['png','svg']:fig.savefig(OUT/f'clock-window-sensitivity.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for dt,color in zip(DESIGN['control_dt_s'],['#0072B2','#D55E00','#009E73','#CC79A7']):
        s=summaries[f'clock-{dt}-window-360.0'];b=s['zero_gain_exact'];axs[0].step(b['time_s'],b['survival_probability'],where='post',label=f'dt={dt:g} s',color=color)
        r=s['groups']['no_response']['means']['reached'];ci=s['groups']['no_response']['reached_wilson95'];axs[1].errorbar(dt,r,yerr=[[r-ci[0]],[ci[1]-r]],marker='o',color=color,capsize=3)
        axs[1].plot(dt,b['arrival_probability'],'x',color='black')
    axs[0].set(xlabel='Time (s)',ylabel='Exact discrete survival',title='Zero-response baseline');axs[0].legend();axs[1].set(xlabel='Control interval (s)',ylabel='Arrival probability by 360 s',title='Simulation (95% Wilson) and exact baseline (x)')
    for ext in ['png','svg']:fig.savefig(OUT/f'zero-response-baseline.{ext}',dpi=180)
    plt.close(fig)
if __name__=='__main__':main()
