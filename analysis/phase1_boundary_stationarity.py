"""Boundary-distance and finite-time occupation diagnostics; original core unchanged."""
from __future__ import annotations
import copy, hashlib, json, platform, subprocess, time
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thermotaxis.phase1 import run_phase1_experiment, _advance_reflecting
from thermotaxis.phase1_config import Phase1ExperimentConfig
ROOT=Path(__file__).resolve().parents[1]
DESIGN=json.loads((ROOT/'configs/phase1_boundary_stationarity.json').read_text())
REF=json.loads((ROOT/DESIGN['reference_config']).read_text())
OUT=ROOT/'results/phase1-boundary-stationarity-2026-10-04'
def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def config(length,offset,index,duration=None):
    raw=copy.deepcopy(REF); raw['name']=f'boundary-{length}-{offset}'
    raw['domain']['x_extent_m']=length
    raw['temperature']['reference_K']=raw['temperature']['preferred_K']-100*length/2
    raw['point_model']['initial_x_m']=length/2+offset
    raw['time'].update(physics_dt_s=DESIGN['physics_dt_s'],duration_s=duration or DESIGN['duration_s'])
    for k,b in DESIGN['seed_bases'].items(): raw['seeds'][k]=b+index
    return Phase1ExperimentConfig.from_dict(raw)
def segment_histogram(starts,ends,length,bins,speed):
    """Exact residence seconds in bins for linear segments (at most one crossed edge)."""
    starts=np.asarray(starts); ends=np.asarray(ends)
    lo=np.minimum(starts,ends); hi=np.maximum(starts,ends); dx=length/bins
    il=np.clip(np.floor(lo/dx).astype(int),0,bins-1)
    ih=np.clip(np.floor(hi/dx).astype(int),0,bins-1)
    if np.any(ih-il>1): raise ValueError('Segment wider than histogram bin')
    same=il==ih; weights=np.zeros(bins)
    weights+=np.bincount(il[same],weights=(hi-lo)[same]/speed,minlength=bins)
    cross=~same; boundary=(il[cross]+1)*dx
    weights+=np.bincount(il[cross],weights=(boundary-lo[cross])/speed,minlength=bins)
    weights+=np.bincount(ih[cross],weights=(hi[cross]-boundary)/speed,minlength=bins)
    return weights

def blocks(result,cfg,edges):
    rows=result['trajectory']; dt=cfg.time.physics_dt_s; v=cfg.point_model.speed_m_s; L=cfg.domain.x_extent_m
    positions=np.array([r['position_m'] for r in rows]); directions=np.array([r['direction'] for r in rows])
    out=[]
    for t0,t1 in zip(edges[:-1],edges[1:]):
        i0=round(t0/dt);i1=round(t1/dt)
        starts=positions[i0:i1];ends=positions[i0+1:i1+1]
        reflect=np.abs(np.abs(ends-starts)-v*dt)>1e-12
        hs=starts[~reflect].tolist();he=ends[~reflect].tolist()
        for j in np.flatnonzero(reflect):
            _,_,segments=_advance_reflecting(starts[j],int(directions[i0+j]),v*dt,L)
            for a,b in segments: hs.append(a);he.append(b)
        hist=segment_histogram(hs,he,L,DESIGN['histogram_bins'],v)/(t1-t0)
        if abs(hist.sum()-1)>1e-9: raise AssertionError('Occupation mass not conserved')
        a=np.minimum(hs,he);b=np.maximum(hs,he);center=L/2; half=.0025
        comfort=np.maximum(0,np.minimum(b,center+half)-np.maximum(a,center-half)).sum()/v/(t1-t0)
        e0=100*(np.asarray(hs)-center);e1=100*(np.asarray(he)-center)
        duration=np.abs(np.asarray(he)-hs)/v
        rms=np.sqrt(np.sum(duration*(e0*e0+e0*e1+e1*e1)/3)/(t1-t0))
        out.append({'start_s':t0,'end_s':t1,'occupation':hist.tolist(),'residence':float(comfort),'rms_K':float(rms)})
    return out

def scalar_ci(values,rng):
    x=np.asarray(values); boot=x[rng.integers(0,len(x),(DESIGN['bootstrap_resamples'],len(x)))].mean(axis=1)
    return {'mean':float(x.mean()),'bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
def tv(a,b): return float(np.abs(np.asarray(a)-np.asarray(b)).sum()/2)
def summarize(records):
    rng=np.random.default_rng(DESIGN['bootstrap_seed']);summaries=[]
    for L in DESIGN['domain_lengths_m']:
        s={'length_m':L,'uniform_residence':.005/L,'uniform_rms_K':100*L/(12**.5),'groups':{},'paired_residence_difference_by_block':[]}
        rr=[r for r in records if r['length_m']==L]
        for group in ['response','no_response']:
            hist=np.array([[b['occupation'] for b in r[group]] for r in rr]); res=np.array([[b['residence'] for b in r[group]] for r in rr]); rms=np.array([[b['rms_K'] for b in r[group]] for r in rr])
            mean=hist.mean(axis=0); cold=hist[:32];hot=hist[32:]
            entry={'cold_mean_occupation_by_block':cold.mean(axis=0).tolist(),'hot_mean_occupation_by_block':hot.mean(axis=0).tolist(),'cold_uniform_TV_by_block':[tv(x,np.ones(len(x))/len(x)) for x in cold.mean(axis=0)],'hot_uniform_TV_by_block':[tv(x,np.ones(len(x))/len(x)) for x in hot.mean(axis=0)],'mean_occupation_by_block':mean.tolist(),'residence_by_block':[scalar_ci(res[:,j],rng) for j in range(3)],'rms_K_by_block':[scalar_ci(rms[:,j],rng) for j in range(3)],'adjacent_block':[],'initial_side':[],'uniform_TV_by_block':[tv(x,np.ones(len(x))/len(x)) for x in mean]}
            idx=rng.integers(0,64,(4000,64))
            for j in range(2):
                dif=hist[:,j+1]-hist[:,j];bt=np.abs(dif[idx].mean(axis=1)).sum(axis=1)/2
                entry['adjacent_block'].append({'TV':tv(mean[j+1],mean[j]),'TV_bootstrap95':np.quantile(bt,[.025,.975]).tolist(),'residence_difference':scalar_ci(res[:,j+1]-res[:,j],rng)})
            ic=rng.integers(0,32,(4000,32));ih=rng.integers(0,32,(4000,32))
            for j in range(3):
                bt=np.abs(cold[ic,j].mean(axis=1)-hot[ih,j].mean(axis=1)).sum(axis=1)/2
                dr=res[:32,j][ic].mean(axis=1)-res[32:,j][ih].mean(axis=1)
                entry['initial_side'].append({'TV':tv(cold[:,j].mean(axis=0),hot[:,j].mean(axis=0)),'TV_bootstrap95':np.quantile(bt,[.025,.975]).tolist(),'cold_minus_hot_residence':{'mean':float(res[:32,j].mean()-res[32:,j].mean()),'bootstrap95':np.quantile(dr,[.025,.975]).tolist()}})
            
            th=DESIGN['diagnostic_thresholds']
            for diagnostic in entry['adjacent_block']:
                ci=diagnostic['residence_difference']['bootstrap95']
                diagnostic['within_prespecified_thresholds']=diagnostic['TV_bootstrap95'][1]<th['adjacent_block_TV'] and max(abs(ci[0]),abs(ci[1]))<th['absolute_residence_difference']
            for diagnostic in entry['initial_side']:
                ci=diagnostic['cold_minus_hot_residence']['bootstrap95']
                diagnostic['within_prespecified_thresholds']=diagnostic['TV_bootstrap95'][1]<th['initial_side_TV'] and max(abs(ci[0]),abs(ci[1]))<th['absolute_residence_difference']
            s['groups'][group]=entry
        for j in range(3): s['paired_residence_difference_by_block'].append(scalar_ci([r['response'][j]['residence']-r['no_response'][j]['residence'] for r in rr],rng))
        summaries.append(s)
    return summaries

def main():
    OUT.mkdir(parents=True,exist_ok=True);dump(OUT/'design.json',DESIGN);start=time.monotonic();gate=[]
    for L in [.02,.06]:
        cfg=config(L,-.007,99999,120)
        for gain in [0,4]:
            cfg=replace(cfg,thermotaxis_controller=replace(cfg.thermotaxis_controller,response_gain_per_K_s=gain))
            pair=[]
            for dt in [.01,.1]:
                cc=replace(cfg,time=replace(cfg.time,physics_dt_s=dt));r=run_phase1_experiment(cc)
                pair.append({'metrics':r['metrics'],'blocks':blocks(r,cc,[0,60,120])})
            metric_diff=max(abs(float(pair[0]['metrics'][k])-float(pair[1]['metrics'][k])) for k in ['comfort_residence_fraction','rms_temperature_error_K','turn_count'])
            histdiff=max(np.max(np.abs(np.array(a['occupation'])-b['occupation'])) for a,b in zip(pair[0]['blocks'],pair[1]['blocks']))
            assert metric_diff<1e-8 and histdiff<1e-8
            gate.append({'length_m':L,'gain':gain,'max_metric_difference':metric_diff,'max_bin_occupation_difference':float(histdiff)})
    dump(OUT/'numerical-gate.json',gate); records=[]
    for li,L in enumerate(DESIGN['domain_lengths_m']):
        for si,offset in enumerate(DESIGN['center_relative_initial_offsets_m']):
            for rep in range(DESIGN['paired_replicates_per_initial_side']):
                index=(li*2+si)*DESIGN['condition_seed_stride']+rep;cfg=config(L,offset,index)
                item={'length_m':L,'initial_offset_m':offset,'replicate':rep,'seeds':cfg.to_dict()['seeds'],'config':cfg.to_dict()}
                for group,gain in [('response',4),('no_response',0)]:
                    cc=replace(cfg,thermotaxis_controller=replace(cfg.thermotaxis_controller,response_gain_per_K_s=gain));r=run_phase1_experiment(cc)
                    item[group]=blocks(r,cc,DESIGN['late_block_edges_s']);item[group+'_whole_metrics']=r['metrics'];del r
                records.append(item)
                with (OUT/'trial-blocks.jsonl').open('a' if len(records)>1 else 'w') as f: f.write(json.dumps(item)+'\n')
                if (rep+1)%8==0: print(f'{L:.3f} {offset:+.3f} {rep+1}/32 elapsed={time.monotonic()-start:.1f}s',flush=True)
    summary=summarize(records);dump(OUT/'summary.json',summary)
    fig,axes=plt.subplots(2,3,figsize=(13,6),sharex=True)
    for col,s in enumerate(summary):
        L=s['length_m']
        for row,g in enumerate(['response','no_response']):
            ax=axes[row,col]
            for j,h in enumerate(s['groups'][g]['mean_occupation_by_block']):
                edges=np.linspace(-L/2*1000,L/2*1000,61);density=np.array(h)/(L/60*1000)
                ax.stairs(density,edges,label=f'{1800+j*600}–{2400+j*600} s',alpha=.8)
            ax.axvspan(-2.5,2.5,color='green',alpha=.1);ax.set_title(f'{g.replace("_"," ")} | L={L*1000:.0f} mm')
            ax.set_ylabel('Occupation density (1/mm)');ax.grid(alpha=.2)
            if row: ax.set_xlabel('Position relative to comfort center (mm)')
    axes[0,0].legend(fontsize=8);fig.tight_layout()
    for suffix in ['png','svg']:fig.savefig(OUT/f'late-occupation.{suffix}',dpi=180)
    plt.close(fig)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'configs/phase1_boundary_stationarity.json',*sorted((ROOT/'src/thermotaxis').glob('*.py'))]}
    dump(OUT/'manifest.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':sources,'python':platform.python_version(),'runtime_s':time.monotonic()-start,'pairs':len(records),'command':'PYTHONPATH=src python3 analysis/phase1_boundary_stationarity.py','histogram_method':'Exact constant-direction movement-segment time integration; wall crossings split using original core _advance_reflecting; bins normalized by block duration.'})
    print(f'Finished {len(records)} pairs in {time.monotonic()-start:.1f}s',flush=True)
if __name__=='__main__':main()
