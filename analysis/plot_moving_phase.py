"""Plot phase replication with fixed mean radial field and explicit startup."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path('results/tracking-moving-phase-2026-10-24')
    report=json.loads((root/'results.json').read_text())
    assert len(report['runs'])==544 and all(report['source_hashes_unchanged'].values())
    out=Path('docs/reports/assets/moving-phase-2026-10-24');out.mkdir(parents=True,exist_ok=True)
    phases=report['design']['phases_rad'];conditions=['wave_005_60','wave_010_120']
    groups=['real8','rewired','gru16','expert'];colors=['#9868a6','#73818b','#377fa8','#bb913e']
    labels=['Real graph (71)','Rewired graphs (71)','GRU16 (1121)','Engineering expert']
    summary=dict(status='frozen_phase_replication',rows=544,conditions={},initialization_check={})
    # Fixed physical mean radius and equal initial center error across phases.
    errors={};means=[]
    for r in report['runs']:
        key=(r['model'],r['training_seed'],r['condition'],r['field'],r['seed'])
        a=.05 if r['condition']=='wave_005_60' else .1
        state=r['initial_state']
        if r['field']=='radial':
            error=40*(np.hypot(state['x_m'],state['y_m'])-.005)+a*np.sin(r['phase_rad'])
            means.append(r['mean_target_radius_m'])
        else:
            angle=state['field_angle_rad'];error=40*(state['x_m']*np.cos(angle)+state['y_m']*np.sin(angle))
        errors.setdefault(key,[]).append(float(error))
    maximum=max(max(v)-min(v) for v in errors.values())
    assert maximum<1e-12
    assert max(abs(x-.005) for x in means)<1e-12
    summary['initialization_check']=dict(maximum_phase_difference_center_error_K=maximum,
        mean_radial_radius_m=.005,min_instantaneous_radius_m=.0025,
        note='Initial headings relative to geometric tangent are shared; relative-to-front initial heading demand changes with phase.')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for j,condition in enumerate(conditions):
        summary['conditions'][condition]={}
        for i,field in enumerate(['linear','radial']):
            ax=axes[i,j];table={}
            for name,label,color in zip(groups,labels,colors):
                y=[];low=[];high=[];late=[]
                for phase in phases:
                    rows=[r for r in report['runs'] if r['condition']==condition and r['field']==field and abs(r['phase_rad']-phase)<1e-9 and
                          (r['model'].startswith('rewired') if name=='rewired' else r['model']==name)]
                    pairs=sorted({(r['model'],str(r['training_seed'])) for r in rows})
                    vals=[np.mean([r['metrics']['moving_tracking_fraction'] for r in rows if (r['model'],str(r['training_seed']))==p]) for p in pairs]
                    y.append(float(np.mean(vals)));low.append(min(vals));high.append(max(vals))
                    late.append(float(np.mean([r['late_metrics']['moving_tracking_fraction'] for r in rows])))
                table[name]=dict(full_tracking_by_phase=y,late_tracking_by_phase=late,
                                all_phase_mean=float(np.mean(y)),late_all_phase_mean=float(np.mean(late)))
                ax.plot(range(4),np.array(y)*100,'-o',label=label,color=color,lw=1.5,markersize=4)
                ax.fill_between(range(4),np.array(low)*100,np.array(high)*100,color=color,alpha=.08)
            summary['conditions'][condition][field]=table
            ax.set_xticks(range(4),['0','pi/2','pi','3pi/2'])
            ax.set(ylim=(0,105),xlabel='Global temperature-wave phase',ylabel='Moving-frame tracking (%)',
                   title=field.capitalize()+(' / 0.05K, 60s' if j==0 else ' / 0.10K, 120s'))
            ax.grid(alpha=.15);ax.legend(fontsize=7,frameon=False,loc='lower left')
    summary['pooled']={}
    for condition in conditions:
        summary['pooled'][condition]={}
        for name in ['real8','rewired8-11','rewired8-22','rewired8-33','gru16','expert']:
            rows=[r for r in report['runs'] if r['condition']==condition and r['model']==name]
            summary['pooled'][condition][name]=dict(full=float(np.mean([r['metrics']['moving_tracking_fraction'] for r in rows])),
                                                 late=float(np.mean([r['late_metrics']['moving_tracking_fraction'] for r in rows])))
    fig.suptitle('Frozen phase replication: initial center error matched; physical mean radial radius fixed',fontsize=12)
    fig.supxlabel('Radial starting radius follows the instantaneous isotherm; bands are observed training / graph ranges, not confidence intervals.',fontsize=9)
    fig.savefig(out/'moving-phase.png',dpi=180);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(out,flush=True)


if __name__=='__main__':main()
