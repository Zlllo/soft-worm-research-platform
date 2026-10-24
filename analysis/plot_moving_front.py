"""Moving-front transfer and metric validation, with all random graphs shown."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path('results/tracking-moving-graphs-2026-10-24')
    report=json.loads((root/'results.json').read_text())
    reference=json.loads(Path('results/tracking-moving-front-2026-10-24/results.json').read_text())
    assert len(report['runs'])==480 and all(report['source_hashes_unchanged'].values())
    out=Path('docs/reports/assets/moving-front-2026-10-24');out.mkdir(parents=True,exist_ok=True)
    conditions=['static','wave_005_60','wave_010_120']
    labels=['Static','0.05 K / 60 s','0.10 K / 120 s']
    names=['real8','rewired','gru3','gru16','expert']
    display=['Real graph (71)','Rewired graphs (71)','GRU3 (94)','GRU16 (1121)','Engineering expert']
    colors=['#9868a6','#73818b','#418d7f','#377fa8','#bb913e']
    summary=dict(status='frozen_zero_shot_transfer_not_training',rows=480,reused=168,new_rollouts=312,cohorts={})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,cohort in enumerate(['previous','fresh']):
        ax=axes[0,i];summary['cohorts'][cohort]={}
        for name,label,color in zip(names,display,colors):
            y=[];low=[];high=[];table={}
            for condition in conditions:
                rows=[r for r in report['runs'] if r['cohort']==cohort and r['condition']==condition and
                      (r['model'].startswith('rewired') if name=='rewired' else r['model']==name)]
                pairs=sorted({(r['model'],str(r['training_seed'])) for r in rows})
                means=[np.mean([r['moving_metrics']['moving_tracking_fraction'] for r in rows if (r['model'],str(r['training_seed']))==p]) for p in pairs]
                y.append(np.mean(means)*100);low.append(min(means)*100);high.append(max(means)*100)
                table[condition]=dict(tracking=float(np.mean(means)),range=[float(min(means)),float(max(means))],
                    center_error_rms_K=float(np.mean([r['moving_metrics']['center_error_rms_K'] for r in rows])),
                    per_graph={n:float(np.mean([r['moving_metrics']['moving_tracking_fraction'] for r in rows if r['model']==n])) for n in sorted({r['model'] for r in rows})})
            summary['cohorts'][cohort][name]=table
            ax.plot(range(3),y,'-o',color=color,label=label,lw=1.5,markersize=4)
            ax.fill_between(range(3),low,high,color=color,alpha=.07)
        ax.set_xticks(range(3),labels,fontsize=9)
        ax.set(ylim=(0,105),ylabel='Moving-frame tracking time (%)',title='Initial cases 9701/9702' if i==0 else 'Fresh cases 9901/9902')
        ax.grid(alpha=.15);ax.legend(fontsize=7,frameon=False,loc='lower left')
    ax=axes[1,0]
    before=[reference['reference_checks'][c]['metrics']['static_tangent_tracking_fraction']*100 for c in conditions[1:]]
    after=[reference['reference_checks'][c]['metrics']['moving_tracking_fraction']*100 for c in conditions[1:]]
    ax.bar(np.arange(2)-.18,before,.34,color='#9b8ca9',label='Old static tangent rule')
    ax.bar(np.arange(2)+.18,after,.34,color='#438e83',label='Relative-to-front rule')
    ax.set_xticks(range(2),labels[1:]);ax.set(ylim=(0,106),ylabel='Tracking time (%)',title='Exact zero-error kinematic reference')
    ax.legend(fontsize=8,frameon=False);ax.grid(axis='y',alpha=.15)
    ax=axes[1,1]
    for name,label,color in zip(names,display,colors):
        values=[summary['cohorts']['fresh'][name][c]['center_error_rms_K']*1000 for c in conditions]
        ax.plot(range(3),values,'-o',color=color,label=label,lw=1.5,markersize=4)
    ax.set_xticks(range(3),labels,fontsize=9);ax.set(ylabel='Center temperature RMSE (mK)',title='Fresh cases: temperature error')
    ax.grid(alpha=.15)
    fig.suptitle('Literature-motivated moving fronts: frozen controllers, no new training',fontsize=13)
    fig.supxlabel('Bands = observed training / graph run ranges; 3 random graphs are not 9 independent graph samples.',fontsize=9)
    fig.savefig(out/'moving-front.png',dpi=180);plt.close(fig)
    summary['analytic_reference']=reference['reference_checks']
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(out,flush=True)


if __name__=='__main__':main()
