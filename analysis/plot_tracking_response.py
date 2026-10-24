"""Frozen-policy response tradeoffs; no energy or training-efficiency claims."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path('results/tracking-response-probe-2026-10-24')
    out=Path('docs/reports/assets/tracking-response-2026-10-24');out.mkdir(parents=True,exist_ok=True)
    report=json.loads((root/'results.json').read_text())
    assert len(report['runs'])==336 and all(report['source_hashes_unchanged'].values())
    names=['real8','rewired8-11','gru3','gru16','expert']
    labels=['Real graph (71)','Rewire11 (71)','GRU3 (94)','GRU16 (1121)','Engineering expert']
    colors=['#9868a6','#73818b','#418d7f','#377fa8','#bb913e']
    summary=dict(status='frozen_policy_response_not_training',rows=336,conditions={},
        caveat='Physics-grid acceleration RMS for tau=0 diverges with grid refinement; not used as an energy or cross-grid smoothness metric.')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,noise in enumerate([.002,.01]):
        ax=axes[0,i];summary['conditions'][str(noise)]={}
        for name,label,color in zip(names,labels,colors):
            y=[];low=[];high=[];table={}
            for tau in [0.,.3,1.2]:
                rows=[r for r in report['runs'] if r['model']==name and r['noise_std_K']==noise and r['tau_s']==tau]
                seeds=sorted({r['training_seed'] for r in rows},key=str)
                means=[np.mean([r['metrics']['tracking_fraction'] for r in rows if r['training_seed']==s]) for s in seeds]
                y.append(np.mean(means)*100);low.append(min(means)*100);high.append(max(means)*100)
                table[str(tau)]=dict(tracking=float(np.mean(means)),training_seed_range=[float(min(means)),float(max(means))],
                    center_error_rms_K=float(np.mean([r['metrics']['center_error_rms_K'] for r in rows])),
                    executed_turn_rms_rad_s=float(np.mean([r['metrics']['executed_turn_rms_rad_s'] for r in rows])))
            summary['conditions'][str(noise)][name]=table
            ax.plot([0,.3,1.2],y,'-o',color=color,label=label,lw=1.5,markersize=4)
            ax.fill_between([0,.3,1.2],low,high,color=color,alpha=.07)
        ax.set(ylim=(0,104),xlabel='Execution response time (s)',ylabel='Tracking time (%)',title=f'Sensor noise SD = {noise} K; policies frozen')
        ax.set_xticks([0,.3,1.2]);ax.grid(alpha=.15);ax.legend(fontsize=7,frameon=False,loc='lower left')
    ax=axes[1,0]
    with np.load(root/'example-traces.npz',allow_pickle=False) as traces:
        for tau,color in zip([0.,.3,1.2],['#b7b9bd','#4c8d9e','#974d7f']):
            trace=traces[f'real8-tau{tau}-trajectory']
            mask=(trace[:,0]>=50)&(trace[:,0]<=70)
            ax.plot(trace[mask,0],trace[mask,4],color=color,label=f'tau={tau} s',lw=1)
    ax.set(title='Preset example: real graph seed7, radial case9501, noise0.01K',xlabel='Time (s)',ylabel='Executed turn rate (rad/s)')
    ax.grid(alpha=.15);ax.legend(fontsize=8,frameon=False)
    ax=axes[1,1]
    for name,label,color in zip(names,labels,colors):
        data=summary['conditions']['0.01'][name]
        ax.plot([0,.3,1.2],[data[str(t)]['center_error_rms_K']*1000 for t in [0.,.3,1.2]],'-o',color=color,label=label,lw=1.5,markersize=4)
    ax.set(title='High noise: temperature tracking error',xlabel='Execution response time (s)',ylabel='Center temperature RMSE (mK)')
    ax.set_xticks([0,.3,1.2]);ax.grid(alpha=.15)
    fig.suptitle('Finite response filters motion and adds delay: paired frozen-policy checks',fontsize=13)
    fig.supxlabel('All policies trained at tau=0.3 s; bands = three training-seed means, not confidence intervals. No retraining.',fontsize=9)
    fig.savefig(out/'tracking-response.png',dpi=180);plt.close(fig)
    check=json.loads((root/'dt-check/results.json').read_text())
    summary['dt_check']=dict(cases=len(check['runs']),comparisons=[])
    for name in ['real8','gru16']:
        for tau in [0.,.3,1.2]:
            a,b=[next(r for r in check['runs'] if r['model']==name and r['tau_s']==tau and r['dt_s']==dt) for dt in [.01,.005]]
            summary['dt_check']['comparisons'].append(dict(model=name,tau_s=tau,
                tracking_delta=b['metrics']['tracking_fraction']-a['metrics']['tracking_fraction'],
                physics_grid_acceleration_rms_ratio=b['physics_grid_acceleration_rms']/a['physics_grid_acceleration_rms'],
                control_grid_rate_change_rms_ratio=b['control_grid_rate_change_rms']/a['control_grid_rate_change_rms']))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(out,flush=True)


if __name__=='__main__':main()
