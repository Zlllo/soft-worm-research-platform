"""Plot the fixed-interface data/updates/capacity comparison on fresh cases."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def stats(values):
    values=[float(x) for x in values]
    return dict(mean=float(np.mean(values)),minimum=min(values),maximum=max(values),n=len(values))


def group(name):
    return 'rewired' if name.startswith('rewired') else name


def main():
    root=Path('results/tracking-learning-surface-2026-10-24')
    output=Path('docs/reports/assets/tracking-surface-2026-10-24');output.mkdir(parents=True,exist_ok=True)
    rows=[];fits=[]
    for shard in ['rates','grus']:
        result=json.loads((root/shard/'results.json').read_text())
        cases=json.loads((root/shard/'fresh-cases.json').read_text())
        assert result['fitting_complete'] and cases['complete'] and all(result['source_hashes_unchanged'].values())
        assert len(cases['runs'])==len(result['runs'])*2
        rows.extend(cases['runs']);fits.extend(result['runs'])
    assert len(rows)==126 and len(fits)==63
    assert len({(r['model'],r['seed'],r['episodes_per_field'],r['updates']) for r in rows})==126
    labels={'real8':'Real graph (71)','rewired':'Rewired graphs (71)','gru2':'GRU2 (57)',
            'gru3':'GRU3 (94)','gru16':'GRU16 (1121)'}
    colors={'real8':'#9868a6','rewired':'#73818b','gru2':'#c69a40','gru3':'#418d7f','gru16':'#377fa8'}
    summary=dict(status='exploratory_fixed_full_io_fresh_cases',training_configs=63,new_fits=sum(r['new_fit'] for r in fits),
        reused_full_fits=sum(not r['new_fit'] for r in fits),endpoints=126,
        verified_parent_512_prefixes=sum('exact_parent_validation_match' in r['endpoints']['512'] for r in fits),
        case_seeds=[9301,9302,9303,9304],updates={},notes=[
            'Ranges are observed extrema, not confidence intervals.',
            'Three rewired graphs crossed with three training seeds; nine runs are not nine graph samples.',
            'GRU2/3 are parameter-near comparisons, not equal hidden-state or computational-capacity controls.',
            'A single fixed nested trajectory subset is used; no data-order replication.',
            'Eight fresh cases were viewed only after both fitting shards finished.'])
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for updates,ax in zip([512,2048],axes):
        summary['updates'][str(updates)]={}
        for label in labels:
            points=[];lo=[];hi=[];table={}
            for budget in [4,16,64]:
                selected=[r for r in rows if r['updates']==updates and r['episodes_per_field']==budget and group(r['model'])==label]
                s=stats(r['mean_tracking'] for r in selected)
                table[str(budget)]=dict(tracking=s,
                    per_graph={n:stats(r['mean_tracking'] for r in selected if r['model']==n) for n in sorted({r['model'] for r in selected})},
                    training_runs_ge_80_percent=sum(r['mean_tracking']>=.8 for r in selected))
                points.append(s['mean']*100);lo.append(s['minimum']*100);hi.append(s['maximum']*100)
            summary['updates'][str(updates)][label]=table
            ax.plot([8,32,128],points,'-o',color=colors[label],label=labels[label],lw=1.8,markersize=4)
            ax.fill_between([8,32,128],lo,hi,color=colors[label],alpha=.08)
        ax.set(xscale='log',ylim=(0,105),xlabel='Total teacher trajectories',ylabel='Tracking time (%)',title=f'{updates} optimizer updates; same 8 fresh cases')
        ax.set_xticks([8,32,128],['8','32','128']);ax.grid(alpha=.15)
        ax.legend(fontsize=8,frameon=False,loc='lower right')
    fig.suptitle('Full learned input/output: data amount, optimization and model capacity',fontsize=13)
    fig.supxlabel('All trajectories last 60 s; shading = observed training / graph run range, not a confidence interval.',fontsize=9)
    fig.savefig(output/'tracking-surface.png',dpi=180);plt.close(fig)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['updates','notes']},indent=2))


if __name__=='__main__':
    main()
