"""Summarize grouped sensorimotor decoding and every frozen graph probe."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from diagnose_navigation_failures import digest,write
from run_active_sensing_stage3 import CONDITIONS,check


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/active-sensing-stage3-2026-11-03'))
    p.add_argument('--output',type=Path,default=Path('docs/reports/assets/active-sensing-stage3-2026-11-03'))
    a=p.parse_args();root,out=a.root,a.output
    d=json.loads((root/'design.json').read_text());check(d)
    dataset=json.loads((root/'dataset.json').read_text());assert dataset['complete']
    assert digest(root/'dataset.npz')==dataset['dataset_sha256']
    regression=json.loads((root/'local-regression.json').read_text())
    transfer=json.loads((root/'frozen-paired-transfer.json').read_text());assert transfer['complete']
    neurons=json.loads((root/'neuronal-probes.json').read_text())
    groups=[]
    for condition in CONDITIONS:
        rows=[json.loads((root/f'gru-{condition}-{s}.json').read_text()) for s in d['generic']['seeds']]
        acute=[r for r in transfer['rows'] if r['condition']==condition]
        reg=next(r for r in regression['rows'] if r['condition']==condition)
        groups.append(dict(condition=condition,by_seed=rows,
            refitted_test_accuracy=float(np.mean([r['test']['balanced_accuracy'] for r in rows])),
            refitted_noisy_accuracy=float(np.mean([r['test_noisy']['balanced_accuracy'] for r in rows])),
            refitted_noiseless_accuracy=float(np.mean([r['test_noiseless']['balanced_accuracy'] for r in rows])),
            frozen_paired_transfer_accuracy=float(np.mean([r['test']['balanced_accuracy'] for r in acute])),
            local_regression=reg))
    neural_groups=[]
    if not neurons.get('skipped'):
        for interface in ['full','restricted']:
            for checkpoint in ['BC','All']:
                rows=[r for r in neurons['rows'] if r['spec']['name']=='real8' and r['spec']['interface']==interface and r['checkpoint']==checkpoint]
                neural_groups.append(dict(interface=interface,checkpoint=checkpoint,
                    layers={layer:float(np.mean([r['layers'][layer]['test']['balanced_accuracy'] for r in rows])) for layer in ['AFD','AIY','RIA','all8']},
                    by_seed=rows))
    result=dict(status='completed_information_assay_no_navigation_training',dataset=dataset,conditions=groups,
        neuronal_groups=neural_groups,all_neuronal_probes=neurons,
        new_generic_fits=12,neural_backbone_fits=0,
        training_wall_s=sum(r['wall_s'] for g in groups for r in g['by_seed']),
        notes=d['limits'],split=d['split'],raw_windows=dataset['windows'])
    delay_path=root/'causal-delay.json'
    if delay_path.exists():
        result['supplemental_continuous_delay']=json.loads(delay_path.read_text())
    out.mkdir(parents=True,exist_ok=True);write(root/'summary.json',result);write(out/'summary.json',result)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    x=np.arange(4)
    for key,label,style in [('refitted_test_accuracy','GRU16 refitted','o-'),
                            ('frozen_paired_transfer_accuracy','Paired GRU16 frozen','s--')]:
        axes[0].plot(x,[100*g[key] for g in groups],style,label=label)
    axes[0].plot(x,[100*g['local_regression']['test']['balanced_accuracy'] for g in groups],'d-',label='Local regression + ridge')
    for i,g in enumerate(groups):
        axes[0].scatter(np.full(3,i),[100*r['test']['balanced_accuracy'] for r in g['by_seed']],s=20,alpha=.6,color='#3267ad')
    axes[0].axhline(50,color='gray',ls=':')
    axes[0].set(xticks=x,xticklabels=['T only','T + actual motion','Mispaired motion','Center T + motion'],
        ylabel='Held-out run balanced accuracy (%)',ylim=(45,102),title='A  Alignment, refitting and sensor position')
    axes[0].tick_params(axis='x',rotation=15)
    for g in neural_groups:
        label=g['interface']+' '+g['checkpoint']
        axes[1].plot(range(3),[100*g['layers'][s] for s in ['AFD','AIY','RIA']],'o'+('-' if g['checkpoint']=='All' else '--'),label=label)
    axes[1].axhline(50,color='gray',ls=':')
    axes[1].set(xticks=range(3),xticklabels=['AFD pair','AIY pair','RIA pair'],ylim=(45,70),
        ylabel='Simple probe balanced accuracy (%)',title='B  Frozen anatomical states on actual body input')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Split by whole pulse-phase groups; all field/noise versions stay together; three decoder seeds',fontsize=10)
    fig.savefig(out/'active-sensing.png',dpi=170);fig.savefig(out/'active-sensing.svg');plt.close(fig)
    if neural_groups:
        names=['real8']+['rewired8-'+str(s) for s in [11,22,33,44,55,66,77,88,99,110]]
        fig,ax=plt.subplots(figsize=(11,3.6),layout='constrained')
        for interface,color in [('full','#3267ad'),('restricted','#d68a2a')]:
            means=[]
            for i,name in enumerate(names):
                rows=[r for r in neurons['rows'] if r['checkpoint']=='BC' and r['spec']['name']==name and r['spec']['interface']==interface]
                vals=[100*r['layers']['RIA']['test']['balanced_accuracy'] for r in rows];means.append(float(np.mean(vals)))
                ax.scatter(np.full(3,i),vals,color=color,alpha=.45,s=16)
            ax.plot(range(len(names)),means,'o-',color=color,label=interface)
        ax.axhline(50,color='gray',ls=':');ax.set(xticks=range(len(names)),xticklabels=['real']+[str(s) for s in [11,22,33,44,55,66,77,88,99,110]],
            ylim=(40,75),ylabel='RIA-pair probe balanced accuracy (%)',xlabel='Original real graph and all 10 rewiring seeds',title='Frozen BC controls; no layout-specific retraining')
        ax.grid(alpha=.2);ax.legend();fig.savefig(out/'frozen-graphs.png',dpi=170);plt.close(fig)
    print(json.dumps({g['condition']:dict(refitted=g['refitted_test_accuracy'],frozen=g['frozen_paired_transfer_accuracy'],ridge=g['local_regression']['test']['balanced_accuracy']) for g in groups},indent=2))


if __name__=='__main__':main()
