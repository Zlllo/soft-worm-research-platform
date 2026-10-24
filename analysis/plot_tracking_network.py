"""Plot supervised tracking curves, paired outcomes and frozen transfer checks."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_tracking_network import VARIANTS,expert_for
from thermotaxis.isothermal_scene import TrackingConfig,run_tracking
from thermotaxis.tracking_learning import TrackingNetworkConfig,TrackingNetwork,NeuralTrackingController


def tracking(rows,field=None):
    return float(np.mean([r['metrics']['tracking_fraction'] for r in rows if field is None or r['field']==field]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('results/neural-tracking-2026-10-24'))
    parser.add_argument('--output',type=Path,default=Path('docs/reports/assets/neural-tracking-2026-10-24/neural-tracking.png'))
    args=parser.parse_args()
    report=json.loads((args.root/'results.json').read_text())
    diagnosis=json.loads((args.root/'diagnostics/results.json').read_text())
    variants=list(VARIANTS)
    labels=['Raw GRU16','Demeaned GRU16','Window MLP']
    colors=['#4d7da3','#008b82','#8e6b9e']
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    ax=axes[0,0]
    for variant,label,color in zip(variants,labels,colors):
        rows=[r for r in report['runs'] if r['variant']==variant]
        curves=[[c for c in r['curves'] if c['stage']=='offline_imitation'] for r in rows]
        x=[c['update'] for c in curves[0]]
        y=np.array([[c['metrics']['all']['normalized_mse'] for c in row] for row in curves])
        ax.plot(x,y.mean(0),label=label,color=color)
        ax.fill_between(x,y.min(0),y.max(0),alpha=.12,color=color)
    ax.axhline(report['runs'][0]['curves'][0]['metrics']['all']['zero_command_mse'],color='#888888',ls='--',lw=1,label='Zero-command baseline')
    ax.set(xlabel='Offline optimizer updates',ylabel='Normalized validation action MSE',title='Same expert dataset; shading = 3-seed range')
    ax.legend(fontsize=8,frameon=False);ax.grid(alpha=.15)
    ax=axes[0,1]
    for shift,stage,label,color in [(-.18,'stage1_same_final_cases','After 512 updates','#74a9bb'),(.18,'final_basic','Final selected stage','#008b82')]:
        means=[];lo=[];hi=[]
        for variant in variants:
            values=np.array([tracking(r[stage])*100 for r in diagnosis['runs'] if r['variant']==variant])
            means.append(values.mean());lo.append(values.mean()-values.min());hi.append(values.max()-values.mean())
        ax.bar(np.arange(3)+shift,means,width=.34,label=label,color=color,yerr=[lo,hi],capsize=2)
    ax.set_xticks(np.arange(3),labels)
    ax.set(ylim=(0,106),ylabel='Tracking time (%)',title='Same 8 basic cases; whiskers = training-seed range')
    ax.legend(fontsize=8,frameon=False);ax.grid(axis='y',alpha=.15)
    # Transfers use the original fixed evaluation seeds. Radius columns are radial only.
    ax=axes[1,0]
    columns=['Basic','Noise\n0.01 K','Gradient\n70 K/m','Response\n1.2 s','Radius\n3 mm*','Radius\n8 mm*','Sweep\n0.4 Hz','Sweep\n0.75 Hz']
    matrix=[]
    for variant in ['expert']+variants:
        if variant=='expert':
            values=[tracking(report['final_baselines']['expert'])]
            for case in ['more_noise','steeper_gradient','slower_response','smaller_radius','larger_radius','slower_sweep','faster_sweep']:
                values.append(tracking(diagnosis['baselines'][case]['expert'],'radial' if 'radius' in case else None))
        else:
            rows=[r for r in report['runs'] if r['variant']==variant]
            diag=[r for r in diagnosis['runs'] if r['variant']==variant]
            values=[np.mean([tracking(r['final_behavior']) for r in rows])]
            values.extend(np.mean([tracking(r['transfer'][case]) for r in rows]) for case in ['more_noise','steeper_gradient','slower_response'])
            values.extend(np.mean([tracking(r['additional_transfer'][case],'radial' if 'radius' in case else None) for r in diag])
                          for case in ['smaller_radius','larger_radius','slower_sweep','faster_sweep'])
        matrix.append(values)
    matrix=np.array(matrix)*100
    ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    for i in range(4):
        for j in range(len(columns)):
            ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>65 else '#162e3e')
    ax.set_xticks(range(len(columns)),columns,fontsize=8)
    ax.set_yticks(range(4),['Expert']+labels,fontsize=9)
    ax.set_title('Exploratory transfer tracking (%); * radial only',loc='left',fontsize=10)
    ax=axes[1,1]
    config=TrackingConfig(field='radial')
    angle=np.linspace(0,2*np.pi,361)
    radius=config.target_radius_m*1000
    width=config.comfort_half_width_K/config.gradient_K_m*1000
    for r in (radius-width,radius+width):ax.plot(r*np.cos(angle),r*np.sin(angle),color='#d7e8e4',lw=1)
    reference=run_tracking(config,expert_for(config),5001,record_trajectory=True)
    trace=reference['trajectory'];ax.plot(trace[:,1]*1000,trace[:,2]*1000,'--',color='#69757b',lw=2,label='Expert')
    for variant,label,color in zip(variants,labels,colors):
        architecture,encoding=VARIANTS[variant]
        model=TrackingNetwork(TrackingNetworkConfig(architecture=architecture,encoding=encoding))
        model.load_state_dict(torch.load(args.root/f'{variant}-7-final.pt',map_location='cpu',weights_only=True));model.eval()
        result=run_tracking(config,NeuralTrackingController(model),5001,record_trajectory=True)
        trace=result['trajectory']
        ax.plot(trace[:,1]*1000,trace[:,2]*1000,label=label,color=color,lw=1.3)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title='Radial example: case 5001, training seed 7')
    ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=8,frameon=False);ax.grid(alpha=.15)
    fig.suptitle('Ordinary neural tracking: supervised imitation, not RL or biological-circuit evidence',fontsize=13)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180);plt.close(fig)
    print(args.output)


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
