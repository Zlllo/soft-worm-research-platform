"""Plot all fixed conditions; noise ranges are not confidence intervals."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ASSETS=Path('docs/reports/assets/physical-thermal-readout-2026-11-13')
CASES=['wave_r1_f0.5','wave_r1_f2','heavy_drag_r1_f2']
LABELS=['0.5 Hz','2 Hz','2 Hz\n4× drag']


def main():
    s=json.loads((ASSETS/'summary.json').read_text())
    conditions={(v['case'],v['mode'],v['noise_mK']):v for v in s['conditions']}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'#fbfaf7','axes.facecolor':'#fbfaf7','savefig.facecolor':'#fbfaf7'})
    fig,axes=plt.subplots(1,3,figsize=(14.8,4.8),constrained_layout=True)
    geometry=[conditions[(n,'matched',.1)] for n in CASES]
    colors=['#268b8b','#dd944b','#6c6ca4']
    x=np.arange(3)
    axes[0].bar(x,[v['normal_excursion_um_median'] for v in geometry],color=colors,width=.6)
    axes[0].set_xticks(x,LABELS)
    axes[0].set(ylabel='Effective transverse head excursion (µm)',title='A  Motion available for sensing')
    axes[0].grid(axis='y',alpha=.15)
    for i,v in enumerate(geometry):
        axes[0].text(i,v['normal_excursion_um_median']+.035,f"{v['normal_excursion_um_median']:.3f}",ha='center',fontsize=10)
    axes[0].set_ylim(0,1.45)
    for ax,noise,title in zip(axes[1:],[.1,1.],['B  Temperature noise: 0.1 mK','C  Temperature noise: 1 mK']):
        for offset,mode,color,label in [(-.18,'matched','#268b8b','Matched motion'),
                                         (.18,'delay_0.6s','#d37a50','Motion delayed 0.6 s')]:
            vals=[conditions[(n,mode,noise)] for n in CASES]
            means=np.array([v['balanced_accuracy'] for v in vals])*100
            low=np.array([min(q['balanced_accuracy'] for q in v['noise_seed_results']) for v in vals])*100
            high=np.array([max(q['balanced_accuracy'] for q in v['noise_seed_results']) for v in vals])*100
            ax.bar(x+offset,means,width=.33,color=color,label=label,zorder=2)
            ax.errorbar(x+offset,means,yerr=np.stack([means-low,high-means]),fmt='none',
                        ecolor='#443e39',capsize=3,lw=1,zorder=3)
            for i,value in enumerate(means):
                ax.text(i+offset,max(2,value-9),f'{value:.1f}',ha='center',color='white',fontsize=9,weight='bold')
        ax.axhline(50,color='#77716a',ls='--',lw=1,label='Chance level',zorder=1)
        ax.set_xticks(x,LABELS);ax.set_ylim(0,106)
        ax.set(ylabel='Direction decoding accuracy (%)',title=title)
        ax.grid(axis='y',alpha=.15)
    axes[1].legend(frameon=False,fontsize=8,loc='upper right')
    fig.suptitle('Body movement sets the information in local temperature samples',fontsize=16,weight='bold')
    fig.savefig(ASSETS/'physical-thermal-summary.png',dpi=180)
    fig.savefig(ASSETS/'physical-thermal-summary.svg')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.7,4.6),constrained_layout=True)
    for i,v in enumerate(geometry):
        axes[0].scatter(v['condition_number_median'],v['predicted_sigma_normal_K_per_m_median'],s=100,color=colors[i])
        axes[0].annotate(LABELS[i].replace('\n',' '),(v['condition_number_median'],v['predicted_sigma_normal_K_per_m_median']),
                         xytext=(7,3),textcoords='offset points',fontsize=10)
    axes[0].set(xlabel='Geometric condition number (median)',ylabel='Predicted gradient uncertainty (K/m)',
                title='A  Condition number misses signal size')
    axes[0].set_xlim(.9,3.8);axes[0].set_ylim(0,175);axes[0].grid(alpha=.15)
    axes[1].plot([45,100],[45,100],color='#aaa49d',ls='--',lw=1)
    for i,name in enumerate(CASES):
        for noise,marker in [(.1,'o'),(1.,'s')]:
            v=conditions[(name,'matched',noise)]
            axes[1].scatter(v['predicted_matched_accuracy']*100,v['balanced_accuracy']*100,
                            s=80,color=colors[i],marker=marker,label=LABELS[i].replace('\n',' ') if noise==.1 else None)
    axes[1].set(xlabel='Gaussian equation prediction (%)',ylabel='Observed four-seed accuracy (%)',
                title='B  Fixed physical readout, no training')
    axes[1].set_xlim(45,100);axes[1].set_ylim(45,100);axes[1].legend(frameon=False,fontsize=9)
    axes[1].grid(alpha=.15)
    fig.suptitle('Absolute motion and noise determine directional uncertainty',fontsize=15,weight='bold')
    fig.savefig(ASSETS/'geometry-and-uncertainty.png',dpi=180)
    fig.savefig(ASSETS/'geometry-and-uncertainty.svg')
    plt.close(fig)


if __name__=='__main__':main()
