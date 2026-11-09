"""Summary figures from frozen B-cell stage traces."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path('results/b-motor-mechanisms-2026-11-07')
OUT=Path('docs/reports/assets/b-motor-mechanisms-2026-11-07')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    s=json.loads((ROOT/'summary.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.5),layout='constrained')
    ax=axes[0,0]
    for name,label in [('prototype_off8','B prototype + AVB'),('calcium0_8','Calcium conductance removed'),('generic_C1_8','Matched C1 cell')]:
        with np.load(ROOT/name/'trace.npz') as z:
            i=z['neuron_names'].tolist().index('DB4');keep=z['time_s']>=4
            ax.plot(z['time_s'][keep],z['neuron_voltage_mV'][keep,i],label=label,lw=1.3)
    ax.set(xlabel='Time (s)',ylabel='DB4 voltage (mV)',title='A. Sustained rhythm within the full302 graph')
    ax.legend(fontsize=8)
    ax=axes[0,1];b=s['bifurcation']
    ax.plot([d['gap_nS'] for d in b],[d['late_amplitude_mV'] for d in b],'o-',color='#1976b8')
    ax.set(xlabel='Total AVB conductance to one B cell (nS)',ylabel='Late voltage amplitude (mV)',title='B. Coupling has an oscillatory window')
    ax.text(.04,.9,'Isolated engineering prototype',transform=ax.transAxes,fontsize=9)
    ax=axes[1,0]
    for g,color in [(.09,'#8996a3'),(.3,'#d67729'),(.9,'#1976b8')]:
        with np.load(ROOT/f'nmj-{g}nS/trace.npz') as z:
            keep=(z['time_s']>=6)&(z['time_s']<=8)
            ax.plot(z['time_s'][keep],z['drive'][keep],label=f'{g} nS NMJ',color=color)
    ax.set(xlabel='Time (s)',ylabel='C1 muscle drive',title='C. Same neuron waveform, different NMJ efficacy')
    ax.legend(fontsize=8)
    ax=axes[1,1]
    with np.load(ROOT/'prototype_off8/trace.npz') as z:
        names=z['neuron_names'].tolist();bn=z['B_names'].tolist();bi=[names.index(n) for n in bn]
        keep=z['time_s']>=4;v=z['neuron_voltage_mV'][keep][:,bi]
        # Normalize each cell for timing visualization; amplitude is in summary.
        normalized=(v-v.mean(0))/np.maximum(v.std(0),1e-10)
        im=ax.imshow(normalized.T,aspect='auto',origin='lower',extent=[4,8,0,18],cmap='RdBu_r',vmin=-2,vmax=2)
        ax.set_yticks(np.arange(18)+.5,bn,fontsize=7)
        ax.set(xlabel='Time (s)',title='D. Timing across all18 B-type model neurons')
        fig.colorbar(im,ax=ax,label='Within-cell voltage z-score',shrink=.8)
    fig.suptitle('B-cell mechanism stage — engineered prototype, no locomotion success claim',fontsize=13)
    fig.savefig(OUT/'b-motor-summary.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
