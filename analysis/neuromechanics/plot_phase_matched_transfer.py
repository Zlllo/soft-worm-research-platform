"""Supplemental figure: phase-matched muscle signals, different body response.

The cases and central position are fixed in this script; no parameter fitting
or simulation is performed. This illustration was added after the main assay.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from thermotaxis.neuromechanics.boyle_output import curvature


ROOT=Path('results/calcium-body-transfer-2026-11-13/cases')
OUTPUT=Path('docs/reports/assets/calcium-body-transfer-2026-11-13')


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'#fbfaf7','axes.facecolor':'#fbfaf7',
                         'savefig.facecolor':'#fbfaf7'})
    pairs=[('wave_r1_f0.5',slice(1800,2000,4),'0.5 Hz / 100 ms muscle','#268b8b','-'),
           ('fast_muscle_r1_f2',slice(1950,2000),'2 Hz / 25 ms muscle','#d37a50','--')]
    phase=np.arange(50)/50
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    for name,selection,label,color,style in pairs:
        z=np.load(ROOT/name/'trace.npz')
        m=z['muscle'][selection,24,0]-z['muscle'][selection,24,1]
        k=curvature(z['pose'][selection])[:,23]*1e-3
        axes[0].plot(phase,m,color=color,ls=style,lw=2.6,label=label)
        axes[1].plot(phase,k,color=color,ls=style,lw=2.6,label=label)
    axes[0].set(title='A  Almost identical muscle output',ylabel='Dorsal − ventral activation')
    axes[1].set(title='B  Different physical bending',ylabel='Central curvature (mm$^{-1}$)')
    for ax in axes:
        ax.set_xlabel('Phase within one cycle');ax.grid(alpha=.15);ax.axhline(0,color='#aaa49d',lw=.7)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Matching muscle output does not match body movement',fontsize=16,weight='bold')
    fig.savefig(OUTPUT/'phase-matched-body.png',dpi=180)
    fig.savefig(OUTPUT/'phase-matched-body.svg')
    plt.close(fig)


if __name__=='__main__':main()
