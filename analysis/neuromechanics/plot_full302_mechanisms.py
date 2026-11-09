"""Figures from archived raw physical and full302 sensitivity traces."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


ASSETS=Path('docs/reports/assets/full302-mechanisms-2026-11-07')
NEURAL=Path('results/full302-operating-range-2026-11-07')
PHYSICAL=Path('results/body-phase-coordination-2026-11-07')
ISOTROPIC=Path('results/body-isotropic-drag-2026-11-07/cases')


def load(root,name):
    with np.load(root/name/'trace.npz') as z:return {k:z[k].copy() for k in z.files}


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    cases=[('Traveling wave',PHYSICAL,'head_to_tail'),('Reverse wave',PHYSICAL,'tail_to_head'),
           ('Standing wave',PHYSICAL,'standing'),('Shuffled phases',PHYSICAL,'phase_shuffled'),
           ('Same wave, isotropic drag',ISOTROPIC,'head_to_tail')]
    speeds=[json.loads((r/n/'record.json').read_text())['late_signed_speed_um_s'] for _,r,n in cases]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    ax=axes[0,0]
    colors=['#1976b8','#d67729','#8996a3','#9970ab','#425569']
    ax.barh([c[0] for c in cases],speeds,color=colors)
    for i,v in enumerate(speeds):
        if abs(v)<1e-10:v=0.
        ax.text(v+1 if v>=0 else v/2,i,f'{v:.2f}',ha='left' if v>=0 else 'center',va='center',color='black' if v>=0 else 'white')
    ax.set(xlim=(-34,36),xlabel='Mean signed speed, 6–12 s (μm/s)',title='A. Muscle timing and directional drag enable propulsion')
    ax.axvline(0,color='black',lw=.7)
    ax=axes[0,1]
    for title,r,n in [cases[0],cases[2],cases[3],cases[4]]:
        z=load(r,n);c=z['pose'][:,:,:2].mean(1)*1e6
        axis=(z['pose'][0,0,:2]-z['pose'][0,-1,:2]);axis/=np.linalg.norm(axis)
        ax.plot(z['time_s'],(c-c[0])@axis,label=title)
    ax.set(xlabel='Time (s)',ylabel='Centroid forward displacement (μm)',title='B. Raw body translation, no prescribed position')
    ax.legend(fontsize=8)
    ax=axes[1,0]
    for k,label,color in [('kernels','Original exported update','#d67729'),('kernels-direct-cnexp','Direct cnexp update','#1976b8')]:
        with np.load(NEURAL/f'synapse-check-{k}.npz') as z:v=z['values']
        ax.plot(v[:,0],np.maximum(abs(v[:,1]-v[:,2]),1e-17),label=label,color=color)
    ax.set(yscale='log',xlabel='Time (ms)',ylabel='Absolute gate error vs exact solution',title='C. A numerical export issue, not neural excitation')
    ax.legend(fontsize=8)
    ax=axes[1,1]
    labels=['Original range','NMJ threshold −30','NN + NMJ −30','NN + NMJ −30\n(no input)']
    ns=['base_AVB_cnexp','nmj30_AVB_cnexp','both30_AVB_cnexp','both30_none_cnexp']
    vals=[json.loads((NEURAL/n/'record.json').read_text())['last_half_mean_nmj_release'] for n in ns]
    ax.bar(range(4),vals,color=['#8996a3','#1976b8','#d67729','#9970ab'])
    ax.set(yscale='log',ylabel='Mean NMJ gate, 1–2 s',title='D. More release can also mean more background')
    ax.set_xticks(range(4),labels,fontsize=8)
    for i,v in enumerate(vals):ax.text(i,v*1.3,f'{v:.3g}',ha='center',fontsize=9)
    ax.set_ylim(1e-5,3)
    fig.suptitle('Full302 model diagnosis and physical controls — 7 Nov 2026',fontsize=14)
    fig.savefig(ASSETS/'mechanism-summary.png',dpi=160)
    plt.close(fig)

    # Two physical models driven by exactly the same named muscle waveform.
    traces=[load(PHYSICAL,'head_to_tail'),load(ISOTROPIC,'head_to_tail')]
    titles=['Directional drag (normal/tangential = 40)', 'Equal drag (same mean resistance)']
    fig,axes=plt.subplots(2,1,figsize=(10.5,4.4))
    fig.subplots_adjust(left=.08,right=.98,bottom=.19,top=.85,hspace=.7)
    lines=[];heads=[];paths=[]
    for i,(ax,z,title) in enumerate(zip(axes,traces,titles)):
        ax.set(xlim=(-360,1040),ylim=(-80,80),xlabel='Position (μm)',ylabel='μm',title=title)
        ax.set_aspect('equal',adjustable='box')
        ax.axhline(0,lw=.4,color='#cccccc')
        line,=ax.plot([],[],lw=3,color=['#1976b8','#9970ab'][i]);lines.append(line)
        head,=ax.plot([],[],'o',ms=6,color='#db752c');heads.append(head)
        path,=ax.plot([],[],lw=1,color='#db752c',alpha=.55);paths.append(path)
    title=fig.suptitle('Identical prescribed muscle waves — physical controls, no neurons',fontsize=13)
    clock=fig.text(.5,.03,'',ha='center',fontsize=10)
    frames=np.arange(0,len(traces[0]['time_s']),5)
    def update(k):
        for z,line,head,path in zip(traces,lines,heads,paths):
            xy=z['pose'][k,:,:2]*1e6
            line.set_data(xy[:,0],xy[:,1]);head.set_data([xy[0,0]],[xy[0,1]])
            c=z['pose'][:k+1,:,:2].mean(1)*1e6;path.set_data(c[:,0],c[:,1])
        clock.set_text(f't = {traces[0]["time_s"][k]:.2f} s    orange marker: head; orange line: centroid path')
        return lines+heads+paths+[clock]
    anim=FuncAnimation(fig,update,frames=frames,interval=50,blit=False)
    anim.save(ASSETS/'same-muscle-wave-different-drag.gif',writer=PillowWriter(fps=20),dpi=100)
    update(frames[-1]);fig.savefig(ASSETS/'physical-control-final.png',dpi=160)
    plt.close(fig)
    print('Saved figures and physical-control GIF')


if __name__=='__main__':main()
