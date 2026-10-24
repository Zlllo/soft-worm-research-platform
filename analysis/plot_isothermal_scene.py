"""Static scientific figure from recorded traces and all paired scene results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('results/isothermal-scene-2026-10-24'))
    parser.add_argument('--output',type=Path,default=Path('docs/reports/assets/isothermal-scene-2026-10-24/scene-validation.png'))
    args=parser.parse_args()
    report=json.loads((args.root/'results.json').read_text())
    conditions=['blind','head_phase','center_phase','head_quarter_offset','head_scrambled']
    labels=['Fixed heading','Head + phase','Center + phase','Phase + 90 deg','Shuffled phase']
    colors=['#7c8592','#008b82','#d38b32','#7860ad','#ca5e63']
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for ax,field in zip(axes[0],('linear','radial')):
        for condition,label,color in zip(conditions,labels,colors):
            row=next(r for r in report['runs'] if r['config']['field']==field and r['condition']==condition
                     and r['config']['noise_std_K']==.002 and r['seed']==101)
            trace=np.load(args.root/row['trace_file'])['trajectory']
            x,y=trace[:,1],trace[:,2]
            if field=='linear':
                phi=row['initial_state']['field_angle_rad']
                x,y=x*np.cos(phi)+y*np.sin(phi),-x*np.sin(phi)+y*np.cos(phi)
            ax.plot(x*1000,y*1000,color=color,label=label,lw=1.7,alpha=.95)
            ax.plot(x[0]*1000,y[0]*1000,'o',color=color,ms=3)
        cfg=row['config']
        width=cfg['comfort_half_width_K']/cfg['gradient_K_m']*1000
        if field=='linear':
            ax.axvspan(-width,width,color='#008b82',alpha=.08)
            ax.axvline(0,color='#008b82',lw=.7,ls='--')
            ax.set(xlabel='Normal to isotherm (mm)',ylabel='Along isotherm (mm)')
        else:
            angle=np.linspace(0,2*np.pi,361)
            r=cfg['target_radius_m']*1000
            ax.fill(np.r_[ (r+width)*np.cos(angle),(r-width)*np.cos(angle[::-1]) ],
                    np.r_[ (r+width)*np.sin(angle),(r-width)*np.sin(angle[::-1]) ],color='#008b82',alpha=.08)
            ax.plot(r*np.cos(angle),r*np.sin(angle),'--',color='#008b82',lw=.7)
            ax.set(xlabel='x (mm)',ylabel='y (mm)')
        ax.set_title(f'{field.title()} field: example seed 101, noise 0.002 K',loc='left',fontsize=11)
        ax.set_aspect('equal',adjustable='datalim')
        ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,frameon=False)
    ax=axes[1,0]
    x=np.arange(len(conditions))
    for shift,field,color in [(-.18,'linear','#497fa7'),(.18,'radial','#008b82')]:
        means=[];low=[];high=[]
        for condition in conditions:
            row=next(s for s in report['summary'] if s['field']==field and s['condition']==condition
                     and s['noise_K']==.002 and s['response_tau_s']==.3)
            stat=row['metrics']['tracking_fraction']
            means.append(100*stat['mean']);low.append(100*(stat['mean']-stat['minimum']));high.append(100*(stat['maximum']-stat['mean']))
        ax.bar(x+shift,means,width=.34,color=color,label=field.title(),yerr=np.array([low,high]),capsize=2)
    ax.set_xticks(x,['Fixed\nheading','Head +\nphase','Center +\nphase','Phase\n+90 deg','Shuffled\nphase'])
    ax.set(ylim=(0,105),ylabel='Tracking time (%)',title='12 paired initial states; whiskers show range')
    ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.15)
    ax=axes[1,1]
    for field,color in [('linear','#497fa7'),('radial','#008b82')]:
        rows=sorted([s for s in report['summary'] if s['field']==field and s['condition']=='head_phase' and s['noise_K']==.002],key=lambda s:s['response_tau_s'])
        ax.plot([r['response_tau_s'] for r in rows],[r['metrics']['center_error_rms_K']['mean'] for r in rows],'-o',label=field.title(),color=color)
    ax.set(xlabel='Turn-response time constant (s)',ylabel='Center temperature RMS error (K)',
           title='Finite response: fixed controller gains, noise 0.002 K',ylim=(0,None))
    ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.suptitle('2D isothermal tracking: prescribed kinematics, no neural training',fontsize=14)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180)
    plt.close(fig)
    print(args.output)


if __name__=='__main__':
    main()
