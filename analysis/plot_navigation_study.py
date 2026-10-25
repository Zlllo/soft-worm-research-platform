"""Scientific figures from reviewed hierarchical summaries and mechanism rows."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


ROOT=Path('results/navigation-factorial-2026-10-25')
OUT=Path('docs/reports/assets/navigation-study-2026-10-25')
BLUE='#2c5c9b';GOLD='#bf8b22';GREY='#737b83';OLIVE='#6a7740'
NAMES=['real8']+['rewired8-'+str(s) for s in [11,22,33,44,55,66,77,88,99,110]]+['isolated8','gru2','gru16']
LABELS=['Real']+['Rewired '+str(s) for s in [11,22,33,44,55,66,77,88,99,110]]+['Isolated','GRU2','GRU16']


def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True);fig.savefig(OUT/(name+'.png'),dpi=170);fig.savefig(OUT/(name+'.svg'))
    p=OUT/(name+'.svg');p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');plt.close(fig)


def main():
    summary=json.loads((ROOT/'summary.json').read_text());g=summary['groups']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    x=np.array([38400,103936,300544])/1000;budgets=['0','65536','262144']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,interface in zip(axes,['full','restricted']):
        null=[]
        for name in NAMES[1:11]:
            values=[g[f'{name}|{interface}|raw']['endpoints'][b]['complete_success']['mean']*100 for b in budgets]
            null.append(values);ax.plot(x,values,color=GREY,alpha=.25,linewidth=.8)
        arr=np.asarray(null);ax.fill_between(x,arr.min(0),arr.max(0),color=GREY,alpha=.08)
        ax.plot(x,arr.mean(0),color=GREY,marker='.',linewidth=1.8,label='10 rewired: mean / range')
        for name,color,label,style in [('real8',BLUE,'Real','-'),('gru16',GOLD,'GRU16','-'),('isolated8',OLIVE,'Isolated','--')]:
            key=f'{name}|'+('full' if name=='gru16' else interface)+'|raw'
            values=np.array([[row['primary'][b]['complete_success']*100 for b in budgets] for row in g[key]['seeds'].values()])
            if not len(values):continue
            avg=values.mean(0);ax.plot(x,avg,color=color,marker='o',linestyle=style,label=label,linewidth=1.8)
            ax.vlines(x,values.min(0),values.max(0),color=color,alpha=.5,linewidth=1.3)
        reference=summary['controls']['reference_correct']['summary']['complete_success']*100
        ax.axhline(reference,color='#222222',linestyle=':',linewidth=1.1,label='Local reference')
        ax.set_xscale('log');ax.minorticks_off();ax.set_xticks(x,['38.4','103.9','300.5']);ax.set_ylim(0,102)
        ax.set_xlabel('Teacher + learner interactions (thousands)');ax.set_ylabel('Complete navigation (%)')
        ax.set_title(interface.capitalize()+' circuit I/O');ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False,fontsize=9,loc='best')
    fig.suptitle('Learning after common demonstrations: fixed endpoints',fontsize=14);save(fig,'learning-curves')
    fig,axes=plt.subplots(1,2,figsize=(11,7),layout='constrained')
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#eeeeee')
    for ax,interface in zip(axes,['full','restricted']):
        values=np.full((len(NAMES),2),np.nan)
        for i,name in enumerate(NAMES):
            key=f'{name}|{interface}|raw'
            if key not in g:continue
            for j,metric in enumerate(['complete_success','full_tracking']):
                v=g[key]['endpoints']['262144'][metric]['mean'];values[i,j]=np.nan if v is None else v*100
        im=ax.imshow(values,vmin=0,vmax=100,cmap=cmap,aspect='auto')
        ax.set_yticks(range(len(LABELS)),LABELS);ax.set_xticks([0,1],['Complete success','Tracking time'])
        ax.set_title(interface.capitalize()+' I/O: final fixed budget')
        for i in range(len(NAMES)):
            for j in range(2):
                v=values[i,j];ax.text(j,i,'—' if np.isnan(v) else f'{v:.1f}',ha='center',va='center',color='white' if v>60 else '#222222',fontsize=9)
        ax.axhline(.5,color=GOLD,linewidth=1.5)
    fig.colorbar(im,ax=axes,label='Percent',shrink=.8);fig.suptitle('Raw shared encoding; three training seeds per cell',fontsize=14);save(fig,'final-topology')
    keys=list(summary['paired_adaptation']);fig,axes=plt.subplots(1,2,figsize=(11,6),layout='constrained')
    for ax,metric,title in zip(axes,['complete_success','full_tracking'],['Complete success change','Tracking time change']):
        for i,key in enumerate(keys):
            values=list(summary['paired_adaptation'][key][metric].values());values=np.array(values)*100
            ax.scatter(values,np.full(len(values),i),s=23,facecolors='none',edgecolors=BLUE)
            if len(values):ax.scatter(values.mean(),i,marker='D',color=GOLD,s=24)
        ax.axvline(0,color='#444444',linewidth=1);ax.set_yticks(range(len(keys)),[k.replace('|adaptive','').replace('|',' / ') for k in keys]);ax.invert_yaxis()
        ax.set_xlabel('Adaptive minus raw (percentage points)');ax.set_title(title);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Shared sensory adaptation: paired training seeds',fontsize=14);save(fig,'adaptation-pairs')
    diverging=LinearSegmentedColormap.from_list('orange_white_blue',[GOLD,'#ffffff',BLUE])
    conditions=['phase_none','phase_shifted','frequency_075','event_early90','center_sensor']
    titles=['No phase','Phase +90°','Sweep 0.75 Hz','Earlier event','Center sensor']
    fig,axes=plt.subplots(1,2,figsize=(13,7.5),layout='constrained')
    for ax,interface in zip(axes,['full','restricted']):
        values=np.full((len(NAMES),len(conditions)),np.nan)
        for i,name in enumerate(NAMES):
            key=f'{name}|{interface}|raw'
            if key in g:
                for j,c in enumerate(conditions):
                    v=g[key]['transfer'][c]['success_delta'];values[i,j]=np.nan if v is None else v*100
        im=ax.imshow(values,cmap=diverging,vmin=-100,vmax=100,aspect='auto')
        ax.set_yticks(range(len(LABELS)),LABELS);ax.set_xticks(range(len(titles)),titles,rotation=35,ha='right');ax.set_title(interface.capitalize()+' I/O')
        for i in range(len(NAMES)):
            for j in range(len(conditions)):
                v=values[i,j];ax.text(j,i,'—' if np.isnan(v) else f'{v:+.1f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>65 else '#222222')
    fig.colorbar(im,ax=axes,label='Complete success change (percentage points)',shrink=.8)
    fig.suptitle('Frozen final policies: paired diagnostic transfer',fontsize=14);save(fig,'frozen-transfer')
    mechanism_figures()


def mechanism_figures():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    mechanism=json.loads(Path('results/navigation-mechanisms-2026-10-25/results.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for step,color,label in [('5.0',GOLD,'Warming: tau45s'),('-5.0',BLUE,'Cooling: tau840s')]:
        a=np.asarray(mechanism['threshold_traces'][step]);axes[0,0].plot(a[:,0],(a[:,1]-293.15)/float(step),color=color,label=label)
    axes[0,0].set(xlabel='Time (s)',ylabel='Adapted threshold fraction',ylim=(0,1.03),title='Paper-anchored threshold relaxation');axes[0,0].legend(frameon=False)
    detection=mechanism['pulse_detection'];xs=np.arange(3)
    axes[0,1].bar(xs-.16,[r['fixed_increment_response'] for r in detection],width=.3,color=GREY,label='Fixed threshold')
    axes[0,1].bar(xs+.16,[r['adaptive_increment_response'] for r in detection],width=.3,color=BLUE,label='Adapted threshold')
    axes[0,1].set_xticks(xs,['15','20','25']);axes[0,1].set(xlabel='Background temperature (°C)',ylabel='Increment response (working units)',title='Same +0.02 K probe');axes[0,1].set_ylim(0,.56);axes[0,1].legend(frameon=False,loc='upper center',ncol=2,fontsize=9)
    for ax,metric,title,ylabel in [(axes[1,0],'mean_forward_run_s','Original-model motor feedback','Forward run duration (s)'),(axes[1,1],'directed_forward_bias','Original-model directed migration','Directed forward bias')]:
        for x,gain in enumerate([-1.,0.,1.]):
            vals=np.array([r['metrics'][metric] for r in mechanism['motor'] if r['task']=='source_static_gradient' and r['feedback']==gain])
            ax.bar(x,vals.mean(),color=BLUE if gain==1 else GREY,width=.55)
            ax.plot([x,x],[vals.min(),vals.max()],color='#222222',linewidth=1.3);ax.scatter(np.full(3,x),vals,color=GOLD,s=18,zorder=3)
        ax.set_xticks(range(3),['Negative','Zero','Positive']);ax.set(xlabel='Feedback gain',ylabel=ylabel,title=title)
    fig.suptitle('Functional mechanism controls; not a connectome ranking',fontsize=14);save(fig,'mechanism-controls')


if __name__=='__main__':
    import sys
    if '--mechanisms-only' in sys.argv:mechanism_figures()
    else:main()
