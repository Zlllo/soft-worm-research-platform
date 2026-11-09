"""Posthoc check: distinguish global clock reset from phase-aligned gait error."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from thermotaxis.neuromechanics.boyle_output import curvature


def cycles(path,after):
    with np.load(path) as z:t=z['time_s'];k=curvature(z['pose'])
    head=k[:,:5].mean(1)
    ids=np.flatnonzero((head[:-1]<=0)&(head[1:]>0))
    crossing=t[ids]-head[ids]*np.diff(t)[ids]/(head[ids+1]-head[ids])
    shapes=[];periods=[];ends=[]
    for start,end in zip(crossing[:-1],crossing[1:]):
        if start<after:continue
        sample=start+np.arange(100)/100*(end-start)
        shapes.append(np.stack([np.interp(sample,t,k[:,j]) for j in range(47)],axis=1))
        periods.append(float(end-start));ends.append(float(end))
    return np.asarray(shapes),np.asarray(periods),ends


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/body-load-stage2-2026-11-03'))
    p.add_argument('--output',type=Path,default=Path('docs/reports/assets/body-load-stage2-2026-11-03'))
    a=p.parse_args();root=a.root;out=a.output
    design_path=root/'phase-alignment-design.json'
    if not design_path.exists():
        d=dict(status='supplemental_after_fixed_clock_metrics',
            reason='fixed-clock lag and neural differences can also result from an overall phase reset',
            alignment='upward head-curvature zero crossings, linear interpolation to 100 phase points; no optimized time warping',
            reference='mean of all complete nominal cycles starting after 3 s',
            primary='normalized posterior shape RMSE in last two complete cycles after pulse removal; period separately reported',
            exclusion='no performance filtering; insufficient complete cycles reported explicitly',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        design_path.write_text(json.dumps(d,indent=2)+'\n')
    else:
        d=json.loads(design_path.read_text())
        assert d['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    base,periods,_=cycles(root/'nominal-live/trajectory.npz',3.)
    template=base.mean(0);scale=float(np.std(template[:,24:]));period=float(np.median(periods))
    original=json.loads((root/'conditions.json').read_text());rows=[]
    for row in original['rows']:
        shape,p,ends=cycles(root/row['path']/'trajectory.npz',row['onset_s']+2.)
        errors=np.sqrt(np.mean((shape[:,:,24:]-template[None,:,24:])**2,axis=(1,2)))/scale if len(shape) else []
        rows.append(dict(**row,complete_cycles=len(shape),cycle_end_s=ends,shape_errors=list(map(float,errors)),
            periods_s=p.tolist(),last_two_shape_rmse=None if len(shape)<2 else float(np.mean(errors[-2:])),
            last_two_period_change_fraction=None if len(p)<2 else float(np.mean(p[-2:])/period-1)))
    groups=[]
    for factor in [1.,1.5,2.]:
        for mode in ['live','replay']:
            selected=[r for r in rows if r['factor']==factor and r['feedback']==mode]
            groups.append(dict(factor=factor,feedback=mode,insufficient_cycles=sum(r['complete_cycles']<2 for r in selected),
                last_two_shape_rmse=float(np.mean([r['last_two_shape_rmse'] for r in selected if r['last_two_shape_rmse'] is not None])),
                last_two_period_change_fraction=float(np.mean([r['last_two_period_change_fraction'] for r in selected if r['last_two_period_change_fraction'] is not None]))))
    result=dict(design=d,nominal_cycle_variability=float(np.mean(np.sqrt(np.mean((base[:,:,24:]-template[None,:,24:])**2,axis=(1,2)))/scale)),
        rows=rows,groups=groups)
    (root/'phase-aligned.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'phase-aligned.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    for mode,color in [('live','#2d6bad'),('replay','#c98024')]:
        for ax,key,label in [(axes[0],'last_two_shape_rmse','Phase-aligned posterior shape error'),
                             (axes[1],'last_two_period_change_fraction','Gait period change (fraction)')]:
            for i,factor in enumerate([1.5,2.]):
                values=[r[key] for r in rows if r['factor']==factor and r['feedback']==mode]
                ax.scatter(np.full(8,i+(-.06 if mode=='live' else .06)),values,color=color,s=20,alpha=.6)
            ax.plot([0,1],[g[key] for g in groups if g['feedback']==mode and g['factor']>1],'o-',color=color,label=mode)
            ax.set(xticks=[0,1],xticklabels=['1.5 x','2 x'],ylabel=label);ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].axhline(result['nominal_cycle_variability'],color='gray',ls=':',label='nominal variability')
    fig.suptitle('Supplemental phase-aligned check; all pulse phases retained',fontsize=11)
    fig.savefig(out/'phase-aligned.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(nominal_cycle_variability=result['nominal_cycle_variability'],groups=groups),indent=2))


if __name__=='__main__':main()
