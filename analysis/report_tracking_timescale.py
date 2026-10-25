"""Produce the readable timescale report from completed frozen and refit batches."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from summarize_tracking_timescale import mean_metric


ROOT=Path('results/tracking-timescale-refit-real-2026-10-25')
OUT=Path('docs/reports/assets/tracking-timescale-2026-10-25')


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    refit=json.loads((ROOT/'results.json').read_text())
    frozen=json.loads((OUT/'results.json').read_text())
    assert refit['complete'] and refit['fitting_complete'] and len(refit['runs'])==6 and len(refit['fits'])==6
    assert all(refit['source_hashes_unchanged'].values())
    fit_summary={}
    for tau in [.1,.9]:
        rows=[r for r in refit['runs'] if r['tau_s']==tau]
        fit_summary[str(tau)]={}
        for label,scene,late in [('static','static',False),('moving','moving',False),('late_moving','moving',True)]:
            vals=[mean_metric(r,scene,late) for r in rows]
            fit_summary[str(tau)][label]={'mean':float(np.mean(vals)),
                'by_training_seed':{str(r['training_seed']):v for r,v in zip(rows,vals)},'min':min(vals),'max':max(vals)}
        for row in rows:
            source=next(r for r in frozen['runs'] if r['model']=='real8' and r['training_seed']==row['training_seed'] and r['tau_s']==tau)
            for a,b in zip(row['conditions'],source['conditions']):
                for key in ['scene','field','phase_rad','case_seed','initial_state']:assert a[key]==b[key]
    (OUT/'refit-summary.json').write_text(json.dumps(fit_summary,indent=2)+'\n')
    for name in ['design.json','results.json']:shutil.copyfile(ROOT/name,OUT/('refit-'+name))
    with zipfile.ZipFile(OUT/'refit-models-and-sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(ROOT))
        for name in ['analysis/report_tracking_timescale.py', 'analysis/write_tracking_timescale_report.py']:
            z.write(name,'additional-source/'+name)
    fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
    colors=['#2c5c9b','#bf8b22']
    for ax,scene,title in zip(axes,['static','moving'],['Static field','Moving isotherm']):
        for mode,color,marker in [('Frozen',colors[0],'o'),('Refitted',colors[1],'s')]:
            values=[]
            for tau in [.1,.3,.9]:
                v=summary['models']['real8'][str(tau)][scene] if mode=='Frozen' or tau==.3 else fit_summary[str(tau)][scene]
                values.append(v['mean']*100)
                ax.plot([tau,tau],[v['min']*100,v['max']*100],color=color,alpha=.5,linewidth=2)
            ax.plot([.1,.3,.9],values,color=color,marker=marker,label=mode,linewidth=1.7)
        ax.set_xscale('log');ax.set_xticks([.1,.3,.9],['0.1','0.3','0.9']);ax.set_ylim(0,102)
        ax.set_xlabel('Node relaxation time (s)');ax.set_ylabel('Tracking time (%)');ax.set_title(title)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle('Real graph: frozen intervention versus fixed-budget refit',fontsize=13)
    fig.savefig(OUT/'real-graph-refit.png',dpi=170);fig.savefig(OUT/'real-graph-refit.svg');plt.close(fig)
    svg=OUT/'real-graph-refit.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(fit_summary,indent=2))


if __name__=='__main__':main()
