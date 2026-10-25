"""Validate paired interventions, summarize model/seed hierarchy, export figures."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path('results/tracking-timescale-2026-10-25')
OUT=Path('docs/reports/assets/tracking-timescale-2026-10-25')


def mean_metric(row, scene, late=False, field=None):
    cells=[c for c in row['conditions'] if c['scene']==scene and (field is None or c['field']==field)]
    return float(np.mean([c['late_metrics' if late else 'metrics']['moving_tracking_fraction'] for c in cells]))


def main():
    d=json.loads((ROOT/'results.json').read_text());p=d['design']
    assert d['complete'] and len(d['runs'])==p['expected_conditions']
    assert all(d['source_hashes_unchanged'].values()) and all(d['checkpoint_hashes_unchanged'].values())
    keys={(r['model'],r['training_seed'],r['tau_s']):r for r in d['runs']}
    assert len(keys)==len(d['runs'])
    summary={'models':{},'controls':{},'paired_checks':0,'result_sha256':hashlib.sha256((ROOT/'results.json').read_bytes()).hexdigest()}
    for name in p['models']:
        summary['models'][name]={}
        for tau in p['tau_s']:
            rows=[keys[(name,seed,tau)] for seed in p['training_seeds']]
            base=[keys[(name,seed,.3)] for seed in p['training_seeds']]
            for row,ref in zip(rows,base):
                assert row['checkpoint']==ref['checkpoint']
                for a,b in zip(row['conditions'],ref['conditions']):
                    for k in ['scene','phase_rad','field','case_seed','initial_state']:assert a[k]==b[k],(name,tau,k)
                    summary['paired_checks']+=1
            item={}
            for label,scene,late in [('static','static',False),('moving','moving',False),('late_moving','moving',True)]:
                vals=[mean_metric(r,scene,late) for r in rows]
                changes=[a-mean_metric(b,scene,late) for a,b in zip(vals,base)]
                item[label]={'mean':float(np.mean(vals)),'by_training_seed':dict(zip(map(str,p['training_seeds']),vals)),
                    'min':min(vals),'max':max(vals),'paired_delta':float(np.mean(changes)),
                    'paired_deltas_by_seed':dict(zip(map(str,p['training_seeds']),changes)),
                    'by_field':{f:float(np.mean([mean_metric(r,scene,late,f) for r in rows])) for f in p['fields']}}
            summary['models'][name][str(tau)]=item
    for name in ['gru16','expert','blind']:
        rows=[r for r in d['runs'] if r['model']==name]
        summary['controls'][name]={label:float(np.mean([mean_metric(r,scene,late) for r in rows]))
            for label,scene,late in [('static','static',False),('moving','moving',False),('late_moving','moving',True)]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,7),layout='constrained')
    labels=['Real graph']+['Rewired '+n.split('-')[1] for n in p['models'][1:]]
    for ax,scene,title in zip(axes,['static','moving'],['Static field','Moving isotherm']):
        values=np.array([[summary['models'][n][str(t)][scene]['mean']*100 for t in p['tau_s']] for n in p['models']])
        image=ax.imshow(values,vmin=0,vmax=100,cmap='Blues',aspect='auto')
        ax.set_xticks(range(3),['0.1','0.3 (trained)','0.9']);ax.set_xlabel('Node relaxation time (s)')
        ax.set_yticks(range(len(labels)),labels);ax.set_title(title,pad=12)
        for i in range(len(labels)):
            for j in range(3):ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',color='white' if values[i,j]>60 else '#222222',fontsize=10)
        ax.axhline(.5,color='#c48b21',linewidth=1.5)
    fig.colorbar(image,ax=axes,label='Tracking time (%)',shrink=.8)
    fig.suptitle('Frozen weights: response-time sensitivity',fontsize=16)
    fig.savefig(OUT/'timescale-sensitivity.png',dpi=170)
    fig.savefig(OUT/'timescale-sensitivity.svg')
    plt.close(fig)
    for name in ['design.json','results.json']:shutil.copyfile(ROOT/name,OUT/name)
    with zipfile.ZipFile(OUT/'models-and-sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for source in sorted(ROOT.rglob('*')):
            if source.is_file():z.write(source,source.relative_to(ROOT))
        for rel in ['analysis/summarize_tracking_timescale.py','tests/test_tracking_interventions.py']:
            z.write(rel,'additional-source/'+rel)
    svg=OUT/'timescale-sensitivity.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    manifest={file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(OUT.iterdir()) if file.is_file() and file.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('paired checks',summary['paired_checks'])
    for name in p['models']:
        print(name,[(tau,round(summary['models'][name][str(tau)]['static']['mean']*100,2),
            round(summary['models'][name][str(tau)]['moving']['mean']*100,2),
            round(summary['models'][name][str(tau)]['late_moving']['mean']*100,2)) for tau in p['tau_s']])
    print('controls',summary['controls'])


if __name__=='__main__':main()
