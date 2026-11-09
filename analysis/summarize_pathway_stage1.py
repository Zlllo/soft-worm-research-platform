"""Aggregate every declared arm/seed and archive the completed stage."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_pathway_stage1 import ARMS, REPO, check


def average(values):
    return float(np.mean(values))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('results/neural-pathway-stage1-2026-11-03'))
    parser.add_argument('--output',type=Path,default=Path('docs/reports/assets/pathway-stage1-2026-11-03'))
    args=parser.parse_args();root,out=args.root,args.output
    design=json.loads((root/'design.json').read_text());check(design)
    fits=[];evaluations=[];probes=[]
    for spec in design['specs']:
        for arm in ARMS:
            stem=f'{spec["id"]:03d}-{arm}'
            fit=json.loads((root/f'fit-{stem}'/'record.json').read_text())
            evaluation=json.loads((root/f'evaluation-{stem}.json').read_text())
            probe=json.loads((root/f'probe-{stem}.json').read_text())
            assert fit['complete'] and evaluation['complete'] and probe['complete']
            assert hashlib.sha256((root/f'fit-{stem}'/'model.pt').read_bytes()).hexdigest()==fit['checkpoint_sha256']==evaluation['checkpoint_sha256']
            fits.append(fit);evaluations.append(evaluation);probes.append(probe)
    controls=json.loads((root/'frozen-controls.json').read_text())
    assert controls['complete'] and len(controls['rows'])==78*3
    result=dict(status='completed_stage1_no_PPO_no_body_coupling',groups=[],
        counts=dict(new_fits=sum(r['updates']>0 for r in fits),snapshots=len(fits),
            new_case_evaluations=sum(len(r['fresh']['cases']) for r in evaluations),
            old_case_evaluations=sum(len(r['old']['cases']) for r in evaluations),
            lesion_new_case_evaluations=sum(len(r['lesion']['cases']) for r in evaluations),
            causal_probe_protocols=sum(len(p['rows']) for p in probes),frozen_control_probes=len(controls['rows'])),
        reference_preflight=json.loads((root/'reference-preflight.json').read_text())['summary'],
        summed_fit_wall_s=sum(r['wall_s'] for r in fits), limitations=design['limitations'])
    for interface in ('full','restricted'):
        for arm in ARMS:
            fs=[r for r in fits if r['spec']['interface']==interface and r['arm']==arm]
            es=[r for r in evaluations if r['spec']['interface']==interface and r['arm']==arm]
            ps=[r for r in probes if r['spec']['interface']==interface and r['arm']==arm]
            group=dict(interface=interface,arm=arm,by_seed=[],
                train_mse=average([r['final']['train_mse'] for r in fs]),
                old_diagnostic_mse=average([r['final']['diagnostic_mse'] for r in fs]),
                changed_parameters_verified=all(set(r['changed_names'])<=set(r['trainable_names']) for r in fs))
            for label in ('fresh','old','lesion'):
                group[label]=dict(complete_success=average([r[label]['summary']['complete_success'] for r in es]),
                    linear_post_stability=average([r[label]['summary']['linear']['post_stable_s']['successes']/16 for r in es]),
                    radial_post_stability=average([r[label]['summary']['radial']['post_stable_s']['successes']/16 for r in es]),
                    full_tracking=average([r[label]['summary']['full_tracking'] for r in es]))
            for f,e,p in zip(fs,es,ps):
                signal=next(v for v in p['rows'] if v['kind']=='phase' and v['frequency_Hz']==.5 and v['amplitude_K']==.002)
                step=next(v for v in p['rows'] if v['kind']=='step' and v['amplitude_K']==.002)
                group['by_seed'].append(dict(seed=f['spec']['seed'],fresh=e['fresh']['summary'],
                    old=e['old']['summary'],lesion=e['lesion']['summary'],
                    train_mse=f['final']['train_mse'],old_diagnostic_mse=f['final']['diagnostic_mse'],
                    calibration_gain=p['calibration_gain'],phase_2mK=signal,step_2mK=step))
            group['phase_2mK']=dict(node_rms=np.mean([r['phase_2mK']['node_rms'] for r in group['by_seed']],axis=0).tolist(),
                action_rms_rad_s=average([r['phase_2mK']['action_rms_rad_s'] for r in group['by_seed']]),
                gain_matched_normalized_rmse_by_seed=[r['phase_2mK']['gain_matched_normalized_rmse'] for r in group['by_seed']])
            result['groups'].append(group)
    result['frozen_controls']=controls
    # Independently compare the reproduced R arm with PR #3's paired old cases.
    parent_archive=Path('docs/reports/assets/navigation-diagnosis-2026-11-02/diagnostic-records.zip')
    verification=[]
    with zipfile.ZipFile(parent_archive) as z:
        for e in evaluations:
            if e['arm']!='R':continue
            old=json.loads(z.read(f'results/navigation-diagnosis-2026-11-02/readout-{e["spec"]["id"]:03d}.json'))
            keys=['seed','first_entry_s','pre_stable_s','post_stable_s','complete_success','full_tracking']
            assert len(old['readout_refit']['cases'])==len(e['old']['cases'])==32
            differing=sum(a[k]!=b[k] for a,b in zip(old['readout_refit']['cases'],e['old']['cases']) for k in keys)
            verification.append(dict(spec=e['spec'],case_differing_fields=differing))
    result['PR3_R_arm_reproduction']=verification
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

    colors={'full':'#3267ad','restricted':'#d58625'}
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
    for interface in ('full','restricted'):
        groups=[g for g in result['groups'] if g['interface']==interface]
        for ax,key,label in [(axes[0,0],'complete_success','Complete cases (%)'),
                             (axes[0,1],'linear_post_stability','Linear post-stable cases (%)')]:
            vals=[g['fresh'][key]*100 for g in groups]
            ax.plot(range(4),vals,'o-',color=colors[interface],label=interface)
            for i,g in enumerate(groups):
                for j,seed in enumerate(g['by_seed']):
                    s=seed['fresh']
                    v=s['complete_success']*100 if key=='complete_success' else s['linear']['post_stable_s']['successes']/16*100
                    ax.scatter(i+(j-1)*.055,v,s=22,alpha=.55,color=colors[interface])
            ax.set(xticks=range(4),xticklabels=ARMS,ylabel=label,ylim=(-3,103))
        axes[1,0].plot(range(4),[g['old_diagnostic_mse'] for g in groups],'o-',color=colors[interface],label=interface)
        for arm,style in [('frozen','--'),('All','-')]:
            g=next(g for g in groups if g['arm']==arm)
            axes[1,1].plot(range(8),g['phase_2mK']['node_rms'],'o'+style,label=f'{interface} {arm}',color=colors[interface])
    axes[0,0].set_title('A  Fresh 15-degree cases: all seeds retained')
    axes[0,1].set_title('B  Recovery in fresh linear fields')
    axes[1,0].set(xticks=range(4),xticklabels=ARMS,ylabel='Normalized teacher MSE',title='C  Old diagnostic episodes (not a new test set)')
    axes[1,1].set(xticks=range(8),xticklabels=['AFDL','AFDR','AIYL','AIYR','AIZL','AIZR','RIAL','RIAR'],yscale='log',
        ylabel='Activity RMS change (model units)',title='D  Causal online 2 mK phase response')
    axes[1,1].tick_params(axis='x',rotation=40)
    for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Equal 1,024-update interventions; 3 seeds; means and individual seeds, no confidence intervals',fontsize=10)
    fig.savefig(out/'stage1-overview.png',dpi=170);fig.savefig(out/'stage1-overview.svg');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,3.9),constrained_layout=True)
    for interface in ('full','restricted'):
        groups=[g for g in result['groups'] if g['interface']==interface]
        axes[0].plot(range(4),[g['fresh']['complete_success']*100 for g in groups],'o-',color=colors[interface],label=f'{interface} intact')
        axes[0].plot(range(4),[g['lesion']['complete_success']*100 for g in groups],'o--',color=colors[interface],label=f'{interface} AFD->AIY off')
        g=next(g for g in groups if g['arm']=='All')
        all_ref=[];all_les=[];t=None
        for spec in design['specs']:
            if spec['interface']!=interface:continue
            with np.load(root/f'probe-{spec["id"]:03d}-All.npz') as z:
                # Protocol 3 is the 2 mK step (ordering fixed by design).
                t=z['time_3'];all_ref.append(np.sqrt(np.mean(z['action_3']**2,axis=1)))
                all_les.append(np.sqrt(np.mean(z['lesion_action_3']**2,axis=1)))
        axes[1].plot(t,np.mean(all_ref,axis=0),color=colors[interface],label=f'{interface} All intact')
        axes[1].plot(t,np.mean(all_les,axis=0),'--',color=colors[interface],label=f'{interface} All lesion')
    axes[0].set(xticks=range(4),xticklabels=ARMS,ylabel='Complete cases (%)',ylim=(-3,103),title='AFD->AIY chemical intervention')
    axes[1].axvline(4,color='gray',lw=1)
    axes[1].set(xlabel='Time (s)',ylabel='Evoked command RMS (rad/s)',title='Online step response, no gain matching')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(out/'stage1-pathway.png',dpi=170);fig.savefig(out/'stage1-pathway.svg');plt.close(fig)

    files={}
    for path in root.rglob('*'):
        if path.is_file():files['evidence/'+str(path.relative_to(root))]=path
    source_names=list(design['source_sha256'])+['analysis/summarize_pathway_stage1.py',
        'docs/short-term-neural-physical-experiments-2026-11-03.md']
    report=Path('docs/reports/pathway-stage1-2026-11-03.md')
    if report.exists():source_names.append(str(report))
    for name in source_names:files[name]=REPO/name
    for name in ('stage1-overview.png','stage1-overview.svg','stage1-pathway.png','stage1-pathway.svg'):
        files['figures/'+name]=out/name
    manifests={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sorted(files.items())}
    archive=out/'reproduction.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,path in sorted(files.items()):z.write(path,name)
        z.writestr('manifest.json',json.dumps(manifests,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,sha in manifests.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
    (out/'manifest.json').write_text(json.dumps(dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=manifests),indent=2)+'\n')
    print(json.dumps({g['interface']+'-'+g['arm']:dict(fresh=g['fresh'],diagnostic_mse=g['old_diagnostic_mse']) for g in result['groups']},indent=2))
    print('Verified archive files:',len(files),flush=True)


if __name__=='__main__':main()
