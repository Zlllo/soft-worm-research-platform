"""Summarize the completed failure probes without selecting models or seeds."""
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from diagnose_navigation_failures import ROOT, ASSETS, check, write


OUT = Path('docs/reports/assets/navigation-diagnosis-2026-11-02')


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def main():
    plan = json.loads((ROOT / 'design.json').read_text()); check(plan)
    failures = json.loads((ROOT / 'failure-decomposition.json').read_text())
    signals = json.loads((ROOT / 'signals.json').read_text())
    rescue = json.loads((ROOT / 'readout-rescue.json').read_text())
    temporal = json.loads((ROOT / 'teacher-temporal-probe.json').read_text())
    gain = json.loads((ROOT / 'temperature-gain.json').read_text())
    original = json.loads((ASSETS / 'feasibility-results.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(status='completed_posthoc_diagnosis_not_confirmation',
                  teacher_positive_control=temporal['positive_control'],
                  counts=dict(archived_specifications=len(plan['specs']),
                              teacher_windows=len(signals['teacher_indices']),
                              actor_snapshots=len(signals['metrics']), signal_conditions=len(signals['profiles']),
                              temporal_conditions=len(temporal['rows']), new_readout_fits=len(rescue['runs']),
                              readout_evaluation_episodes=sum(len(r[k]['cases']) for r in rescue['runs'] for k in ['baseline','readout_refit']),
                              gain_evaluation_episodes=sum(len(r['cases']) for r in gain['runs'])),
                  stages=[], signal=[], readout=[], temporal=[], temperature_gain=[],
                  limitations=plan['limitations'])
    for name in ['real8']+['rewired8-'+str(i) for i in [11,22,33,44,55,66,77,88,99,110]]+['isolated8','gru16']:
        for budget in [0, 65536, 262144]:
            for group in ['all', 'linear', 'radial', 'cold', 'hot']:
                rows = [r for r in failures['rows'] if r['scenario']=='replication15deg' and
                        r['spec']['name']==name and r['budget']==budget and r['group']==group]
                if rows:
                    result['stages'].append(dict(name=name, budget=budget, group=group,
                                                  **{k:mean(rows,k) for k in ['first_entry','pre_stable','post_stable','complete']}))
    for interface in ['full', 'restricted']:
        for checkpoint in ['initial','BC.pt','model-RL262144.pt']:
            rows = [r for r in signals['profiles'] if r['spec']['name']=='real8' and r['spec']['interface']==interface
                    and r['checkpoint']==checkpoint and r['pattern']=='phase' and r['amplitude_K']==.002]
            metrics = [r for r in signals['metrics'] if r['spec']['name']=='real8' and r['spec']['interface']==interface and r['checkpoint']==checkpoint]
            result['signal'].append(dict(interface=interface, checkpoint=checkpoint,
                                          action_rms_delta_rad_s=mean(rows,'action_rms_delta_rad_s'),
                                          node_final_rms_delta=np.mean([r['node_final_rms_delta'] for r in rows],axis=0).tolist(),
                                          tanh_saturation=np.mean([r['node_tanh_saturation'] for r in metrics],axis=0).tolist(),
                                          teacher_mse=mean(metrics,'teacher_mse')))
        rows = [r for r in rescue['runs'] if r['spec']['interface']==interface]
        result['readout'].append(dict(interface=interface, circuit_unchanged=all(r['circuit_unchanged'] for r in rows),
                                      by_seed=[dict(seed=r['spec']['seed'], before=r['baseline']['summary']['complete_success'],
                                                    after=r['readout_refit']['summary']['complete_success']) for r in rows],
                                      **{arm:{'success':float(np.mean([r[arm]['summary']['complete_success'] for r in rows])),
                                              'train_mse':float(np.mean([r[arm]['train_mse'] for r in rows])),
                                              'diagnostic_mse':float(np.mean([r[arm]['diagnostic_mse'] for r in rows]))}
                                         for arm in ['baseline','readout_refit']}))
    for name, interface in [('real8','full'),('real8','restricted'),('gru16','full')]:
        for bin_ in [[0,10],[10,20],[20,30],[30,41]]:
            rows = [r for r in temporal['rows'] if r['spec']['name']==name and r['spec']['interface']==interface
                    and r['checkpoint']=='BC.pt' and r['sample_bin']==bin_]
            result['temporal'].append(dict(name=name, interface=interface, sample_bin=bin_,
                                           actor_rms=mean(rows,'actor_rms_delta_rad_s'), teacher_rms=rows[0]['teacher_rms_delta_rad_s']))
    for scale in gain['design']['gains']:
        rows = [r for r in gain['runs'] if r['gain']==scale]
        result['temperature_gain'].append(dict(gain=scale,
                                               success=float(np.mean([r['summary']['complete_success'] for r in rows])),
                                               by_seed=[dict(seed=r['spec']['seed'], summary=r['summary']) for r in rows]))
    # The current-instance baseline must reproduce archived full-I/O BC behavior.
    reproduction = []
    for r in rescue['runs']:
        if r['spec']['interface']!='full':
            continue
        archived = next(v for v in original['runs'] if v['spec']['id']==r['spec']['id'])
        archived = next(v for v in archived['primary'] if v['learner_interactions']==0)['cases']
        keys = ['first_entry_s','pre_stable_s','post_stable_s','complete_success','full_tracking']
        differences = sum(a[k]!=b[k] for a,b in zip(archived,r['baseline']['cases']) for k in keys)
        reproduction.append(dict(seed=r['spec']['seed'], cases=len(archived), differing_fields=differences))
    result['archived_baseline_reproduction'] = reproduction
    write(OUT/'summary.json', result)
    fig, axes = plt.subplots(2,2,figsize=(12,8.2),layout='constrained')
    ax = axes[0,0]; keys=['first_entry','pre_stable','post_stable','complete']; positions=np.arange(4)
    for offset,(name,label) in enumerate([('real8','Real8'),('rewires','10 rewires, graph mean'),('gru16','GRU16')]):
        rows = [r for r in result['stages'] if r['budget']==262144 and r['group']=='all'
                and (r['name'].startswith('rewired8-') if name=='rewires' else r['name']==name)]
        ax.bar(positions+(offset-1)*.25,[mean(rows,k)*100 for k in keys],.25,label=label)
    ax.set(xticks=positions,xticklabels=['Enter','Pre-stable','Post-stable','Complete'],ylabel='Cases (%)',title='A  Archived 15-degree task: failure stage',ylim=(0,105));ax.legend(fontsize=8)
    ax=axes[0,1]
    for interface in ['full','restricted']:
        row=next(r for r in result['signal'] if r['interface']==interface and r['checkpoint']=='model-RL262144.pt')
        ax.plot(range(8),row['node_final_rms_delta'],'o-',label=interface)
    ax.set(yscale='log',xticks=range(8),xticklabels=signals['neurons'],ylabel='Node activity RMS change',title='B  Frozen response to 2 mK phase signal');ax.tick_params(axis='x',rotation=40);ax.legend()
    ax=axes[1,0]
    for j,row in enumerate(result['readout']):
        for i,seed in enumerate(row['by_seed']):
            ax.plot(np.array([0,1])+j*1.5,np.array([seed['before'],seed['after']])*100,'o-',label=f"{row['interface']} seed{seed['seed']}")
    ax.set(xticks=[0,1,1.5,2.5],xticklabels=['Full BC','+readout','Restricted BC','+readout'],ylabel='Complete cases (%)',ylim=(-2,105),title='C  Fixed circuit, 1,024 extra readout updates');ax.legend(fontsize=7)
    ax=axes[1,1]
    for name,interface,label in [('real8','full','Real8 full'),('real8','restricted','Real8 restricted'),('gru16','full','GRU16')]:
        rows=[r for r in result['temporal'] if r['name']==name and r['interface']==interface]
        ax.plot(range(4),[r['actor_rms'] for r in rows],'o-',label=label)
    ax.plot(range(4),[r['teacher_rms'] for r in rows],'k--',label='Local teacher')
    ax.set(yscale='log',xticks=range(4),xticklabels=['Oldest','Older','Recent','Newest'],ylabel='Command RMS change (rad/s)',title='D  BC response to time-local 2 mK signal');ax.legend(fontsize=8)
    fig.suptitle('Posthoc diagnosis; paired reused cases; three training seeds; no confidence intervals',fontsize=11)
    fig.savefig(OUT/'diagnosis.png',dpi=180);fig.savefig(OUT/'diagnosis.svg');plt.close(fig)
    sources=['analysis/diagnose_navigation_failures.py','analysis/probe_navigation_teacher.py',
             'analysis/probe_navigation_temperature_gain.py','analysis/report_navigation_diagnosis.py','tests/test_navigation_diagnosis.py']
    with zipfile.ZipFile(OUT/'diagnostic-records.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(ROOT.rglob('*')):
            if p.is_file():archive.write(p,str(p))
        for p in sources:archive.write(p,'source/'+p)
    write(OUT/'manifest.json',{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in OUT.iterdir() if p.name!='manifest.json'})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
