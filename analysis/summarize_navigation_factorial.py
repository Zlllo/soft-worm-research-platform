"""Hierarchical summaries of a completed navigation factorial and controls."""
import json
from pathlib import Path

import numpy as np


ROOT=Path('results/navigation-factorial-2026-10-25')


def mean(values):return float(np.mean(values)) if values else None


def main():
    d=json.loads((ROOT/'results.json').read_text());assert d['complete'] and len(d['runs'])==105
    assert all(d['source_hashes_unchanged'].values()) and all(d['input_hashes_unchanged'].values())
    groups={}
    for row in d['runs']:
        s=row['spec'];key='|'.join([s['name'],s['interface'],s['encoding']])
        group=groups.setdefault(key,dict(name=s['name'],interface=s['interface'],encoding=s['encoding'],seeds={},failed_seeds=[]))
        if row['status']!='complete':group['failed_seeds'].append(s['seed']);continue
        primary={str(x['learner_interactions']):x for x in row['primary']}
        for endpoint in primary.values():
            displaced=[r for r in endpoint['cases'] if r['reentry_s'] is None or abs(r['reentry_s'])>1e-8]
            recovered=[r for r in displaced if r['reentry_s'] is not None]
            endpoint['summary']['event_displaced_fraction']=len(displaced)/len(endpoint['cases'])
            endpoint['summary']['recovery_success_among_displaced']=len(recovered)/len(displaced) if displaced else None
            endpoint['summary']['conditional_displaced_reentry_s']=mean([r['reentry_s'] for r in recovered])
        threshold=d['design']['learning_threshold'];reached=None
        for budget in [0,65536,262144]:
            metrics=primary[str(budget)]['summary']
            if metrics['complete_success']>=threshold['complete_success'] and metrics['post_tracking']>=threshold['post_tracking']:
                reached=38400+budget;break
        base=next(x for x in row['primary'] if x['learner_interactions']==262144)
        paired_base=[x for x in base['cases'] if x['seed'] in d['design']['transfer_seeds']]
        assert len(paired_base)==16
        sampling=json.loads((Path('results/navigation-sampling-2026-10-25')/f'evaluation-{s["id"]:03d}.json').read_text())
        transfers={}
        for name,transfer in row['transfer'].items():
            assert [x['seed'] for x in transfer['cases']]==[x['seed'] for x in paired_base]
            transfers[name]=dict(summary=transfer['summary'],paired_base_success=mean([x['complete_success'] for x in paired_base]),
                success_delta=transfer['summary']['complete_success']-mean([x['complete_success'] for x in paired_base]),
                tracking_delta=transfer['summary']['full_tracking']-mean([x['full_tracking'] for x in paired_base]))
        transfers['center_sensor']=dict(summary=sampling['summary'],paired_base_success=mean([x['complete_success'] for x in paired_base]),
            success_delta=sampling['summary']['complete_success']-mean([x['complete_success'] for x in paired_base]),
            tracking_delta=sampling['summary']['full_tracking']-mean([x['full_tracking'] for x in paired_base]))
        group['seeds'][str(s['seed'])]=dict(primary={budget:v['summary'] for budget,v in primary.items()},
            threshold_total_interactions=reached,threshold_censored=reached is None,transfer=transfers)
    for group in groups.values():
        group['endpoints']={}
        for budget in [0,65536,262144]:
            metrics=[v['primary'][str(budget)] for v in group['seeds'].values()]
            group['endpoints'][str(budget)]={k:dict(mean=mean([x[k] for x in metrics]),
                by_training_seed={s:v['primary'][str(budget)][k] for s,v in group['seeds'].items()})
                for k in ['complete_success','full_tracking','pre_tracking','post_tracking','residence']}
        group['threshold_reached']=sum(v['threshold_total_interactions'] is not None for v in group['seeds'].values())
        group['threshold_not_reached']=sum(v['threshold_total_interactions'] is None for v in group['seeds'].values())
        group['transfer']={name:{k:mean([v['transfer'][name][k] for v in group['seeds'].values()]) for k in ['success_delta','tracking_delta']}
            for name in ['phase_none','phase_shifted','frequency_075','event_early90','center_sensor']}
    paired_adaptation={}
    for key,group in groups.items():
        if group['encoding']!='adaptive':continue
        raw=groups['|'.join([group['name'],group['interface'],'raw'])]
        paired_adaptation[key]={metric:{s:group['seeds'][s]['primary']['262144'][metric]-raw['seeds'][s]['primary']['262144'][metric]
            for s in group['seeds'] if s in raw['seeds']} for metric in ['complete_success','full_tracking','post_tracking']}
    null_comparisons={}
    for interface in ['full','restricted']:
        real=groups['real8|'+interface+'|raw']['endpoints']['262144']['complete_success']['mean']
        graphs={str(seed):groups[f'rewired8-{seed}|{interface}|raw']['endpoints']['262144']['complete_success']['mean']
            for seed in [11,22,33,44,55,66,77,88,99,110]}
        vals=[v for v in graphs.values() if v is not None]
        null_comparisons[interface]=dict(real_mean=real,null_graph_means=graphs,mean_of_graph_means=mean(vals),
            graphs_above_real=sum(v>real for v in vals) if real is not None else None,graphs_equal_real=sum(v==real for v in vals) if real is not None else None)
    result=dict(groups=groups,paired_adaptation=paired_adaptation,null_comparisons=null_comparisons,
        controls=d['controls'],fit_count=len(d['fits']),failed_fits=sum(r['status']!='complete' for r in d['fits']),
        primary_cases=sum(len(e['cases']) for r in d['runs'] for e in r['primary']),
        transfer_cases=sum(len(e['cases']) for r in d['runs'] for e in r['transfer'].values()),
        sampling_cases=sum(len(json.loads((Path('results/navigation-sampling-2026-10-25')/f'evaluation-{r["spec"]["id"]:03d}.json').read_text()).get('cases',[])) for r in d['runs']))
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('null comparisons',result['null_comparisons'])
    for key in ['real8|full|raw','real8|restricted|raw','gru16|full|raw','isolated8|full|raw']:
        g=groups[key];print(key,{b:g['endpoints'][b]['complete_success']['mean'] for b in ['0','65536','262144']},'threshold reached',g['threshold_reached'])


if __name__=='__main__':main()
