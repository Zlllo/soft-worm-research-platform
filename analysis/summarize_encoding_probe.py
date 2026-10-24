"""Check completeness and summarize paired encoding contrasts descriptively."""
import argparse
import json
from pathlib import Path

import numpy as np


def key(run):
    return tuple(run[k] for k in ('architecture','encoding','topology_seed','seed'))


def stats(values):
    values=np.asarray(values,dtype=float)
    return dict(n=len(values),mean=float(values.mean()),minimum=float(values.min()),maximum=float(values.max()))


def summarize(report):
    design=report['design']
    planned={key(x) for x in design['runs']}
    runs={key(x):x for x in report['runs']}
    if len(runs)!=len(report['runs']) or set(runs)!=planned:
        raise ValueError('duplicate, unexpected or missing experiment cases')
    if any(x['curves'][-1]['update']!=design['updates'] for x in runs.values()):
        raise ValueError('incomplete training budget')
    summary=dict(status='complete_exploratory_supervised_grid',run_count=len(runs),
        interpretation='Descriptive paired training outcomes, not RL efficiency or independent behavior-replicate inference',
        constant_predictor_mse=report['constant_predictor_mse'],teacher_behavior=report['teacher_behavior'],
        architectures={})
    architectures=list(dict.fromkeys(x['architecture'] for x in design['runs']))
    for arch in architectures:
        rows=[x for x in runs.values() if x['architecture']==arch]
        parameters={x['parameters'] for x in rows}
        if len(parameters)!=1:
            raise ValueError('unexpected parameter-count mismatch within architecture')
        entry=dict(parameters=parameters.pop(),encodings={},paired_deltas=[],by_topology={})
        for encoding in ('raw','adaptive'):
            selected=[x for x in rows if x['encoding']==encoding]
            if not selected:
                raise ValueError('both encoding conditions required')
            entry['encodings'][encoding]=dict(
                mse=stats([x['curves'][-1]['validation_mse'] for x in selected]),
                residence=stats([x['behavior_validation']['residence_fraction'] for x in selected]),
                mae_K=stats([x['behavior_validation']['mean_absolute_error_K'] for x in selected]),
                holdouts={case:stats([x['exploratory_holdouts'][case]['residence_fraction'] for x in selected])
                    for case in selected[0]['exploratory_holdouts']})
        for raw in [x for x in rows if x['encoding']=='raw']:
            adapted=runs[(arch,'adaptive',raw['topology_seed'],raw['seed'])]
            entry['paired_deltas'].append(dict(topology_seed=raw['topology_seed'],seed=raw['seed'],
                residence_delta=adapted['behavior_validation']['residence_fraction']-raw['behavior_validation']['residence_fraction'],
                mse_delta=adapted['curves'][-1]['validation_mse']-raw['curves'][-1]['validation_mse']))
        for topo in sorted({x['topology_seed'] for x in rows}):
            entry['by_topology'][str(topo)]={encoding:stats([x['behavior_validation']['residence_fraction'] for x in rows
                if x['encoding']==encoding and x['topology_seed']==topo]) for encoding in ('raw','adaptive')}
        summary['architectures'][arch]=entry
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('results/encoding-topology-2026-10-24'))
    args=parser.parse_args()
    report=json.loads((args.root/'results.json').read_text())
    result=summarize(report)
    (args.root/'paired_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for arch,x in result['architectures'].items():
        raw,adapt=x['encodings']['raw'],x['encodings']['adaptive']
        print(arch,'params',x['parameters'],'raw/adaptive residence',raw['residence']['mean'],adapt['residence']['mean'],
              'raw/adaptive MSE',raw['mse']['mean'],adapt['mse']['mean'])


if __name__=='__main__':
    main()
