"""Predeclared extra null graphs for frozen fast-wave robustness replication.

Existing3 graphs are development context; 7 new graph seeds and fresh cases
are identified separately. All graphs retained regardless of static success.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from tracking_learning_surface import atomic_json,cache_data,fit_one
from compare_tracking_topology import make_model
from train_tracking_network import evaluate,expert_for
from thermotaxis.isothermal_scene import TrackingConfig,BlindTrackingController,run_tracking
from thermotaxis.tracking_learning import NeuralTrackingController
from thermotaxis.phased_field import PhasedTemperatureField,phased_tracking_metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    out=Path('results/tracking-expanded-graphs-2026-10-24')
    sources=['analysis/expand_tracking_null_graphs.py','analysis/tracking_learning_surface.py',
             'analysis/compare_tracking_topology.py','analysis/train_tracking_network.py',
             'src/thermotaxis/tracking_circuit.py','src/thermotaxis/tracking_learning.py',
             'src/thermotaxis/isothermal_scene.py','src/thermotaxis/moving_isotherm.py','src/thermotaxis/phased_field.py']
    if not out.exists():
        out.mkdir(parents=True)
        plan=dict(status='exploratory_new_graph_and_case_replication',old_graph_seeds=[11,22,33],
            new_graph_seeds=[44,55,66,77,88,99,110],training_seeds=[7,17,27],
            episodes_per_field=16,updates=2048,interface='full',
            case_seeds=[10301,10302],phases_rad=[0.,np.pi/2,np.pi,3*np.pi/2],
            wave_amplitude_K=.05,wave_period_s=60.,fields=['linear','radial'],
            primary_metric='Mean relative-to-front tracking across phases, fields and cases, averaged over training seeds per graph.',
            failure_threshold=.5,static_competence_threshold=.99,
            hierarchy='Graph, training seed and case reported separately; new7 and combined10 graph summaries both retained.',
            retention='Keep all graphs, repeated layouts and failed static fits; no performance-dependent exclusions.',
            source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources})
        atomic_json(out/'design.json',plan)
        for n in sources:
            p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,p)
    elif not args.resume:raise RuntimeError('use --resume for an interrupted graph replication')
    plan=json.loads((out/'design.json').read_text())
    for n,h in plan['source_sha256'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
    data,val,_=cache_data()
    from compare_tracking_topology import subset
    data=subset(data,16)
    path=out/'fitting.json'
    fitting=json.loads(path.read_text()) if path.exists() else dict(design=plan,runs=[])
    existing={(r['model'],r['seed']) for r in fitting['runs']}
    for name in ['real8','gru16']+['rewired8-'+str(s) for s in plan['old_graph_seeds']+plan['new_graph_seeds']]:
        for seed in plan['training_seeds']:
            if (name,seed) in existing:continue
            if name.startswith('rewired8-') and int(name.split('-')[1]) in plan['new_graph_seeds']:
                row=fit_one(name,seed,16,data,val,{},out)
            else:
                shard='grus' if name=='gru16' else 'rates'
                parent=Path('results/tracking-learning-surface-2026-10-24')/shard
                old=json.loads((parent/'results.json').read_text())
                record=next(r for r in old['runs'] if r['model']==name and r['seed']==seed and r['episodes_per_field']==16)
                row=json.loads(json.dumps(record));row['new_fit']=False
                for endpoint in row['endpoints'].values():
                    source=parent/endpoint['checkpoint'];shutil.copyfile(source,out/endpoint['checkpoint'])
                    endpoint['reused_from']=str(source)
            fitting['runs'].append(row);atomic_json(path,fitting)
            print('GRAPH FIT',name,seed,'new' if row['new_fit'] else 'reused',flush=True)
    fitting['complete']=True;atomic_json(path,fitting)
    result_path=out/'results.json'
    result=json.loads(result_path.read_text()) if result_path.exists() else dict(design=plan,runs=[],graph_masks={})
    completed={(r['model'],r['training_seed']) for r in result['runs']}
    specs=[(r['model'],r['seed'],r) for r in fitting['runs']]+[(n,None,None) for n in ['expert','blind']]
    for name,seed,fit in specs:
        if (name,seed) in completed:continue
        model=None
        if fit is not None:
            model=make_model(name,'demeaned','full')
            model.load_state_dict(torch.load(out/fit['endpoints']['2048']['checkpoint'],map_location='cpu',weights_only=True));model.eval()
            if hasattr(model,'structural_metadata'):result['graph_masks'][name]=model.structural_metadata()['chemical_mask']
        base=TrackingConfig()
        static=evaluate(name if model is None else model,base,plan['case_seeds'])
        waves=[]
        for phase in plan['phases_rad']:
            for field in plan['fields']:
                config=replace(base,field=field,uniform_wave_amplitude_K=.05,uniform_wave_period_s=60.)
                if field=='radial':config=replace(config,target_radius_m=base.target_radius_m-.05*np.sin(phase)/base.gradient_K_m)
                for case in plan['case_seeds']:
                    controller=expert_for(config) if name=='expert' else BlindTrackingController() if name=='blind' else NeuralTrackingController(model)
                    row=run_tracking(config,PhasedTemperatureField(controller,config,phase),case,record_trajectory=True)
                    waves.append(dict(field=field,seed=case,phase_rad=phase,metrics=phased_tracking_metrics(config,row,phase)))
        result['runs'].append(dict(model=name,training_seed=seed,static=static,waves=waves,
            static_mean_tracking=float(np.mean([r['metrics']['tracking_fraction'] for r in static])),
            moving_mean_tracking=float(np.mean([r['metrics']['moving_tracking_fraction'] for r in waves]))))
        atomic_json(result_path,result);print('GRAPH CASES',name,seed,flush=True)
    assert len(result['runs'])==38
    result['source_hashes_unchanged']={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in plan['source_sha256'].items()}
    atomic_json(result_path,result)


if __name__=='__main__':main()
