"""AFD->AIY necessity versus strength-matched lesions on calibrated real8 models."""
import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_body_input_small_step import evaluate
from diagnose_navigation_failures import digest,representation,write
from run_active_sensing_stage3 import make_features,metric
from thermotaxis.navigation_window_policy import WindowActorCritic

REPO=Path(__file__).resolve().parents[1]
PARENT=Path('results/body-topology-calibration-2026-11-03')
DATA=Path('results/active-sensing-stage3-2026-11-03')


def prepare(root):
    if root.exists():raise FileExistsError('preserve prior probes')
    parent=json.loads((PARENT/'design.json').read_text())
    models=[]
    for init in parent['initializations']:
        for s in parent['specs']:
            if s['name']!='real8':continue
            folder=PARENT/'fits'/f'{init}-{s["id"]:03d}'
            record=json.loads((folder/'record.json').read_text())
            assert record['complete'] and digest(folder/'model.pt')==record['model_sha256']
            models.append(dict(spec=s,initialization=init,path=str(folder/'model.pt')))
    assert len(models)==6
    sources=['analysis/probe_calibrated_thermal_path.py','analysis/diagnose_navigation_failures.py',
             'analysis/run_active_sensing_stage3.py','src/thermotaxis/navigation_window_policy.py']
    inputs=[DATA/'dataset.npz',PARENT/'new-body-scenes.npz']+[Path(m['path']) for m in models]
    d=dict(status='declared_before_any_lesion_result',models=models,target='existing AFD->AIY chemical edges',
        fractions=[.5,1.],control_rule='other off-diagonal pairs, excluding target; pair L1>=target removal L1; closest20 pairs by L1',
        control_fraction='target removal L1 / pair L1; no sign flip; same number of altered edges',
        probe_rule='standardized RIA pair; ridge fitted on old phase0,2,4,6; lambda chosen on1,5, not test',
        ridge_lambdas=[.0001,.01,1.,100.],
        limits=['All six real-model endpoints retained, regardless of intact accuracy.',
                'Matched-weight perturbation controls, not a uniform random topology null.',
                'Input routing at AFD/AIY and RIA output is an engineering/anatomical convention; necessity is conditional on it.'],
        source_sha256={s:digest(REPO/s) for s in sources},inputs_sha256={str(p):digest(REPO/p) for p in inputs})
    write(root/'design.json',d)


def check(d):
    for p,h in {**d['source_sha256'],**d['inputs_sha256']}.items():
        if digest(REPO/p)!=h:raise RuntimeError('changed probe input/source: '+p)


def altered_model(model,indices,fraction):
    altered=copy.deepcopy(model)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    with torch.no_grad():altered.chemical[indices]=altered.chemical[indices]*(1-fraction)
    assert all(torch.equal(v,altered.state_dict()[k]) for k,v in before.items() if k!='chemical')
    keep=np.ones(len(model.chemical),bool);keep[indices]=False
    assert torch.equal(altered.chemical[keep],model.chemical[keep])
    return altered,float((model.chemical-altered.chemical).detach().abs().sum())


def representation_array(model,x):
    with torch.no_grad():
        return torch.cat([representation(model,torch.from_numpy(x[i:i+1024])) for i in range(0,len(x),1024)]).numpy()


def ridge_transfer(model,x,data,xf,fresh,d):
    h=representation_array(model,x);hf=representation_array(model,xf)
    train=data['split']==0;val=data['split']==1
    mean=h[train].mean(0);scale=np.maximum(h[train].std(0),1e-8)
    a=np.column_stack([np.ones(len(h)),(h-mean)/scale]);af=np.column_stack([np.ones(len(hf)),(hf-mean)/scale])
    xx=a[train].T@a[train];xy=a[train].T@(data['label'][train]*2-1);choices=[]
    for lam in d['ridge_lambdas']:
        reg=np.eye(3)*lam;reg[0,0]=0;w=np.linalg.solve(xx+reg,xy)
        acc=metric(data['label'][val],a[val]@w,data['run'][val])['balanced_accuracy']
        choices.append((acc,lam,w))
    _,lam,w=max(choices,key=lambda r:r[0]);score=af@w
    return dict(lambda_selected=lam,all=metric(fresh['label'],score,fresh['run']),
        matched_load=metric(fresh['label'][fresh['family']==0],score[fresh['family']==0],fresh['run'][fresh['family']==0]),
        novel_load=metric(fresh['label'][fresh['family']==1],score[fresh['family']==1],fresh['run'][fresh['family']==1]))


def run(root,d):
    with np.load(DATA/'dataset.npz') as z:data={k:z[k] for k in z.files}
    with np.load(PARENT/'new-body-scenes.npz') as z:fresh={k:z[k] for k in z.files}
    x=make_features(data,np.arange(len(data['label'])),'paired')
    xf=make_features(fresh,np.arange(len(fresh['label'])),'paired');rows=[]
    for entry in d['models']:
        key=entry['initialization']+'-'+str(entry['spec']['seed'])
        if (root/(key+'.json')).exists():rows.append(json.loads((root/(key+'.json')).read_text()));continue
        model=WindowActorCritic('real8','restricted');model.load_state_dict(torch.load(entry['path'],map_location='cpu',weights_only=True));model.eval()
        target=np.flatnonzero(((model.pre<2)&(model.post>=2)&(model.post<4)).numpy()).tolist()
        other=np.flatnonzero(((model.pre!=model.post)&~((model.pre<2)&(model.post>=2)&(model.post<4))).numpy()).tolist()
        weight=model.chemical.detach().abs().numpy();target_norm=float(weight[target].sum())
        intact,score=evaluate(model,xf,fresh['label'],fresh['run'],fresh['noise'])
        row=dict(model=entry,intact=intact,intact_ridge=ridge_transfer(model,x,data,xf,fresh,d),target_indices=target,targets=[])
        for fraction in d['fractions']:
            lesion,removed=altered_model(model,target,fraction)
            result,score=evaluate(lesion,xf,fresh['label'],fresh['run'],fresh['noise'])
            probe=ridge_transfer(lesion,x,data,xf,fresh,d)
            candidates=[];budget=target_norm*fraction
            for indices in itertools.combinations(other,len(target)):
                norm=float(weight[list(indices)].sum())
                if norm>=budget and norm>1e-12:candidates.append((norm,indices))
            candidates.sort(key=lambda item:(item[0]-budget,item[1]));controls=[]
            for norm,indices in candidates[:20]:
                ablated,actual=altered_model(model,list(indices),budget/norm)
                assert abs(actual-budget)<=max(1e-6,budget*1e-5)
                performance,pred=evaluate(ablated,xf,fresh['label'],fresh['run'],fresh['noise'])
                controls.append(dict(indices=indices,fraction=budget/norm,removed_l1=actual,new=performance))
            row['targets'].append(dict(fraction=fraction,removed_l1=removed,new=result,ridge=probe,
                eligible_control_pairs=len(candidates),controls=controls))
        write(root/(key+'.json'),row);rows.append(row);print('Completed matched path probe',key,flush=True)
    write(root/'summary.json',dict(rows=rows,complete=True));check(d)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run'])
    p.add_argument('--root',type=Path,default=Path('results/body-thermal-path-probe-2026-11-03'))
    a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='prepare':prepare(a.root)
    else:
        d=json.loads((a.root/'design.json').read_text());check(d);run(a.root,d)
