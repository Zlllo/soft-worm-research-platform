"""Internal actor calibration versus four-times-longer input/readout fitting."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from calibrate_body_input_small_step import evaluate
from diagnose_navigation_failures import digest,write
from run_active_sensing_stage3 import make_features,NEURAL
from run_pathway_stage1 import configure_trainable
from thermotaxis.navigation_window_policy import WindowActorCritic

REPO=Path(__file__).resolve().parents[1]
SMALL=Path('results/body-input-calibration-small-2026-11-03')
DATA=Path('results/active-sensing-stage3-2026-11-03')


def prepare(root):
    if root.exists():raise FileExistsError('preserve previous experiment')
    previous=json.loads((SMALL/'design.json').read_text())
    sources=['analysis/compare_body_calibration_budget.py','analysis/calibrate_body_input_small_step.py',
        'analysis/run_active_sensing_stage3.py','analysis/run_pathway_stage1.py',
        'src/thermotaxis/navigation_window_policy.py']
    inputs=[DATA/'dataset.npz',SMALL/'fresh-scenes.npz',SMALL/'summary.json']
    inputs += [NEURAL/f'fit-{s["id"]:03d}-All/model.pt' for s in previous['specs']]
    inputs += [SMALL/f'IR-{s["seed"]}.pt' for s in previous['specs']]
    d=dict(status='declared_before_new_fits',specs=previous['specs'],
        arms=[dict(name='All1024',scope='All',updates=1024),dict(name='IR4096',scope='IR',updates=4096)],
        batch=256,lr=.003,seed_rule='440300 + model seed, same prefix as prior IR1024',
        training='phase0,2,4,6 only; independent restart from stage1 restricted-All',
        evaluation='paired diagnosis on already-inspected scene sets, not new confirmation',
        scope='same graph, fixed neuron time constants; All additionally frees chemical/gap/bias, not critic/log_std',
        source_sha256={p:digest(REPO/p) for p in sources},inputs_sha256={str(p):digest(REPO/p) for p in inputs})
    write(root/'design.json',d)


def check(d):
    for p,h in {**d['source_sha256'],**d['inputs_sha256']}.items():
        if digest(REPO/p)!=h:raise RuntimeError('source/input changed: '+p)


def run(root,d):
    if (root/'summary.json').exists():raise FileExistsError('experiment already complete')
    with np.load(DATA/'dataset.npz') as z:data={k:z[k] for k in z.files}
    with np.load(SMALL/'fresh-scenes.npz') as z:fresh={k:z[k] for k in z.files}
    x=make_features(data,np.arange(len(data['label'])),'paired')
    xf=make_features(fresh,np.arange(len(fresh['label'])),'paired')
    train=np.flatnonzero(data['split']==0);test=np.flatnonzero(data['split']==2)
    target=torch.from_numpy(data['label']);rows=[]
    for s in d['specs']:
        initial=NEURAL/f'fit-{s["id"]:03d}-All/model.pt'
        for arm in d['arms']:
            key=arm['name']+'-'+str(s['seed']);dest=root/(key+'.json')
            if dest.exists():rows.append(json.loads(dest.read_text()));continue
            started=time.monotonic();torch.manual_seed(s['seed'])
            model=WindowActorCritic('real8','restricted')
            model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True));model.eval()
            before={k:v.clone() for k,v in model.state_dict().items()}
            selected=configure_trainable(model,arm['scope'])
            optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=d['lr'])
            rng=np.random.default_rng(440300+s['seed']);curve=[];prefix=None
            for update in range(arm['updates']):
                ids=rng.choice(train,d['batch'],replace=True)
                logit=model.actor_mean(torch.from_numpy(x[ids]))
                loss=F.binary_cross_entropy_with_logits(logit,target[ids])
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                optimizer.zero_grad();loss.backward();optimizer.step()
                if update%128==0 or update+1==arm['updates']:
                    curve.append(dict(update=update+1,loss=float(loss.detach())))
                if arm['scope']=='IR' and update+1==1024:
                    old=torch.load(SMALL/f'IR-{s["seed"]}.pt',map_location='cpu',weights_only=True)
                    prefix=max(float((v-old[k]).abs().max()) for k,v in model.state_dict().items() if v.dtype.is_floating_point)
                    if prefix>1e-6:raise RuntimeError('IR prefix no longer reproduces the earlier calibration')
            changed=[k for k,v in before.items() if not torch.equal(v,model.state_dict()[k])]
            assert set(changed)<=set(selected)
            old,so=evaluate(model,x[test],data['label'][test],data['run'][test],data['noise'][test])
            new,sn=evaluate(model,xf,fresh['label'],fresh['run'],fresh['noise'])
            torch.save(model.state_dict(),root/(key+'.pt'))
            np.savez_compressed(root/(key+'-predictions.npz'),old=so,fresh=sn)
            row=dict(spec=s,arm=arm,old=old,fresh=new,changed_names=changed,trainable_names=selected,
                ir_1024_prefix_max_difference=prefix,curve=curve,wall_s=time.monotonic()-started,complete=True)
            write(dest,row);rows.append(row);print('Completed',key,flush=True)
    result=dict(rows=rows,new_fits=6,baseline=json.loads((SMALL/'summary.json').read_text()),
        means={a['name']:dict(old=float(np.mean([r['old']['all']['balanced_accuracy'] for r in rows if r['arm']['name']==a['name']])),
                            fresh=float(np.mean([r['fresh']['all']['balanced_accuracy'] for r in rows if r['arm']['name']==a['name']]))) for a in d['arms']},complete=True)
    write(root/'summary.json',result);check(d);print(json.dumps(result['means'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run'])
    p.add_argument('--root',type=Path,default=Path('results/body-calibration-budget-2026-11-03'))
    a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='prepare':prepare(a.root)
    else:
        d=json.loads((a.root/'design.json').read_text());check(d);run(a.root,d)
