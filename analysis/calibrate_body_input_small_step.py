"""Six fixed-budget fits: readout vs input+readout on the existing body task."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from diagnose_navigation_failures import digest,representation,write
from run_active_sensing_stage3 import BODY,NEURAL,causal_motion,make_features,metric
from run_pathway_stage1 import configure_trainable
from thermotaxis.navigation_window_policy import WindowActorCritic

REPO=Path(__file__).resolve().parents[1]
PARENT=Path('results/active-sensing-stage3-2026-11-03')


def prepare(root):
    if root.exists():raise FileExistsError('preserve prior output')
    parent=json.loads((NEURAL/'design.json').read_text())
    specs=[s for s in parent['specs'] if s['interface']=='restricted'];assert len(specs)==3
    meta=json.loads((PARENT/'dataset.json').read_text())
    runs=[r for r in meta['runs'] if r['phase'] in (3,7)];assert len(runs)==8
    sources=['analysis/calibrate_body_input_small_step.py','analysis/run_active_sensing_stage3.py',
        'analysis/run_pathway_stage1.py','src/thermotaxis/navigation_window_policy.py']
    inputs=[PARENT/'dataset.npz',PARENT/'dataset.json']
    inputs += [NEURAL/f'fit-{s["id"]:03d}-All/model.pt' for s in specs]
    inputs += [PARENT/f'gru-paired-{s}.pt' for s in (7,17,27)]
    inputs += [BODY/r['path']/'trajectory.npz' for r in runs]
    d=dict(status='declared_before_six_calibration_fits',specs=specs,arms=['R','IR'],updates=1024,batch=256,lr=.003,
        start='stage1 restricted-All; independent restart for each arm',train='stage3 phase groups0,2,4,6 only',
        old_test='phase3,7 already inspected; paired diagnostic only',
        fresh_scene=dict(runs=runs,angles_rad=(np.arange(8)*np.pi/4).tolist(),
            noise_seeds=[0,109,119,129],noise_std_K=.001,gradient_K_m=40.,rng_prefix=440817),
        scope='new field directions/noise on reused held-out body trajectories, not independent new physical runs',
        frozen='all chemical/gap/bias/dynamics, graph buffers, critic and log_std',
        source_sha256={s:digest(REPO/s) for s in sources},inputs_sha256={str(s):digest(REPO/s) for s in inputs})
    write(root/'design.json',d)


def check(d):
    for name,h in {**d['source_sha256'],**d['inputs_sha256']}.items():
        if digest(REPO/name)!=h:raise RuntimeError('changed source/input: '+name)


def fresh_data(root,d):
    meta=json.loads((PARENT/'dataset.json').read_text());scale=np.array(meta['motion_scale'])
    heads=[];motions=[];labels=[];runs=[];noises=[]
    for row in d['fresh_scene']['runs']:
        rid=row['run_id']
        with np.load(BODY/row['path']/'trajectory.npz') as z:t=z['time_s'][::5];pose=z['pose'][::5]
        motion=(causal_motion(pose,.1)/scale).astype('float32')
        axis=pose[:,0,:2]-pose[:,-1,:2];axis/=np.linalg.norm(axis,axis=1,keepdims=True)
        normal=np.stack([-axis[:,1],axis[:,0]],axis=1)
        for a,angle in enumerate(d['fresh_scene']['angles_rad']):
            g=40*np.array([np.cos(angle),np.sin(angle)]);side=normal@g
            ids=np.flatnonzero((np.arange(len(t))>=40)&(t>=row['onset_s']+4-1e-9)&(abs(side)>=4))
            history=ids[:,None]+np.arange(-40,1)[None,:]
            for seed in d['fresh_scene']['noise_seeds']:
                std=0. if seed==0 else .001
                rng=np.random.default_rng(np.random.SeedSequence([440817,rid,a,seed]))
                background=rng.uniform(-.04,.04);eta=rng.normal(0,std,len(t))
                temp=(293.15+background+pose[:,0,:2]@g+eta).astype('float32')-np.float32(293.15)
                heads.append(temp[history]);motions.append(motion[history]);labels.append((side[ids]>0).astype('float32'))
                runs.append(np.full(len(ids),rid,dtype='int16'));noises.append(np.full(len(ids),std,dtype='float32'))
    data=dict(head_temperature=np.concatenate(heads),motion=np.concatenate(motions),label=np.concatenate(labels),
              run=np.concatenate(runs),noise=np.concatenate(noises))
    np.savez_compressed(root/'fresh-scenes.npz',**data)
    write(root/'fresh-scenes.json',dict(windows=len(data['label']),physical_runs=len(np.unique(data['run'])),
        sha256=digest(root/'fresh-scenes.npz'),note=d['scope']))
    return data


def evaluate(model,x,y,r,noise):
    scores=[]
    with torch.no_grad():
        for i in range(0,len(x),1024):scores.append(model.actor_mean(torch.from_numpy(x[i:i+1024])).numpy())
    score=np.concatenate(scores)
    return dict(all=metric(y,score,r),noisy=metric(y[noise>0],score[noise>0],r[noise>0])),score


def run(root,d):
    with np.load(PARENT/'dataset.npz') as z:data={k:z[k] for k in z.files}
    fresh=fresh_data(root,d)
    x=make_features(data,np.arange(len(data['label'])),'paired');xf=make_features(fresh,np.arange(len(fresh['label'])),'paired')
    train=np.flatnonzero(data['split']==0);test=np.flatnonzero(data['split']==2)
    y=torch.from_numpy(data['label']);rows=[]
    for spec in d['specs']:
        path=NEURAL/f'fit-{spec["id"]:03d}-All/model.pt'
        for arm in d['arms']:
            started=time.monotonic();torch.manual_seed(spec['seed'])
            model=WindowActorCritic('real8','restricted')
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
            before={k:v.clone() for k,v in model.state_dict().items()};selected=configure_trainable(model,arm)
            features=None
            if arm=='R':
                with torch.no_grad():features=torch.cat([representation(model,torch.from_numpy(x[i:i+1024])) for i in range(0,len(x),1024)])
            opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=d['lr'])
            rng=np.random.default_rng(440300+spec['seed']);curve=[]
            for update in range(d['updates']):
                ids=rng.choice(train,d['batch'],replace=True)
                logit=model.actor(features[ids]).squeeze(-1) if features is not None else model.actor_mean(torch.from_numpy(x[ids]))
                loss=F.binary_cross_entropy_with_logits(logit,y[ids])
                if not torch.isfinite(loss):raise RuntimeError('nonfinite calibration')
                opt.zero_grad();loss.backward();opt.step()
                if update%128==0:curve.append(dict(update=update+1,loss=float(loss.detach())))
            changed=[k for k,v in before.items() if not torch.equal(v,model.state_dict()[k])]
            assert set(changed)<=set(selected)
            old,so=evaluate(model,x[test],data['label'][test],data['run'][test],data['noise'][test])
            new,sn=evaluate(model,xf,fresh['label'],fresh['run'],fresh['noise'])
            name=f'{arm}-{spec["seed"]}'
            torch.save(model.state_dict(),root/(name+'.pt'))
            np.savez_compressed(root/(name+'-predictions.npz'),old=so,fresh=sn)
            row=dict(arm=arm,seed=spec['seed'],old=old,fresh=new,trainable_names=selected,
                changed_names=changed,curve=curve,wall_s=time.monotonic()-started,complete=True)
            write(root/(name+'.json'),row);rows.append(row)
            print('Completed calibration',arm,spec['seed'],flush=True)
    positive=[]
    for seed in (7,17,27):
        model=WindowActorCritic('gru16','full')
        model.load_state_dict(torch.load(PARENT/f'gru-paired-{seed}.pt',map_location='cpu',weights_only=True));model.eval()
        new,_=evaluate(model,xf,fresh['label'],fresh['run'],fresh['noise'])
        positive.append(dict(seed=seed,fresh=new))
    result=dict(rows=rows,positive_control=positive,new_fits=6,limits=[d['scope'],d['old_test'],
        'Supervised warmer-side decoding; no navigation or biology validation.'],
        means={arm:dict(old=float(np.mean([r['old']['all']['balanced_accuracy'] for r in rows if r['arm']==arm])),
                        fresh=float(np.mean([r['fresh']['all']['balanced_accuracy'] for r in rows if r['arm']==arm]))) for arm in d['arms']},
        positive_fresh=float(np.mean([r['fresh']['all']['balanced_accuracy'] for r in positive])),complete=True)
    write(root/'summary.json',result);check(d);print(json.dumps(result['means'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run'])
    p.add_argument('--root',type=Path,default=Path('results/body-input-calibration-small-2026-11-03'))
    a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='prepare':prepare(a.root)
    else:
        d=json.loads((a.root/'design.json').read_text());check(d);run(a.root,d)
