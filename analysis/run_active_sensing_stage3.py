"""Grouped decoding of warmer-side labels from actual body-generated samples.

No environment truth enters features. All variants of a physical pulse phase
stay in one split; windows contain only observations after the pulse begins.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from diagnose_navigation_failures import digest,representation,trace_actor,write
from thermotaxis.neuromechanics.boyle_output import curvature
from thermotaxis.navigation_window_policy import WindowActorCritic


REPO=Path(__file__).resolve().parents[1]
BODY=Path('results/body-load-stage2-2026-11-03')
NEURAL=Path('results/neural-pathway-stage1-2026-11-03')
CONDITIONS=('temperature','paired','mispaired','center')


def group_split(phase):
    return 0 if phase in (0,2,4,6) else (1 if phase in (1,5) else 2)


def prepare(root):
    if root.exists():raise FileExistsError('preserve previous analysis')
    body=json.loads((BODY/'conditions.json').read_text());assert body['complete']
    selected=[r for r in body['rows'] if r['factor']>1]
    assert len(selected)==32
    sources=['analysis/run_active_sensing_stage3.py','tests/test_active_sensing_stage3.py',
        'analysis/diagnose_navigation_failures.py','src/thermotaxis/navigation_window_policy.py',
        'src/thermotaxis/neuromechanics/boyle_output.py']
    inputs=[BODY/'conditions.json',BODY/'nominal-live/trajectory.npz',NEURAL/'design.json']
    inputs += [BODY/r['path']/'trajectory.npz' for r in selected]
    original=json.loads((NEURAL/'design.json').read_text())
    inputs += [Path('results/navigation-factorial-2026-10-25')/f'fit-{s["id"]:03d}'/'BC.pt' for s in original['frozen_control_specs']]
    inputs += [NEURAL/f'fit-{s["id"]:03d}-All/model.pt' for s in original['specs']]
    d=dict(status='declared_before_decoding',source_runs=selected,conditions=list(CONDITIONS),
        split=dict(train_phases=[0,2,4,6],validation_phases=[1,5],test_phases=[3,7]),
        sample_dt_s=.1,history_samples=41,gradient_K_m=40.,gradient_angles_rad=(.125*math.pi+np.arange(8)*math.pi/4).tolist(),
        noise_conditions=[dict(std_K=0.,seed=0)]+[dict(std_K=.001,seed=s) for s in (19,29,39)],
        minimum_transverse_gradient_fraction=.1,background_range_K=[-.04,.04],
        motion_normalization='standard deviations from the nominal trajectory after 3s; causal backward curvature difference',
        mismatch='past 41-sample motion history circularly shifted by a random nonzero offset shared across all field/noise versions of each body window',
        mismatch_scope='offline alignment intervention on an available past window, not a physically continuous corrupted sensor stream',
        generic=dict(name='gru16',seeds=[7,17,27],updates=1024,batch=256,lr=.003,loss='binary cross entropy on warmer-side labels'),
        ridge_lambdas=[.0001,.01,1.,100.],positive_control_threshold=.65,
        neuronal_probe='only if paired GRU mean test balanced accuracy >=.65; frozen BC78 and All6, identical ridge protocol',
        labels='sign of true gradient projected on the in-plane normal to head-tail axis; evaluator only',
        limits=['Existing body runs and deterministic motor parameters; exploratory split over pulse phases, not animal replication.',
            'No policy learns to navigate; this is supervised information availability/representation.',
            'Neural probes use a declared engineering projection of curvature into old phase slots; no corollary-discharge claim.'],
        source_sha256={name:digest(REPO/name) for name in sources},
        inputs_sha256={str(name):digest(REPO/name) for name in inputs})
    write(root/'design.json',d)


def check(d):
    for name,h in {**d['source_sha256'],**d['inputs_sha256']}.items():
        if digest(REPO/name)!=h:raise RuntimeError('source/input changed: '+name)


def causal_motion(pose,dt):
    k=curvature(pose)[:,:5].mean(1)
    return np.stack([k,np.r_[0.,np.diff(k)/dt]],axis=1)


def circular_mismatch(motion,shifts):
    index=(np.arange(41)[None,:]-np.asarray(shifts)[:,None])%41
    return np.take_along_axis(motion,index[:,:,None],axis=1)


def make_features(data,indices,condition):
    if condition not in CONDITIONS:raise ValueError(condition)
    temp=data['center_temperature' if condition=='center' else 'head_temperature'][indices]
    mean=temp.mean(axis=1,keepdims=True)
    motion=data['motion'][indices]
    if condition=='temperature':motion=np.zeros_like(motion)
    elif condition=='mispaired':motion=circular_mismatch(motion,data['shift'][indices])
    x=np.zeros((len(indices),41,7),dtype='float32')
    x[:,:,0]=(temp-mean)/.05;x[:,:,1]=mean/.05
    x[:,:,4:6]=motion;x[:,:,6]=1
    return x


def dataset(root,d):
    if (root/'dataset.npz').exists():raise FileExistsError('preserve existing dataset')
    with np.load(BODY/'nominal-live/trajectory.npz') as z:
        t=z['time_s'][::5];motion=causal_motion(z['pose'][::5],.1)
    scale=motion[t>=3].std(axis=0);assert np.all(scale>0)
    head_all=[];center_all=[];motion_all=[];labels=[];runs=[];phases=[];splits=[];noises=[];shifts=[]
    metadata=[]
    for run_id,row in enumerate(d['source_runs']):
        with np.load(BODY/row['path']/'trajectory.npz') as z:
            t=z['time_s'][::5];pose=z['pose'][::5]
        body_motion=(causal_motion(pose,.1)/scale).astype('float32')
        head=pose[:,0,:2];center=pose[:,:,:2].mean(1)
        axis=pose[:,0,:2]-pose[:,-1,:2];axis/=np.linalg.norm(axis,axis=1,keepdims=True)
        normal=np.stack([-axis[:,1],axis[:,0]],axis=1)
        shift=np.random.default_rng(310803+run_id).integers(1,41,len(t))
        retained=0
        for angle_id,angle in enumerate(d['gradient_angles_rad']):
            gradient=d['gradient_K_m']*np.array([np.cos(angle),np.sin(angle)])
            transverse=normal@gradient
            eligible=np.flatnonzero((np.arange(len(t))>=40)&(t>=row['onset_s']+4-1e-9)&
                (abs(transverse)>=d['gradient_K_m']*d['minimum_transverse_gradient_fraction']))
            history=eligible[:,None]+np.arange(-40,1)[None,:]
            for noise_id,noise in enumerate(d['noise_conditions']):
                rng=np.random.default_rng(np.random.SeedSequence([330811,run_id,angle_id,noise['seed']]))
                background=rng.uniform(*d['background_range_K'])
                eta=rng.normal(0,noise['std_K'],len(t))
                # Quantize at the same float32 Kelvin sensor boundary as old policies.
                ht=(293.15+background+head@gradient+eta).astype('float32')-np.float32(293.15)
                ct=(293.15+background+center@gradient+eta).astype('float32')-np.float32(293.15)
                n=len(eligible);retained+=n
                head_all.append(ht[history]);center_all.append(ct[history]);motion_all.append(body_motion[history])
                labels.append((transverse[eligible]>0).astype('float32'));runs.append(np.full(n,run_id,dtype='int16'))
                phases.append(np.full(n,row['phase'],dtype='int8'));splits.append(np.full(n,group_split(row['phase']),dtype='int8'))
                noises.append(np.full(n,noise['std_K'],dtype='float32'));shifts.append(shift[eligible].astype('int8'))
        metadata.append(dict(run_id=run_id,**row,windows=retained))
    values=dict(head_temperature=np.concatenate(head_all),center_temperature=np.concatenate(center_all),
        motion=np.concatenate(motion_all),label=np.concatenate(labels),run=np.concatenate(runs),phase=np.concatenate(phases),
        split=np.concatenate(splits),noise=np.concatenate(noises),shift=np.concatenate(shifts))
    # No body phase, time-overlapping run or virtual-field version crosses a split.
    assert not(set(values['run'][values['split']==0])&set(values['run'][values['split']==2]))
    sample=np.arange(min(2048,len(values['label'])))
    original=values['motion'][sample];changed=circular_mismatch(original,values['shift'][sample])
    rms_error=float(np.max(abs(np.mean(original.astype('float64')**2,axis=1)-np.mean(changed.astype('float64')**2,axis=1))))
    spectrum_error=float(np.max(abs(abs(np.fft.rfft(original.astype('float64'),axis=1))-abs(np.fft.rfft(changed.astype('float64'),axis=1)))))
    np.savez_compressed(root/'dataset.npz',**values)
    write(root/'dataset.json',dict(runs=metadata,motion_scale=scale.tolist(),windows=len(values['label']),
        split_counts={str(k):int(np.sum(values['split']==k)) for k in (0,1,2)},
        split_runs={str(k):sorted(map(int,np.unique(values['run'][values['split']==k]))) for k in (0,1,2)},
        mismatch_max_mean_square_error=rms_error,mismatch_max_spectrum_error=spectrum_error,
        dataset_sha256=digest(root/'dataset.npz'),complete=True))
    print('Dataset prepared',len(values['label']),'windows; physical-run grouped split',flush=True)


def metric(y,score,run):
    pred=score>=0
    values=[]
    for r in np.unique(run):
        ids=run==r
        pos=ids&(y>0);neg=ids&(y<=0)
        if not pos.any() or not neg.any():raise ValueError('unbalanced physical-run label set')
        values.append(dict(run_id=int(r),balanced_accuracy=float(.5*(np.mean(pred[pos])+np.mean(~pred[neg])))))
    return dict(balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in values])),by_run=values)


def ridge_probe(features,data,d):
    train=data['split']==0;val=data['split']==1;test=data['split']==2
    mean=features[train].mean(0);scale=features[train].std(0);scale=np.maximum(scale,1e-8)
    x=np.column_stack([np.ones(len(features)),(features-mean)/scale]);target=data['label']*2-1
    xx=x[train].T@x[train];xy=x[train].T@target[train]
    candidates=[]
    for penalty in d['ridge_lambdas']:
        reg=np.eye(x.shape[1])*penalty;reg[0,0]=0
        weight=np.linalg.solve(xx+reg,xy)
        result=metric(data['label'][val],x[val]@weight,data['run'][val])
        candidates.append((result['balanced_accuracy'],penalty,weight))
    _,penalty,weight=max(candidates,key=lambda r:r[0])
    return dict(penalty=penalty,validation=metric(data['label'][val],x[val]@weight,data['run'][val]),
        test=metric(data['label'][test],x[test]@weight,data['run'][test]),
        test_noisy=metric(data['label'][test&(data['noise']>0)],x[test&(data['noise']>0)]@weight,data['run'][test&(data['noise']>0)]),
        test_noiseless=metric(data['label'][test&(data['noise']==0)],x[test&(data['noise']==0)]@weight,data['run'][test&(data['noise']==0)]))


def regression_features(x):
    # Local temperature regression on a past-window body reference; no labels used.
    n=len(x);clock=np.broadcast_to(np.linspace(-1,0,41),(n,41))
    matrix=np.stack([np.ones((n,41)),clock,x[:,:,4],x[:,:,5]],axis=2).astype('float64')
    temp=(x[:,:,0]+x[:,:,1]).astype('float64')
    a=np.einsum('nti,ntj->nij',matrix,matrix)+np.eye(4)[None]*1e-6
    b=np.einsum('nti,nt->ni',matrix,temp)
    return np.linalg.solve(a,b[:,:,None])[:,:,0]


def fit(root,d):
    with np.load(root/'dataset.npz') as z:data={k:z[k] for k in z.files}
    assert digest(root/'dataset.npz')==json.loads((root/'dataset.json').read_text())['dataset_sha256']
    train_ids=np.flatnonzero(data['split']==0);test_ids=np.flatnonzero(data['split']==2)
    local=[]
    for condition in CONDITIONS:
        all_x=make_features(data,np.arange(len(data['label'])),condition)
        features=np.concatenate([regression_features(all_x[i:i+1024]) for i in range(0,len(all_x),1024)])
        local.append(dict(condition=condition,**ridge_probe(features,data,d)))
        for seed in d['generic']['seeds']:
            dest=root/f'gru-{condition}-{seed}.json'
            if dest.exists():continue
            started=time.monotonic();torch.manual_seed(seed)
            model=WindowActorCritic('gru16','full')
            for name,p in model.named_parameters():p.requires_grad_(name.startswith(('memory.','actor.')))
            optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=d['generic']['lr'])
            rng=np.random.default_rng(340001+seed);curve=[]
            for update in range(d['generic']['updates']):
                idx=rng.choice(train_ids,d['generic']['batch'],replace=True)
                logit=model.actor_mean(torch.from_numpy(all_x[idx]))
                loss=F.binary_cross_entropy_with_logits(logit,torch.from_numpy(data['label'][idx]))
                if not torch.isfinite(loss):raise RuntimeError('nonfinite classification loss')
                optimizer.zero_grad();loss.backward();optimizer.step()
                if update%128==0:curve.append(dict(update=update+1,loss=float(loss.detach())))
            model.eval();scores=[]
            with torch.no_grad():
                for start in range(0,len(test_ids),1024):
                    scores.append(model.actor_mean(torch.from_numpy(all_x[test_ids[start:start+1024]])).numpy())
            score=np.concatenate(scores);y=data['label'][test_ids];r=data['run'][test_ids];noise=data['noise'][test_ids]
            torch.save(model.state_dict(),root/f'gru-{condition}-{seed}.pt')
            np.savez_compressed(root/f'gru-{condition}-{seed}-predictions.npz',indices=test_ids,score=score)
            write(dest,dict(condition=condition,seed=seed,curve=curve,wall_s=time.monotonic()-started,
                test=metric(y,score,r),test_noisy=metric(y[noise>0],score[noise>0],r[noise>0]),
                test_noiseless=metric(y[noise==0],score[noise==0],r[noise==0]),complete=True))
            print('Completed generic decoder',condition,seed,flush=True)
    write(root/'local-regression.json',dict(rows=local,complete=True));check(d)


def neurons(root,d):
    controls=[json.loads((root/f'gru-paired-{seed}.json').read_text()) for seed in d['generic']['seeds']]
    score=float(np.mean([r['test']['balanced_accuracy'] for r in controls]))
    if score<d['positive_control_threshold']:
        write(root/'neuronal-probes.json',dict(skipped=True,reason='positive control below declared threshold',positive_control=score));return
    parent=json.loads((NEURAL/'design.json').read_text())
    with np.load(root/'dataset.npz') as z:data={k:z[k] for k in z.files}
    x=make_features(data,np.arange(len(data['label'])),'paired')
    sources=[(s,'BC',Path('results/navigation-factorial-2026-10-25')/f'fit-{s["id"]:03d}'/'BC.pt') for s in parent['frozen_control_specs']]
    sources += [(s,'All',NEURAL/f'fit-{s["id"]:03d}-All/model.pt') for s in parent['specs']]
    rows=[]
    for s,label,path in sources:
        torch.manual_seed(s['seed']);model=WindowActorCritic(s['name'],s['interface'])
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad():
            if model.is_gru:
                h=torch.cat([representation(model,torch.from_numpy(x[i:i+1024])) for i in range(0,len(x),1024)]).numpy()
                layers={'readout':h}
            else:
                h=torch.cat([trace_actor(model,torch.from_numpy(x[i:i+1024]))[1][:,-1] for i in range(0,len(x),1024)]).numpy()
                layers={'AFD':h[:,:2],'AIY':h[:,2:4],'RIA':h[:,6:8],'all8':h}
        rows.append(dict(spec=s,checkpoint=label,layers={name:ridge_probe(value,data,d) for name,value in layers.items()}))
        print('Completed frozen neural decoder',s['id'],label,flush=True)
    write(root/'neuronal-probes.json',dict(rows=rows,positive_control=score,skipped=False,complete=True));check(d)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','dataset','fit','neurons'])
    p.add_argument('--root',type=Path,default=Path('results/active-sensing-stage3-2026-11-03'))
    a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='prepare':prepare(a.root);return
    d=json.loads((a.root/'design.json').read_text());check(d)
    {'dataset':dataset,'fit':fit,'neurons':neurons}[a.stage](a.root,d)


if __name__=='__main__':main()
