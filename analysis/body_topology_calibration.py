"""Restricted-topology functional calibration, BC/random starts, sealed new body cases."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from calibrate_body_input_small_step import evaluate
from diagnose_navigation_failures import digest,write
from run_active_sensing_stage3 import BODY,NEURAL,causal_motion,make_features,metric
from run_pathway_stage1 import configure_trainable
from thermotaxis.navigation_window_policy import WindowActorCritic

REPO=Path(__file__).resolve().parents[1]
DATA=Path('results/active-sensing-stage3-2026-11-03')
PARENT=Path('results/navigation-factorial-2026-10-25')


def select_parameters(model):
    if not model.is_gru:return configure_trainable(model,'All')
    selected=[]
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(('memory.','actor.')))
        if p.requires_grad:selected.append(name)
    return selected


def reachable(model,source,target):
    a=np.zeros((8,8),bool)
    a[model.pre.numpy(),model.post.numpy()]=True
    a[model.gap_i.numpy(),model.gap_j.numpy()]=True;a[model.gap_j.numpy(),model.gap_i.numpy()]=True
    if model.name=='isolated8':a&=np.eye(8,dtype=bool)
    seen=set(source)
    while True:
        new=seen|{int(j) for i in seen for j in np.flatnonzero(a[i])}
        if new==seen:break
        seen=new
    return dict(zip(map(str,target),[int(t) in seen for t in target]))


def prepare(root):
    if root.exists():raise FileExistsError('preserve completed or partial experiment')
    parent=json.loads((NEURAL/'design.json').read_text())
    specs=[s for s in parent['frozen_control_specs'] if s['interface']=='restricted' or s['name'].startswith('gru')]
    assert len(specs)==42
    clock=json.loads((BODY/'baseline-clock.json').read_text())
    cases=[]
    for phase in [3.5,7.5]:
        onset=round((clock['reference_upward_crossing_s']+phase*clock['period_s']/8)/.001)*.001
        for factor in [.75,1.5,2.,3.]:
            for feedback in ['live','replay']:
                cases.append(dict(phase=phase,onset_s=onset,factor=factor,feedback=feedback,
                    family='matched_load' if factor in (1.5,2.) else 'novel_load',
                    path=f'physical/phase-{phase:g}-factor-{factor:g}-{feedback}'))
    sources=['analysis/body_topology_calibration.py','tests/test_body_topology_calibration.py',
        'analysis/run_active_sensing_stage3.py','analysis/run_pathway_stage1.py',
        'analysis/neuromechanics/build_boyle_load_worker.py','analysis/neuromechanics/run_load_feedback_stage2.py',
        'src/thermotaxis/navigation_window_policy.py','src/thermotaxis/connectome.py']
    inputs=[DATA/'dataset.npz',DATA/'dataset.json',BODY/'design.json',BODY/'reference-1ms.bin',
        Path('data/connectome/cook2019_herm.json')]
    inputs += [PARENT/f'fit-{s["id"]:03d}'/'BC.pt' for s in specs]
    topology=[]
    for s in specs:
        if s['seed']!=7 or s['name'].startswith('gru'):continue
        torch.manual_seed(7);m=WindowActorCritic(s['name'],s['interface'])
        topology.append(dict(name=s['name'],thermal_to_RIA=reachable(m,[0,1],[6,7]),
            motor_to_RIA=reachable(m,[2,3],[6,7]),chemical_edges=len(m.pre)))
    d=dict(status='declared_before_all_topology_fits_and_new_body_cases',specs=specs,
        initializations=['BC','random'],updates=1024,batch=256,lr=.003,
        minibatch_seed='550300 + training seed; shared across initialization and graph',
        training='original phase groups 0,2,4,6; fixed budget, final endpoint only',
        old_test='previously inspected phase3,7; diagnostic only',new_body_cases=cases,
        primary_new_family='matched_load: same load range, new pulse phases/fields/noise',
        secondary_new_family='novel_load: unseen anterior drag factors .75 and3, reported separately',
        new_scene=dict(angles_rad=(np.pi/16+np.arange(8)*np.pi/4).tolist(),noise_seeds=[0,149,159,169],
            gradient_K_m=40.,background_K=[-.04,.04],noise_std_K=.001,rng_prefix=550817),
        topology=topology,notes=['All ten rewired graphs retained, including structural disconnections and duplicate layouts.',
            'GRU2/16 are capacity references, not parameter-matched anatomical models.',
            'BC and random starts reported separately; BC has earlier equal-budget kinematic pretraining.',
            'New cases vary pulse phase and scene, but share one deterministic body parameter set.',
            'Supervised direction decoding; no closed-loop thermotaxis or measured cellular physiology.'],
        source_sha256={p:digest(REPO/p) for p in sources},inputs_sha256={str(p):digest(REPO/p) for p in inputs},
        versions=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform()))
    write(root/'design.json',d)


def check(d):
    for name,h in {**d['source_sha256'],**d['inputs_sha256']}.items():
        if digest(REPO/name)!=h:raise RuntimeError('changed source/input: '+name)


def physical(root,d):
    import sys
    sys.path.insert(0,str(REPO/'analysis/neuromechanics'))
    from run_load_feedback_stage2 import simulate
    body_design=json.loads((BODY/'design.json').read_text())
    for c in d['new_body_cases']:
        simulate(root,c['path'],body_design,factor=c['factor'],onset=c['onset_s'],
                 replay=BODY/'reference-1ms.bin' if c['feedback']=='replay' else None)
    meta=json.loads((DATA/'dataset.json').read_text());scale=np.array(meta['motion_scale'])
    temp=[];motion=[];target=[];runs=[];noises=[];ends=[];families=[]
    for rid,c in enumerate(d['new_body_cases']):
        with np.load(root/c['path']/'trajectory.npz') as z:t=z['time_s'][::5];pose=z['pose'][::5]
        body_motion=(causal_motion(pose,.1)/scale).astype('float32')
        axis=pose[:,0,:2]-pose[:,-1,:2];axis/=np.linalg.norm(axis,axis=1,keepdims=True)
        normal=np.stack([-axis[:,1],axis[:,0]],axis=1)
        for j,angle in enumerate(d['new_scene']['angles_rad']):
            g=40*np.array([np.cos(angle),np.sin(angle)]);side=normal@g
            ids=np.flatnonzero((np.arange(len(t))>=40)&(t>=c['onset_s']+4-1e-9)&(abs(side)>=4))
            history=ids[:,None]+np.arange(-40,1)[None,:]
            for seed in d['new_scene']['noise_seeds']:
                std=0. if seed==0 else .001
                rng=np.random.default_rng(np.random.SeedSequence([550817,rid,j,seed]))
                background=rng.uniform(-.04,.04);eta=rng.normal(0,std,len(t))
                value=(293.15+background+pose[:,0,:2]@g+eta).astype('float32')-np.float32(293.15)
                temp.append(value[history]);motion.append(body_motion[history]);target.append((side[ids]>0).astype('float32'))
                runs.append(np.full(len(ids),rid,dtype='int16'));noises.append(np.full(len(ids),std,dtype='float32'));ends.append(ids)
                families.append(np.full(len(ids),0 if c['family']=='matched_load' else 1,dtype='int8'))
    values=dict(head_temperature=np.concatenate(temp),motion=np.concatenate(motion),label=np.concatenate(target),
        run=np.concatenate(runs),noise=np.concatenate(noises),window_end=np.concatenate(ends),family=np.concatenate(families))
    np.savez_compressed(root/'new-body-scenes.npz',**values)
    write(root/'new-body-scenes.json',dict(windows=len(values['label']),runs=len(d['new_body_cases']),sha256=digest(root/'new-body-scenes.npz'),
        physical_run_sha256={c['path']:digest(root/c['path']/'trajectory.npz') for c in d['new_body_cases']},complete=True))
    check(d)


def train(root,d):
    with np.load(DATA/'dataset.npz') as z:data={k:z[k] for k in z.files}
    x=make_features(data,np.arange(len(data['label'])),'paired');y=torch.from_numpy(data['label'])
    ids_train=np.flatnonzero(data['split']==0);ids_old=np.flatnonzero(data['split']==2)
    for init in d['initializations']:
        for s in d['specs']:
            folder=root/'fits'/f'{init}-{s["id"]:03d}'
            if (folder/'record.json').exists():
                old=json.loads((folder/'record.json').read_text())
                if digest(folder/'model.pt')!=old['model_sha256']:raise RuntimeError('changed completed model')
                continue
            folder.mkdir(parents=True,exist_ok=True);started=time.monotonic();torch.manual_seed(s['seed'])
            model=WindowActorCritic(s['name'],s['interface'])
            if init=='BC':model.load_state_dict(torch.load(PARENT/f'fit-{s["id"]:03d}'/'BC.pt',map_location='cpu',weights_only=True))
            model.eval();before={k:v.clone() for k,v in model.state_dict().items()}
            selected=select_parameters(model)
            optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=d['lr'])
            rng=np.random.default_rng(550300+s['seed']);curve=[]
            for update in range(d['updates']):
                ids=rng.choice(ids_train,d['batch'],replace=True)
                loss=F.binary_cross_entropy_with_logits(model.actor_mean(torch.from_numpy(x[ids])),y[ids])
                if not torch.isfinite(loss):raise RuntimeError('nonfinite fit')
                optimizer.zero_grad();loss.backward();optimizer.step()
                if update%128==0:curve.append(dict(update=update+1,loss=float(loss.detach())))
            changed=[k for k,v in before.items() if not torch.equal(v,model.state_dict()[k])]
            assert set(changed)<=set(selected)
            old,score=evaluate(model,x[ids_old],data['label'][ids_old],data['run'][ids_old],data['noise'][ids_old])
            torch.save(model.state_dict(),folder/'model.pt');np.savez_compressed(folder/'old-predictions.npz',score=score)
            write(folder/'record.json',dict(spec=s,initialization=init,old=old,curve=curve,changed_names=changed,
                trainable_names=selected,model_sha256=digest(folder/'model.pt'),wall_s=time.monotonic()-started,complete=True))
            print('Completed topology fit',init,s['name'],s['seed'],flush=True)
    check(d)


def evaluate_new(root,d):
    meta=json.loads((root/'new-body-scenes.json').read_text())
    if digest(root/'new-body-scenes.npz')!=meta['sha256']:raise RuntimeError('changed fresh data')
    with np.load(root/'new-body-scenes.npz') as z:data={k:z[k] for k in z.files}
    x=make_features(data,np.arange(len(data['label'])),'paired');rows=[]
    for init in d['initializations']:
        for s in d['specs']:
            folder=root/'fits'/f'{init}-{s["id"]:03d}';r=json.loads((folder/'record.json').read_text())
            model=WindowActorCritic(s['name'],s['interface']);model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True));model.eval()
            result,score=evaluate(model,x,data['label'],data['run'],data['noise'])
            for name,code in [('matched_load',0),('novel_load',1)]:
                ids=data['family']==code
                result[name]=metric(data['label'][ids],score[ids],data['run'][ids])
            np.savez_compressed(folder/'new-predictions.npz',score=score)
            rows.append(dict(spec=s,initialization=init,old=r['old'],new=result))
    write(root/'evaluation.json',dict(rows=rows,complete=True));check(d)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','physical','train','evaluate'])
    p.add_argument('--root',type=Path,default=Path('results/body-topology-calibration-2026-11-03'))
    a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='prepare':prepare(a.root)
    else:
        d=json.loads((a.root/'design.json').read_text());check(d)
        {'physical':physical,'train':train,'evaluate':evaluate_new}[a.stage](a.root,d)
