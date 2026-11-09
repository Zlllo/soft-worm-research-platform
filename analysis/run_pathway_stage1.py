"""Declared equal-budget layer interventions; no PPO or new topology fitting.

Run prepare once, then train, evaluate and probe from the repository root.
Every fitting arm starts from the same BC state and uses identical minibatches.
"""
from dataclasses import asdict, replace
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from diagnose_navigation_failures import digest, representation, stimuli, temperature_perturbation, trace_actor, write
from navigation_feasibility_replication import evaluate
from navigation_preflight import summary
from thermotaxis.connectome import CORE
from thermotaxis.navigation import NavigationBatch, NavigationConfig
from thermotaxis.navigation_window_policy import NavigationFeatures, WindowActorCritic


REPO = Path(__file__).resolve().parents[1]
PARENT = Path('results/navigation-factorial-2026-10-25')
TEACHER = Path('results/navigation-feature-teacher-raw-2026-10-25.npz')
ARMS = ('frozen', 'R', 'IR', 'All')


def configure_trainable(model, arm):
    if arm not in ARMS:
        raise ValueError('unknown intervention')
    selected = []
    for name, parameter in model.named_parameters():
        enabled = arm != 'frozen' and name.startswith('actor.')
        enabled |= arm in ('IR', 'All') and name == 'input_weights'
        enabled |= arm == 'All' and name in ('chemical', 'gap_logits', 'bias')
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append(name)
    return selected


def ablate_afd_aiy(model):
    """Copy and silence only existing AFD->AIY chemical edges, no new edges."""
    altered = copy.deepcopy(model)
    mask = (altered.pre < 2) & (altered.post >= 2) & (altered.post < 4)
    edges = [dict(source=CORE[int(i)], target=CORE[int(j)], weight=float(w))
             for i, j, w in zip(altered.pre[mask], altered.post[mask], altered.chemical[mask].detach())]
    with torch.no_grad():
        altered.chemical[mask] = 0
    return altered, edges


def balanced_cases(config):
    """Use initial geometry only, before loading/evaluating any policy."""
    seeds, fields, counts = [], [], {(f, s): 0 for f in (0, 1) for s in (-1, 1)}
    for seed in range(810001, 820001):
        field = (seed-810001) % 2
        env = NavigationBatch(config, [seed], fields=[field])
        sign = -1 if env.offset[0] < 0 else 1
        if counts[field, sign] < 8:
            seeds.append(seed); fields.append(field); counts[field, sign] += 1
        if all(n == 8 for n in counts.values()):
            break
    if len(seeds) != 32:
        raise RuntimeError('could not construct balanced cases')
    # evaluate() assigns fields by batch index, so order must alternate 0,1.
    by_field = [[s for s, f in zip(seeds, fields) if f == i] for i in (0, 1)]
    seeds = [s for pair in zip(*by_field) for s in pair]
    env = NavigationBatch(config, seeds)
    return dict(seeds=seeds, cases=[dict(seed=s, field='linear' if i % 2 == 0 else 'radial',
        initial_error_K=float(env.offset[i]), event_s=float(env.event[i]),
        initial_x_m=float(env.x[i]), initial_y_m=float(env.y[i]), initial_heading_rad=float(env.heading[i]))
        for i, s in enumerate(seeds)])


def prepare(root):
    if root.exists():
        raise FileExistsError('preserve previous output; use a new root')
    parent = json.loads((PARENT/'design.json').read_text())
    controls = [s for s in parent['specs'] if s['encoding'] == 'raw']
    specs = [s for s in controls if s['name'] == 'real8']
    assert len(specs) == 6 and len(controls) == 78
    source_paths = [Path(__file__).resolve().relative_to(REPO), Path('tests/test_pathway_stage1.py'),
        Path('analysis/diagnose_navigation_failures.py'), Path('analysis/navigation_feasibility_replication.py'),
        Path('analysis/navigation_preflight.py'), Path('src/thermotaxis/navigation_window_policy.py'),
        Path('src/thermotaxis/navigation.py'), Path('src/thermotaxis/sensory_adaptation.py'),
        Path('src/thermotaxis/connectome.py')]
    inputs = [TEACHER, PARENT/'design.json', Path('data/connectome/cook2019_herm.json'),
        Path('docs/reports/assets/navigation-diagnosis-2026-11-02/diagnostic-records.zip')]
    inputs += [PARENT/f'fit-{s["id"]:03d}'/'BC.pt' for s in controls]
    config = replace(NavigationConfig(), field_turn_rad=math.pi/12)
    plan = dict(status='declared_before_new_fits_and_evaluation', date='2026-11-03',
        parent_main_commit='3b47ec48411b75d89eb2bf3dc4e96891fd3c1c8a', specs=specs,
        frozen_control_specs=controls, arms=list(ARMS), updates=1024, batch=256, lr=.003,
        fit_minibatch_seed='110300 + training seed, identical across arms/interfaces',
        teacher_train_episodes=list(range(24)), teacher_diagnostic_episodes=list(range(24,32)),
        config=asdict(config), fresh=balanced_cases(config), old_seeds=list(range(71001,71033)),
        stimulus_amplitudes_K=[.001,.002,.005], stimulus_seed=110201,
        harmonic_frequencies_Hz=[.25,.5,1.], online_probe_duration_s=12., online_probe_onset_s=4.,
        calibration='80 phase/background windows at 2mK, separate from online test streams',
        gain_matching='nonnegative RMS ratio on calibration evoked actions only; applied to held-out evoked differences, not deployed as a policy',
        primary='paired fresh-case complete success and linear-field post-stability; same updates, no claim of equal compute',
        limitations=['BC training and old diagnostic episodes are reused; fresh evaluation cases are separate.',
          'Eight-node anatomical masks, learned effective signs; no calibrated cell physiology.',
          'All arms independently restart BC; only real8 refitted; all 10 null graphs retained as frozen probes.',
          'Prescribed kinematics at 15deg, not the Boyle body; no new RL interactions.',
          'Online probe states are final states of causal windows, not intermediate unroll positions.'],
        source_sha256={str(p):digest(REPO/p) for p in source_paths},
        inputs_sha256={str(p):digest(REPO/p) for p in inputs},
        versions=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, platform=platform.platform()))
    write(root/'design.json', plan)
    print('Design frozen: 18 new fits, 6 baseline snapshots, 32 new balanced cases', flush=True)


def check(plan):
    for name, sha in {**plan['source_sha256'], **plan['inputs_sha256']}.items():
        if digest(REPO/name) != sha:
            raise RuntimeError('source/input changed: '+name)


def load_model(spec, path=None):
    torch.manual_seed(spec['seed'])
    model = WindowActorCritic(spec['name'], spec['interface'])
    path = path or PARENT/f'fit-{spec["id"]:03d}'/'BC.pt'
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.eval()


def split_ids(length):
    episode = np.arange(length) % 32
    return np.flatnonzero(episode < 24), np.flatnonzero(episode >= 24)


def dataset_mse(model, x, y, indices, features=None):
    total = 0.
    with torch.no_grad():
        for start in range(0, len(indices), 1024):
            ids = indices[start:start+1024]
            logit = model.actor(features[ids]).squeeze(-1) if features is not None else model.actor_mean(x[ids])
            total += float((torch.tanh(logit)-y[ids]).square().sum())
    return total/len(indices)


def train(root, plan):
    with np.load(TEACHER) as z:
        x, y = torch.from_numpy(z['x'].copy()), torch.from_numpy(z['y'].copy())
    train_ids, diagnostic_ids = split_ids(len(y))
    for spec in plan['specs']:
        for arm in plan['arms']:
            dest = root/f'fit-{spec["id"]:03d}-{arm}'
            if (dest/'record.json').exists():
                record = json.loads((dest/'record.json').read_text())
                if record['checkpoint_sha256'] != digest(dest/'model.pt'):
                    raise RuntimeError('changed completed model')
                continue
            dest.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            model = load_model(spec)
            before = {k:v.detach().clone() for k,v in model.state_dict().items()}
            selected = configure_trainable(model, arm)
            features = None
            if arm in ('frozen','R'):
                with torch.no_grad():
                    features = torch.cat([representation(model, x[i:i+1024]).detach() for i in range(0,len(x),1024)])
            initial = dict(train_mse=dataset_mse(model,x,y,train_ids,features),
                           diagnostic_mse=dataset_mse(model,x,y,diagnostic_ids,features))
            curves = []
            if selected:
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=plan['lr'])
                rng = np.random.default_rng(110300+spec['seed'])
                for update in range(plan['updates']):
                    ids = rng.choice(train_ids, plan['batch'], replace=True)
                    logit = model.actor(features[ids]).squeeze(-1) if features is not None else model.actor_mean(x[ids])
                    loss = (torch.tanh(logit)-y[ids]).square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite fitting loss')
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    if update % 128 == 0 or update+1 == plan['updates']:
                        curves.append(dict(update=update+1,minibatch_mse=float(loss.detach())))
            changed = [k for k,v in before.items() if not torch.equal(v,model.state_dict()[k])]
            if not set(changed) <= set(selected):
                raise RuntimeError('frozen parameter or buffer changed')
            torch.save(model.state_dict(),dest/'model.pt')
            record = dict(spec=spec,arm=arm,updates=plan['updates'] if selected else 0,
                trainable_names=selected,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                changed_names=changed,initial=initial,
                final=dict(train_mse=dataset_mse(model,x,y,train_ids,features),
                           diagnostic_mse=dataset_mse(model,x,y,diagnostic_ids,features)),
                curves=curves,checkpoint_sha256=digest(dest/'model.pt'),wall_s=time.monotonic()-started,
                complete=True)
            write(dest/'record.json',record)
            print('Completed fit',spec['id'],spec['interface'],spec['seed'],arm,record['final'],flush=True)
    check(plan)


def case_summary(cases):
    result = summary(cases)
    for field in ('linear','radial'):
        selected = [c for c in cases if c['field']==field]
        result[field] = summary(selected)
    return result


def evaluation(root, plan):
    config = NavigationConfig(**plan['config'])
    for spec in plan['specs']:
        for arm in plan['arms']:
            dest = root/f'evaluation-{spec["id"]:03d}-{arm}.json'
            if dest.exists():
                continue
            checkpoint = root/f'fit-{spec["id"]:03d}-{arm}'/'model.pt'
            model = load_model(spec,checkpoint)
            altered, edges = ablate_afd_aiy(model)
            fresh = evaluate(model,config,plan['fresh']['seeds'])
            old = evaluate(model,config,plan['old_seeds'])
            lesion = evaluate(altered,config,plan['fresh']['seeds'])
            row = dict(spec=spec,arm=arm,checkpoint_sha256=digest(checkpoint),
                fresh=dict(cases=fresh,summary=case_summary(fresh)),
                old=dict(cases=old,summary=case_summary(old)),
                lesion=dict(cases=lesion,summary=case_summary(lesion),edges=edges),complete=True)
            write(dest,row)
            print('Completed evaluation',spec['id'],arm,flush=True)
    check(plan)


def online_windows(amplitude, kind, frequency):
    """Recompute features at each observation; do not trace 'time' within a window."""
    phases = .19 + np.arange(4)*math.pi/2
    phases = np.tile(phases,3)
    backgrounds = np.repeat([-.06,0.,.06],4)
    a, b = NavigationFeatures(12,'raw'), NavigationFeatures(12,'raw')
    altered, base = [], []
    times = np.arange(121)*.1
    for t in times:
        phase = phases+2*math.pi*frequency*t
        pattern = np.sin(phase) if kind=='phase' else np.ones(12)
        delta = amplitude*pattern if t >= 4-1e-9 else np.zeros(12)
        packet = np.stack([293.15+backgrounds,np.sin(phase),np.cos(phase),np.ones(12),np.ones(12)],axis=1)[None].astype('float32')
        changed = packet.copy();changed[0,:,0] += delta.astype('float32')
        base.append(b.observe(packet));altered.append(a.observe(changed))
    return times,np.stack(base),np.stack(altered)


def last_states(model, x):
    shape = x.shape[:-2]
    x = torch.from_numpy(x.reshape(-1,41,7))
    actions, states = [], []
    with torch.no_grad():
        for start in range(0,len(x),512):
            logit,h,_ = trace_actor(model,x[start:start+512])
            actions.append((.3*torch.tanh(logit)).numpy())
            states.append(h[:,-1].numpy())
    return np.concatenate(actions).reshape(shape),np.concatenate(states).reshape(*shape,8)


def calibration_gain(model, altered):
    x, patterns = stimuli()
    changed = temperature_perturbation(x,.002*patterns['phase'])
    with torch.no_grad():
        ref = .3*(torch.tanh(model.actor_mean(changed))-torch.tanh(model.actor_mean(x)))
        les = .3*(torch.tanh(altered.actor_mean(changed))-torch.tanh(altered.actor_mean(x)))
    ref_rms,les_rms = float(ref.square().mean().sqrt()),float(les.square().mean().sqrt())
    return dict(reference_rms=ref_rms,lesion_rms=les_rms,
                gain=None if les_rms<1e-10 else ref_rms/les_rms)


def response_metrics(time_s, action, lesion, reference_node, lesion_node, gain, kind, frequency):
    selected = time_s>=4-1e-9
    late = time_s>=8-1e-9
    ref,les = action[selected],lesion[selected]
    ref_rms = float(np.sqrt(np.mean(ref**2)))
    scale = gain['gain']
    cosine = None if np.linalg.norm(ref)*np.linalg.norm(les)<1e-15 else float(np.sum(ref*les)/(np.linalg.norm(ref)*np.linalg.norm(les)))
    matched_error = None if scale is None or ref_rms<1e-10 else float(np.sqrt(np.mean((scale*les-ref)**2))/ref_rms)
    def half_latency(v):
        rms=np.sqrt(np.mean(v**2,axis=1));peak=float(rms.max())
        ids=np.flatnonzero((time_s>=4-1e-9)&(rms>=.5*peak))
        return None if peak<1e-8 or not len(ids) else float(time_s[ids[0]]-4)
    def harmonic_phase(v):
        if kind!='phase':return None
        phase = .19+np.tile(np.arange(4)*math.pi/2,3)[None,:]+2*math.pi*frequency*time_s[late,None]
        design=np.stack([np.sin(phase).ravel(),np.cos(phase).ravel(),np.ones(phase.size)],axis=1)
        coef=np.linalg.lstsq(design,v[late].ravel(),rcond=None)[0]
        return None if np.hypot(coef[0],coef[1])<1e-9 else float(np.arctan2(coef[1],coef[0]))
    return dict(action_rms_rad_s=ref_rms,lesion_action_rms_rad_s=float(np.sqrt(np.mean(les**2))),
        node_rms=np.sqrt(np.mean(reference_node[selected]**2,axis=(0,1))).tolist(),
        lesion_node_rms=np.sqrt(np.mean(lesion_node[selected]**2,axis=(0,1))).tolist(),
        evoked_waveform_cosine=cosine,gain_matched_normalized_rmse=matched_error,
        reference_t50_s=half_latency(action) if kind=='step' else None,
        lesion_t50_s=half_latency(lesion) if kind=='step' else None,
        reference_harmonic_phase_rad=harmonic_phase(action),lesion_harmonic_phase_rad=harmonic_phase(lesion))


def probe(root, plan):
    protocols = [(a,k,.5) for a in plan['stimulus_amplitudes_K'] for k in ('phase','step')]
    protocols += [(.002,'phase',f) for f in (.25,1.)]
    windows = [(a,k,f,*online_windows(a,k,f)) for a,k,f in protocols]
    for spec in plan['specs']:
        for arm in plan['arms']:
            dest=root/f'probe-{spec["id"]:03d}-{arm}.json'
            if dest.exists():continue
            model=load_model(spec,root/f'fit-{spec["id"]:03d}-{arm}'/'model.pt')
            altered,edges=ablate_afd_aiy(model)
            gain=calibration_gain(model,altered)
            rows=[];raw={}
            for i,(amplitude,kind,frequency,t,x0,x1) in enumerate(windows):
                a0,h0=last_states(model,x0);a1,h1=last_states(model,x1)
                b0,g0=last_states(altered,x0);b1,g1=last_states(altered,x1)
                da,db,dh,dg=a1-a0,b1-b0,h1-h0,g1-g0
                metrics=response_metrics(t,da,db,dh,dg,gain,kind,frequency)
                rows.append(dict(amplitude_K=amplitude,kind=kind,frequency_Hz=frequency,**metrics))
                raw.update({f'time_{i}':t,f'action_{i}':da,f'lesion_action_{i}':db,
                    f'node_{i}':dh,f'lesion_node_{i}':dg})
            np.savez_compressed(root/f'probe-{spec["id"]:03d}-{arm}.npz',**raw)
            write(dest,dict(spec=spec,arm=arm,calibration_gain=gain,edges=edges,rows=rows,complete=True))
            print('Completed causal probes',spec['id'],arm,flush=True)
    control_dest=root/'frozen-controls.json'
    if not control_dest.exists():
        x,patterns=stimuli();rows=[]
        with torch.no_grad():
            for spec in plan['frozen_control_specs']:
                model=load_model(spec)
                base=.3*torch.tanh(model.actor_mean(x))
                for amp in plan['stimulus_amplitudes_K']:
                    changed=temperature_perturbation(x,amp*patterns['phase'])
                    delta=.3*torch.tanh(model.actor_mean(changed))-base
                    row=dict(spec=spec,amplitude_K=amp,action_rms_rad_s=float(delta.square().mean().sqrt()))
                    if not model.is_gru:
                        _,h0,_=trace_actor(model,x);_,h1,_=trace_actor(model,changed)
                        row['node_final_rms_delta']=(h1[:,-1]-h0[:,-1]).square().mean(0).sqrt().tolist()
                    rows.append(row)
        write(control_dest,dict(rows=rows,complete=True))
    check(plan)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','train','evaluate','probe'])
    parser.add_argument('--root',type=Path,default=Path('results/neural-pathway-stage1-2026-11-03'))
    args=parser.parse_args();torch.set_num_threads(1)
    if args.stage=='prepare':prepare(args.root);return
    plan=json.loads((args.root/'design.json').read_text());check(plan)
    {'train':train,'evaluate':evaluation,'probe':probe}[args.stage](args.root,plan)


if __name__=='__main__':main()
