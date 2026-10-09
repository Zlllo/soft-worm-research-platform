#!/usr/bin/env python3
"""Predeclared deterministic cardinal-command bridge benchmarks."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, subprocess
from pathlib import Path
from dataclasses import replace
import numpy as np
from thermotaxis.phase2_control import ControlConfig, run_four_action, action_heading, wrap_angle, ACTION_NAMES


def summarize(result, action):
    rows=result['trajectory']; dt=result['config']['physics_dt_s']
    error=np.array([abs(wrap_angle(action_heading(action)-r['axis_heading_rad'])) for r in rows])
    window=round(1/dt)+1
    settle=next((rows[i]['time_s'] for i in range(len(rows)-window+1) if max(error[i:i+window]) <= .25),None)
    tail=[r for r in rows if r['time_s'] >= rows[-1]['time_s']-4]
    return {'action':action,'name':ACTION_NAMES[action],'final_x_mm':rows[-1]['x_m']*1e3,'final_y_mm':rows[-1]['y_m']*1e3,'final_axis_heading_rad':rows[-1]['axis_heading_rad'],'tail_mean_abs_axis_error_rad':float(np.mean(error[-len(tail):])),'first_1s_within_025rad_s':settle,'dissipation_pJ':rows[-1]['dissipation_J']*1e12,'max_power_pW':max(r['power_W'] for r in rows)*1e12,**result['diagnostics']}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results/phase2-four-action-2026-10-04'); args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]; out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    config=ControlConfig.from_dict(json.loads((root/'configs/phase2_control_reference.json').read_text()))
    design={'schema_version':1,'kind':'deterministic_interface_benchmark','cardinal_actions':[0,1,2,3],'cardinal_duration_s':24,'sequence_actions':[2,0,3,1],'sequence_hold_s':24,'sequence_duration_s':96,'settling_metric':'first complete 1s with every axis heading error <=.25rad','tail_window_s':4,'dt_compare_s':[.02,.01],'no_contact':True,'no_training':True,'no_randomness':False,'randomness':'frozen Q-table tie RNG only; seed 44001','frozen_policy_demo':{'planned_duration_s':96,'initial_center_m':[.001,.002],'initial_orientation_rad':0,'grid_extent_m':[.004,.004],'tie_seed':44001,'on_head_exit':'terminal at first out-of-domain physics sample; no projection'},'initial_forward_axis':'left; theta=0, forward axis=theta+pi','observation_mode':'local_temperature (scripted commands ignore observation)'}
    (out/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    results={}; summary=[]; convergence=[]
    for a in range(4):
        r=run_four_action(config,lambda obs,a=a:a); results[ACTION_NAMES[a]]=r; summary.append(summarize(r,a))
        fine=run_four_action(replace(config,physics_dt_s=.01),lambda obs,a=a:a)
        end,fend=r['trajectory'][-1],fine['trajectory'][-1]
        convergence.append({'action':a,'position_error_nm':math.hypot(end['x_m']-fend['x_m'],end['y_m']-fend['y_m'])*1e9,'orientation_error_rad':abs(wrap_angle(end['orientation_rad']-fend['orientation_rad'])),'dissipation_relative_error':abs(end['dissipation_J']/fend['dissipation_J']-1)})
    results['sequence']=run_four_action(replace(config,duration_s=96),lambda obs:design['sequence_actions'][min(3,int(obs.time_s//24))])
    results['never_started']=run_four_action(replace(config,duration_s=4),lambda obs:None)
    results['relax_after_stop']=run_four_action(replace(config,duration_s=16),lambda obs:2 if obs.time_s<4 else None)
    from thermotaxis.legacy_policy import FrozenQTable, LegacyGridMap
    policy_path=root/'docs/reports/assets/legacy-policy-bridge-2026-10-04/new-training-interface-qtable.json'
    frozen=FrozenQTable.from_json(policy_path,tie_seed=44001)
    grid=LegacyGridMap(17,17,.004,.004,y_axis_up=True)
    before=hashlib.sha256(frozen.q_table.tobytes()).hexdigest()
    results['frozen_qtable']=run_four_action(replace(config,duration_s=96,initial_x_m=.001,initial_y_m=.002,observation_mode='legacy_privileged'),lambda obs:frozen.select_action(*obs),legacy_observer=lambda obs,head:grid.indices(head),terminal_condition=lambda head,time:any(x<0 or x>.004 for x in head))
    after=hashlib.sha256(frozen.q_table.tobytes()).hexdigest()
    frozen_result=results['frozen_qtable']
    (out/'frozen-policy-summary.json').write_text(json.dumps({'policy_asset':str(policy_path.relative_to(root)),'asset_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest(),'q_values_sha256_before':before,'q_values_sha256_after':after,'unchanged':before==after,'provenance':'new old-platform training sample, not recovered historical weights','training_metadata':json.loads(policy_path.read_text())['training'],'observation_condition':'legacy_privileged head-position lookup; no local-temperature policy','mapping':'SI y up; x_idx=round(x/.004*16), y_idx=round((.004-y)/.004*16)','config':frozen_result['config'],'controls_executed':len(frozen_result['controls']),'terminal':frozen_result['termination'],'final':frozen_result['trajectory'][-1]},indent=2)+'\n')
    for name,result in results.items():
        (out/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
        with (out/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(result['trajectory'][0]));writer.writeheader();writer.writerows(result['trajectory'])
    (out/'summary.json').write_text(json.dumps({'cardinal':summary,'convergence':convergence,'rest':results['never_started']['trajectory'][-1],'stopped':results['relax_after_stop']['trajectory'][-1]},indent=2)+'\n')
    def git(*args):
        return subprocess.run(['git',*args],cwd=root,capture_output=True,text=True).stdout.strip()
    files=['analysis/phase2_four_action_bridge.py','src/thermotaxis/phase2_control.py','src/thermotaxis/phase2_body.py','configs/phase2_control_reference.json','src/thermotaxis/legacy_policy.py','docs/reports/assets/legacy-policy-bridge-2026-10-04/new-training-interface-qtable.json']
    (out/'manifest.json').write_text(json.dumps({'git_sha':git('rev-parse','HEAD'),'git_dirty':bool(git('status','--porcelain')),'python':platform.python_version(),'sources_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},'command':'PYTHONPATH=src python3 analysis/phase2_four_action_bridge.py','design':design},indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    for a in range(4):
        r=results[ACTION_NAMES[a]]['trajectory'];t=np.array([z['time_s'] for z in r]);xy=np.array([[z['x_m'],z['y_m']] for z in r])*1e3
        axes[0].plot(xy[:,0],xy[:,1],label=f'{a}: {ACTION_NAMES[a]}');axes[0].scatter(*xy[-1],s=18)
        err=[abs(wrap_angle(action_heading(a)-z['axis_heading_rad'])) for z in r]
        axes[1].plot(t,err,label=ACTION_NAMES[a])
    axes[0].set(xlabel='centre x (mm)',ylabel='centre y (mm)',title='Held cardinal requests; actual RFT motion');axes[0].axis('equal');axes[0].legend()
    axes[1].axhline(.25,color='black',ls=':',label='0.25 rad band');axes[1].set(xlabel='time (s)',ylabel='absolute axis error (rad)',title='Finite turn delay and gait oscillation');axes[1].legend()
    seq=results['sequence']['trajectory']; axes[2].plot([z['x_m']*1e3 for z in seq],[z['y_m']*1e3 for z in seq]);
    for block,a in enumerate(design['sequence_actions']):
        q=seq[round(block*24/config.physics_dt_s)];axes[2].scatter(q['x_m']*1e3,q['y_m']*1e3);axes[2].annotate(f'{block*24}s: {ACTION_NAMES[a]}',(q['x_m']*1e3,q['y_m']*1e3),xytext=(-35,5),textcoords='offset points')
    axes[2].set(xlabel='centre x (mm)',ylabel='centre y (mm)',title='Four held commands; continuous trajectory');axes[2].axis('equal')
    fig.tight_layout();fig.savefig(out/'four-action-bridge.png',dpi=160);fig.savefig(out/'four-action-bridge.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,5))
    rr=frozen_result['trajectory']; ax.plot([z['head_x_m']*1e3 for z in rr],[z['head_y_m']*1e3 for z in rr],label='actual head trajectory')
    ax.scatter(rr[0]['head_x_m']*1e3,rr[0]['head_y_m']*1e3,label='start')
    ax.scatter(rr[-1]['head_x_m']*1e3,rr[-1]['head_y_m']*1e3,marker='x',label='first sampled exit')
    ax.scatter(2,2,marker='*',s=100,label='old-task heat maximum')
    ax.set(xlim=(-.1,4.1),ylim=(-.1,4.1),xlabel='SI head x (mm)',ylabel='SI head y (mm)',title='Frozen Q-table transfer; early domain exit')
    ax.set_aspect('equal');ax.legend();fig.tight_layout();fig.savefig(out/'frozen-policy-transfer.png',dpi=160);fig.savefig(out/'frozen-policy-transfer.svg');plt.close(fig)
    print(json.dumps({'output':str(out),'cardinal':summary,'convergence':convergence},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
