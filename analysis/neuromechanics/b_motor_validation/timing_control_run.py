"""V2 controls only. No main/full-network/body execution entry point exists."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[3]/'results/c1-current-timing-2026-11-10'
EXPECTED='8b9d11bda03f952c96d64d5b2b58f92496ecdcc008e5c471a51463664320f12a'
GATES=[('ca_boyle','e'),('ca_boyle','f'),('k_fast','p'),('k_fast','q'),('k_slow','n')]
NAMES=['time_ms','voltage_mV','concentration_mM','cai_mM','h_q','h_fcond',
    'e','f','p','q','n','cache_e','cache_f','cache_p','cache_q','cache_n',
    'I_ca_mA_cm2','I_fast_mA_cm2','I_slow_mA_cm2','sum_ica_mA_cm2','sum_ik_mA_cm2',
    'gion_ca_S_cm2','gion_fast_S_cm2','gion_slow_S_cm2','fopen_ca','fopen_fast','fopen_slow']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def run(kind,variant,voltage):
    lock=read(ROOT/'control-execution-lock.json')
    assert sha(ROOT/'protocol.json')==EXPECTED==lock['protocol_sha256']
    assert lock['runner_sha256']==sha(Path(__file__)) and lock['scientific_stage']=='v2_controls_only'
    assert variant in ['euler_cached','exp_cached','euler_direct','exp_direct']
    if kind=='short':assert voltage==-10
    else:assert voltage in [-20,-10]
    build=read(ROOT/'generated-code-validation-v2.json');assert build['passed']
    parent='original' if variant.startswith('euler') else 'both'
    kernel=ROOT/'frozen'/parent if variant.endswith('cached') else ROOT/'proposed-variants'/variant
    for name,digest in lock['kernel_sha256'][variant].items():assert sha(kernel/name)==digest
    name=f'{kind}_{variant}_v{voltage:g}';out=ROOT/'controls'/name;out.mkdir(parents=True,exist_ok=False)
    spec=dict(kind=kind,variant=variant,voltage_mV=voltage,dt_ms=.025,duration_ms=0 if kind=='zero' else 5,
        protocol_sha256=EXPECTED,runner_sha256=sha(Path(__file__)),execution_lock_sha256=sha(ROOT/'control-execution-lock.json'))
    write(out/'spec.json',spec);start=time.monotonic();rows=[];phases=[];step_ids=[];failure=None;explicit_fcurrent=0;steps=0
    import neuron
    h=neuron.h
    try:
        before=Path.cwd()
        try:
            os.chdir(kernel);assert neuron.load_mechanisms('.')
            h.load_file('stdrun.hoc');assert h.load_file('GenericMuscleCell.hoc')
        finally:os.chdir(before)
        assert h.secondorder==0
        h.dt=.025;h.steps_per_ms=40;h.cvode_active(0)
        cell=h.GenericMuscleCell();seg=cell.soma(.5);clamp=None
        if kind=='short':
            for m in ['Leak','ca_boyle','k_slow']:getattr(seg,m).gmax=0
            clamp=h.SEClamp(seg);clamp.rs=1e-9;clamp.dur1=1e9;clamp.amp1=-45
        h.cai0_ca_ion=0;h.cao0_ca_ion=2
        keep=[h.FInitializeHandler(0,lambda:cell.set_initial_v()),h.FInitializeHandler(1,lambda:cell.set_initial_ion_properties())]
        h.finitialize(-45)
        def snapshot(phase,index):
            row=[float(h.t),float(seg.v),float(seg.CaPool.concentration),float(seg.cai),float(seg.ca_boyle.h_q),float(seg.ca_boyle.h_fcond),
                *[float(getattr(getattr(seg,m),p+'_q')) for m,p in GATES],
                *[float(getattr(getattr(seg,m),p+'_fcond')) for m,p in GATES],
                -float(seg.ca_boyle.i__ca_boyle),-float(seg.k_fast.i__k_fast),-float(seg.k_slow.i__k_slow),float(seg.ica),float(seg.ik),
                float(seg.ca_boyle.gion),float(seg.k_fast.gion),float(seg.k_slow.gion),
                float(seg.ca_boyle.fopen),float(seg.k_fast.fopen),float(seg.k_slow.fopen)]
            rows.append(row);phases.append(phase);step_ids.append(index)
        snapshot('native_finitialize_return',0)
        seg.v=voltage
        if kind=='zero':
            for (mech,p),value in zip(GATES,[.8,.2,.7,.1,.6]):setattr(getattr(seg,mech),p+'_q',value)
            for (mech,p),value in zip(GATES,[.11,.61,.08,.77,.19]):setattr(getattr(seg,mech),p+'_fcond',value)
            snapshot('algebra_before_explicit_fcurrent',0)
            h.fcurrent();explicit_fcurrent+=1
            snapshot('algebra_after_explicit_fcurrent',0)
        else:
            clamp.amp1=voltage
            k=seg.k_fast
            q_inf=k.q_steadyState_rate/(1+math.exp(-(voltage-k.q_steadyState_midpoint)/k.q_steadyState_scale))
            k.p_q=.7;k.q_q=q_inf
            k.p_fcond=.7**k.p_instances;k.q_fcond=q_inf**k.q_instances
            snapshot('control_state_and_cache_aligned_once',0)
            for step in range(1,201):
                snapshot('before_fadvance',step)
                h.fadvance();steps+=1
                snapshot('after_fadvance',step)
        info=dict(secondorder=float(h.secondorder),cvode_active=float(h.cvode_active()),actual_dt_ms=float(h.dt),
            actual_time_ms=float(h.t),area_um2=float(h.area(.5,sec=cell.soma)),
            gmax_S_cm2={m:float(getattr(seg,m).gmax) for m in ['ca_boyle','k_fast','k_slow']},
            reversal_mV=dict(ca=float(seg.eca),k=float(seg.ek)),
            gate_parameters={p:dict(power=float(getattr(getattr(seg,m),p+'_instances')),
                tau_ms=float(getattr(getattr(seg,m),p+'_timeCourse_tau')),
                midpoint_mV=float(getattr(getattr(seg,m),p+'_steadyState_midpoint')),
                scale_mV=float(getattr(getattr(seg,m),p+'_steadyState_scale')),
                rate=float(getattr(getattr(seg,m),p+'_steadyState_rate'))) for m,p in GATES},
            h_alpha=float(seg.ca_boyle.h_alpha),versions=dict(neuron=neuron.__version__,numpy=np.__version__,python=sys.version))
    except Exception:
        failure=traceback.format_exc();info={}
    if rows:
        values=np.array(rows)
        if not np.isfinite(values).all():failure='Nonfinite saved data\n'+str(failure)
        np.savez_compressed(out/'trace.npz',data=values,columns=np.array(NAMES),phase=np.array(phases),step=np.array(step_ids))
    result=dict(spec=spec,complete=failure is None,failure=failure,executed_fadvance=steps,explicit_fcurrent_calls=explicit_fcurrent,
        physical_observer_hooks=0,rows=len(rows),wall_s=time.monotonic()-start,metadata=info,
        trace_sha256=sha(out/'trace.npz') if rows else None)
    write(out/'record.json',result);print(json.dumps(dict(name=name,complete=result['complete'],steps=steps,explicit_fcurrent=explicit_fcurrent)))
    if failure:raise RuntimeError(failure)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['zero','short'],required=True)
    p.add_argument('--variant',required=True);p.add_argument('--voltage',type=float,required=True)
    a=p.parse_args();run(a.kind,a.variant,a.voltage)
