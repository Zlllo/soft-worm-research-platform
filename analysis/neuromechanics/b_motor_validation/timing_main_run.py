"""Six authorized main cases only, old loop/32 fields plus passive reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[3]/'results/c1-current-timing-2026-11-10'
EXPECTED='8b9d11bda03f952c96d64d5b2b58f92496ecdcc008e5c471a51463664320f12a'
GATES=[('ca_boyle','e_q'),('ca_boyle','f_q'),('k_fast','p_q'),('k_fast','q_q'),('k_slow','n_q')]
COLUMNS=['time_ms','voltage_mV','peer_at_record_mV','last_applied_peer_mV','synapse_s','synapse_i_nA',
    'ica_mA_cm2','concentration_mM','cai_mM','drive_cai','drive_STATE',
    'e_q','f_q','p_q','q_q','n_q','cache_e','cache_f','cache_p','cache_q','cache_n',
    'cache_h','h_q','h_inf','previous_e','previous_f','previous_p','previous_q','previous_n',
    'commanded_ica_mA_cm2','source_ica_mA_cm2','pool_iCa_nA']
EXTRA_COLUMNS=['pre_last_time_ms','pre_last_voltage_mV','pre_last_e','pre_last_f','pre_last_p','pre_last_q','pre_last_n',
    'pre_last_cache_e','pre_last_cache_f','pre_last_cache_p','pre_last_cache_q','pre_last_cache_n',
    'pre_last_h_fcond','pre_last_h_q','pre_last_STATE_concentration_mM','pre_last_cai_mM',
    'post_I_ca_mA_cm2','post_I_fast_mA_cm2','post_I_slow_mA_cm2',
    'post_gion_ca_S_cm2','post_gion_fast_S_cm2','post_gion_slow_S_cm2',
    'post_fopen_ca','post_fopen_fast','post_fopen_slow','post_ion_eca_mV','post_ion_ek_mV']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def run(variant,dt):
    assert variant in ['euler_direct','exp_direct'] and dt in [.025,.0125,.00625]
    lock=read(ROOT/'main-execution-lock.json')
    assert lock['protocol_sha256']==EXPECTED==sha(ROOT/'protocol.json')
    assert lock['runner_sha256']==sha(Path(__file__)) and lock['main_conditions_authorized']==6
    kernel=ROOT/'proposed-variants'/variant
    for name,digest in lock['kernel_sha256'][variant].items():assert sha(kernel/name)==digest
    assert sha(ROOT/'presynaptic-input.npz')==lock['input_sha256']
    out=ROOT/'main-runs'/f'{variant}_dt{dt:g}';out.mkdir(parents=True,exist_ok=False)
    spec=dict(variant=variant,dt_ms=dt,duration_ms=8000,record_ms=.025,protocol_sha256=EXPECTED,
        runner_sha256=sha(Path(__file__)),execution_lock_sha256=sha(ROOT/'main-execution-lock.json'),
        input_sha256=lock['input_sha256'],scientific_scope='isolated C1 muscle only')
    write(out/'spec.json',spec);started=time.monotonic();failure=None;row=0;executed=0
    import neuron
    h=neuron.h;before=Path.cwd()
    try:
        os.chdir(kernel);assert neuron.load_mechanisms('.')
        h.load_file('stdrun.hoc');assert h.load_file('GenericMuscleCell.hoc')
    finally:os.chdir(before)
    cell=h.GenericMuscleCell();seg=cell.soma(.5)
    h.cai0_ca_ion=0;h.cao0_ca_ion=2
    keep=[h.FInitializeHandler(0,lambda:cell.set_initial_v()),h.FInitializeHandler(1,lambda:cell.set_initial_ion_properties())]
    assert h.secondorder==0
    h.dt=dt;h.steps_per_ms=1/dt;h.cvode_active(0)
    peer=h.Vector(1);peer.x[0]=-60
    with np.load(ROOT/'presynaptic-input.npz') as z:inp=z['voltage_mV'].copy()
    nmj=h.neuron_to_neuron_exc_syn(seg);nmj.Vth=-30;nmj.conductance=.0009;nmj.weight=1
    h.setpointer(peer._ref_x[0],'vpeer',nmj)
    h.finitialize(-45)
    gate_objects=[(getattr(seg,m),s) for m,s in GATES]
    def gates():return [float(getattr(obj,name)) for obj,name in gate_objects]
    def caches():return [float(seg.ca_boyle.e_fcond),float(seg.ca_boyle.f_fcond),float(seg.k_fast.p_fcond),float(seg.k_fast.q_fcond),float(seg.k_slow.n_fcond)]
    # This is the old main record list in the same order, with zero-valued
    # external calcium-source fields because that helper was never inserted.
    def record(t,previous,last):
        ca=float(seg.cai);state=float(seg.CaPool.concentration)
        ca_pos=max(ca,0.);state_pos=max(state,0.)
        return [t,float(seg.v),float(peer.x[0]),last,float(nmj.s),float(nmj.i),
            float(seg.ica),state,ca,.3*ca_pos/(ca_pos+2.5e-7),.3*state_pos/(state_pos+2.5e-7),
            *gates(),*caches(),float(seg.ca_boyle.h_fcond),float(seg.ca_boyle.h_q),float(seg.ca_boyle.h_inf),*previous,
            0.,0.,float(h.iCa_CaPool)]
    def pre_snapshot():
        return [float(h.t),float(seg.v),*gates(),*caches(),float(seg.ca_boyle.h_fcond),float(seg.ca_boyle.h_q),
            float(seg.CaPool.concentration),float(seg.cai)]
    def post_snapshot():
        return [-float(seg.ca_boyle.i__ca_boyle),-float(seg.k_fast.i__k_fast),-float(seg.k_slow.i__k_slow),
            float(seg.ca_boyle.gion),float(seg.k_fast.gion),float(seg.k_slow.gion),
            float(seg.ca_boyle.fopen),float(seg.k_fast.fopen),float(seg.k_slow.fopen),float(seg.eca),float(seg.ek)]
    count=round(8000/dt);stride=round(.025/dt);input_stride=round(dt/.00625)
    data=np.empty((320001,len(COLUMNS)));extra=np.empty((320001,len(EXTRA_COLUMNS)));previous=gates()
    data[0]=record(0.,previous,-60.);extra[0]=pre_snapshot()+post_snapshot();row=1
    try:
        for tick in range(count):
            peer.x[0]=float(inp[tick*input_stride])
            last=float(peer.x[0])
            if (tick+1)%stride==0:
                previous=gates();pre=pre_snapshot()
            h.fadvance();executed+=1
            if (tick+1)%stride==0:
                peer.x[0]=float(inp[(tick+1)*input_stride])
                data[row]=record((tick+1)*dt,previous,last);extra[row]=pre+post_snapshot();row+=1
            if (tick+1)%round(1000/dt)==0:print(out.name,(tick+1)*dt/1000,'s',flush=True)
        assert abs(h.t-8000)<1e-5 and h.dt==dt
        if not np.isfinite(data[:row]).all() or not np.isfinite(extra[:row]).all():failure='Nonfinite recorded value'
    except Exception:failure=traceback.format_exc()
    np.savez_compressed(out/'trace.npz',columns=np.array(COLUMNS),data=data[:row],extra_columns=np.array(EXTRA_COLUMNS),extras=extra[:row],
        neural_clock_ms=np.array(float(h.t)),record_interval_ms=np.array(.025))
    result=dict(spec=spec,complete=failure is None,failure=failure,rows=row,executed_fadvance=executed,
        explicit_fcurrent_calls=0,manual_rates_calls=0,observer_hooks=0,new_initialization_handlers=0,
        actual_dt_ms=float(h.dt),actual_neural_clock_ms=float(h.t),secondorder=float(h.secondorder),cvode_active=float(h.cvode_active()),
        area_um2=float(h.area(.5,sec=cell.soma)),geometry=dict(L_um=float(cell.soma.L),diam_um=float(seg.diam),cm_uF_cm2=float(cell.soma.cm)),
        gmax_S_cm2={m:float(getattr(seg,m).gmax) for m in ['ca_boyle','k_fast','k_slow']},
        channel_internal_reversal_mV=dict(ca=40.,fast_K=-60.,slow_K=-60.),
        channel_reversal_provenance='Frozen MOD INITIAL and generated storage; separately confirmed by zero-step V-I/gion. Not seg.eca.',
        instantaneous_ion_reversal_mV=dict(ca=float(seg.eca),k=float(seg.ek)),
        current_phase='post current fields are existing values after natural fadvance; pre_last fields capture the last pre-advance V/gates/cache/h. No active query.',
        versions=dict(neuron=neuron.__version__,numpy=np.__version__,python=sys.version),trace_sha256=sha(out/'trace.npz'),wall_s=time.monotonic()-started)
    write(out/'record.json',result);print(json.dumps(dict(name=out.name,complete=result['complete'],rows=row,wall_s=result['wall_s'])))
    if failure:raise RuntimeError(failure)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['euler_direct','exp_direct'],required=True)
    p.add_argument('--dt',choices=[.025,.0125,.00625],type=float,required=True);a=p.parse_args();run(a.variant,a.dt)
