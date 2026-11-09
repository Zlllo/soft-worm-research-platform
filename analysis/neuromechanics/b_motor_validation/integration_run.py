"""One isolated muscle or analytic control per fresh NEURON process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'results/c1-muscle-integration-isolation-2026-11-09'
GATES=[('ca_boyle','e_q'),('ca_boyle','f_q'),('k_fast','p_q'),('k_fast','q_q'),('k_slow','n_q')]
COLUMNS=['time_ms','voltage_mV','peer_at_record_mV','last_applied_peer_mV','synapse_s','synapse_i_nA',
    'ica_mA_cm2','concentration_mM','cai_mM','drive_cai','drive_STATE',
    'e_q','f_q','p_q','q_q','n_q','cache_e','cache_f','cache_p','cache_q','cache_n',
    'cache_h','h_q','h_inf','previous_e','previous_f','previous_p','previous_q','previous_n',
    'commanded_ica_mA_cm2','source_ica_mA_cm2','pool_iCa_nA']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def run(case,variant,dt):
    protocol=read(ROOT/'protocol.json');assert sha(ROOT/'protocol.json')==read(ROOT/'protocol-lock.json')['protocol_sha256']
    builds=read(ROOT/'build-evidence.json');kernel=ROOT/'variants'/variant
    for path,digest in builds[variant]['sha256'].items():assert sha(kernel/path)==digest
    assert dt in protocol['dt_ms']
    assert variant in (protocol['variants'] if case=='main' else protocol['controls']['variants'])
    out=ROOT/'runs'/f'{case}_{variant}_dt{dt:g}';out.mkdir(parents=True,exist_ok=False)
    spec=dict(case=case,variant=variant,dt_ms=dt,protocol_sha256=sha(ROOT/'protocol.json'),
        input_sha256=protocol['input_sha256'] if case=='main' else None,
        runner_sha256=sha(Path(__file__)))
    write(out/'spec.json',spec);start=time.monotonic()
    import neuron
    h=neuron.h;before=Path.cwd()
    try:
        os.chdir(kernel);assert neuron.load_mechanisms('.')
        h.load_file('stdrun.hoc');assert h.load_file('GenericMuscleCell.hoc')
    finally:os.chdir(before)
    cell=h.GenericMuscleCell();seg=cell.soma(.5)
    h.cai0_ca_ion=0;h.cao0_ca_ion=2
    keep=[h.FInitializeHandler(0,lambda:cell.set_initial_v()),h.FInitializeHandler(1,lambda:cell.set_initial_ion_properties())]
    h.dt=dt;h.steps_per_ms=1/dt;h.cvode_active(0)
    peer=h.Vector(1);peer.x[0]=-60
    nmj=None;source=None;clamp=None
    if case=='main':
        duration=protocol['main']['duration_ms'];sample=protocol['main']['record_ms']
        assert sha(ROOT/'presynaptic-input.npz')==protocol['input_sha256']
        with np.load(ROOT/'presynaptic-input.npz') as z:inp=z['voltage_mV'].copy()
        nmj=h.neuron_to_neuron_exc_syn(seg);nmj.Vth=-30;nmj.conductance=.0009;nmj.weight=1
        h.setpointer(peer._ref_x[0],'vpeer',nmj)
    else:
        settings=protocol['controls'][case];duration=settings['duration_ms'];sample=dt
        for name in ['Leak','ca_boyle','k_fast','k_slow']:getattr(seg,name).gmax=0
        clamp=h.SEClamp(seg);clamp.rs=settings['SEClamp_rs_MOhm'];clamp.dur1=1e9;clamp.amp1=settings['voltage_mV']
        if case=='pool':
            cell.soma.insert('IntegrationCaInput');source=seg.IntegrationCaInput;source.commanded=-.002
    h.finitialize(-45)
    if case=='gates':
        seg.v=protocol['controls']['gates']['voltage_mV']
        for (mech,state),value in zip(GATES,protocol['controls']['gates']['initial_states']):setattr(getattr(seg,mech),state,value)
    elif case=='pool':
        seg.CaPool.concentration=protocol['controls']['pool']['initial_concentration_mM']
        seg.cai=protocol['controls']['pool']['initial_concentration_mM']
    gate_objects=[(getattr(seg,m),s) for m,s in GATES]
    def gates():return [float(getattr(obj,name)) for obj,name in gate_objects]
    def record(t,previous,last):
        ca=float(seg.cai);state=float(seg.CaPool.concentration)
        ca_pos=max(ca,0.);state_pos=max(state,0.)
        return [t,float(seg.v),float(peer.x[0]),last,float(nmj.s) if nmj else 0.,float(nmj.i) if nmj else 0.,
            float(seg.ica),state,ca,.3*ca_pos/(ca_pos+2.5e-7),.3*state_pos/(state_pos+2.5e-7),
            *gates(),float(seg.ca_boyle.e_fcond),float(seg.ca_boyle.f_fcond),float(seg.k_fast.p_fcond),float(seg.k_fast.q_fcond),float(seg.k_slow.n_fcond),
            float(seg.ca_boyle.h_fcond),float(seg.ca_boyle.h_q),float(seg.ca_boyle.h_inf),*previous,
            float(source.commanded) if source else 0.,float(source.ica) if source else 0.,float(h.iCa_CaPool)]
    count=round(duration/dt);stride=round(sample/dt);input_stride=round(dt/.00625)
    data=np.empty((count//stride+1,len(COLUMNS)));previous=gates()
    data[0]=record(0.,previous,-60.);row=1;failure=None
    for tick in range(count):
        if case=='main':peer.x[0]=float(inp[tick*input_stride])
        elif case=='pool':source.commanded=(-.002 if tick*dt<50 else 0. if tick*dt<100 else .002)
        last=float(peer.x[0])
        if (tick+1)%stride==0:previous=gates()
        h.fadvance()
        if (tick+1)%stride==0:
            # The applied interval input is retained separately before showing
            # the common endpoint input for the next interval.
            if case=='main':peer.x[0]=float(inp[(tick+1)*input_stride])
            data[row]=record((tick+1)*dt,previous,last);row+=1
        if (tick+1)%round(1000/dt)==0:print(out.name,(tick+1)*dt/1000,'s',flush=True)
    data=data[:row]
    if not np.isfinite(data).all():failure='Nonfinite saved state'
    assert abs(h.t-duration)<1e-5 and abs(h.dt-dt)<1e-15
    np.savez_compressed(out/'trace.npz',columns=np.array(COLUMNS),data=data,
        neural_clock_ms=np.array(float(h.t)),record_interval_ms=np.array(sample))
    attributes={}
    for mech,state in GATES:
        prefix=state[0];obj=getattr(seg,mech)
        attributes[f'{mech}.{state}']=dict(tau_ms=float(getattr(obj,prefix+'_tau')),
            initial=float(data[0,COLUMNS.index(state)]),inf=float(getattr(obj,prefix+'_inf')),
            power=float(getattr(obj,prefix+'_instances')))
    result=dict(spec=spec,complete=failure is None,failure=failure,duration_ms=duration,sample_ms=sample,
        rows=row,gate_parameters=attributes,area_um2=float(h.area(.5,sec=cell.soma)),
        geometry=dict(L_um=float(cell.soma.L),diam_um=float(seg.diam),cm_uF_cm2=float(cell.soma.cm)),
        pool_rho=float(seg.CaPool.rho),pool_decay_ms=float(seg.CaPool.decayConstant),pool_rest_mM=float(seg.CaPool.restingConc),
        actual_dt_ms=float(h.dt),actual_neural_clock_ms=float(h.t),trace_sha256=sha(out/'trace.npz'),
        wall_s=time.monotonic()-start,versions=dict(neuron=neuron.__version__,numpy=np.__version__,python=sys.version))
    write(out/'record.json',result)
    print(json.dumps(dict(name=out.name,rows=row,wall_s=result['wall_s'],complete=result['complete'])),flush=True)
    if failure:raise RuntimeError(failure)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['main','gates','pool'],required=True)
    p.add_argument('--variant',choices=['original','gates','pool','both'],required=True);p.add_argument('--dt',type=float,required=True)
    a=p.parse_args();run(a.case,a.variant,a.dt)
