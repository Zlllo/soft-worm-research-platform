"""302-cell heterogeneous prototype with actual body strain feedback.

Replace only18 B-cell intrinsic currents; preserve all anatomical connections,
all other284 neuronal C1 dynamics and95 C1 muscle cells. No muscle waveform or
phase clock is supplied. AVB voltage clamp is an explicit descending intervention.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from thermotaxis.neuromechanics.full302_c1 import C1Network
from thermotaxis.neuromechanics.anatomical_body import load_anatomical_body_wiring,PlanarMuscleProjection
from thermotaxis.neuromechanics.boyle_muscle_body import BoyleMuscleBody


ROOT=Path('results/b-motor-mechanisms-2026-11-07')


def configure(network,a):
    h=network.h;plan=network.plan
    p=json.loads((ROOT/'selected-parameters.json').read_text())['parameters']
    names=tuple(n for n in plan['neurons'] if n.startswith(('DB','VB')) and n[2:].isdigit())
    assert len(names)==18
    points=[]
    for name in names:
        sec=network.cells[name].soma
        sec.cm=p['capacitance_nF']*1000/(.01*h.area(.5,sec=sec))
        if a.model=='prototype':
            seg=sec(.5)
            for mech in ['Leak','ca_boyle','k_fast','k_slow']:getattr(seg,mech).gmax=0
            point=h.BMotorPoint(seg)
            for dest,src in [('gL','leak_nS'),('gCa','calcium_nS'),('gK','potassium_nS'),('EL','leak_reversal_mV'),('ECa','calcium_reversal_mV'),('EK','potassium_reversal_mV'),('midCa','calcium_midpoint_mV'),('widthCa','calcium_width_mV'),('midK','potassium_midpoint_mV'),('widthK','potassium_width_mV')]:setattr(point,dest,p[src])
            point.gCa*=a.calcium_scale
            point.tauK=p['potassium_tau_s']*1000;points.append(point)
    avb={'AVBL','AVBR'}
    partners={n:sum((n in pair and bool(avb.intersection(pair))) for pair in plan['gap_pairs']) for n in names}
    changed=[]
    for pair,syns in zip(plan['gap_pairs'],network.gap_connections):
        b=next((n for n in pair if n in names),None)
        if b is not None and avb.intersection(pair):
            for syn in syns:syn.conductance=a.gap_nS/1000/partners[b]
            changed.append(dict(pair=pair,conductance_nS=a.gap_nS/partners[b]))
    # Localize the release-range intervention to B-cell outputs only.
    for (pre,post),syn in network.chemical_connections.items():
        if pre in names:
            syn.Vth=-30
            if post in plan['muscles']:syn.conductance=a.nmj_nS/1000
    rng=np.random.default_rng(a.seed)
    initial=-60+rng.normal(0,.2,18)
    def initialize():
        for n,v in zip(names,initial):network.cells[n].soma.v=float(v)
    network.initializers.append(h.FInitializeHandler(0,initialize))
    clamps=[]
    for name in avb:
        clamp=h.SEClamp(network.cells[name].soma(.5))
        clamp.rs=1e-4;clamp.dur1=500;clamp.amp1=-60
        clamp.dur2=1e9;clamp.amp2=-20 if a.avb_on else -60
        clamps.append(clamp)
    afferents=[network.add_current(n,start_ms=0,duration_ms=1e9,amplitude_pA=0) for n in names]
    return names,points,clamps,afferents,changed,initial


def field(w,names):
    projection=PlanarMuscleProjection.rank_bins(w.muscle_names)
    weights=np.zeros((18,48));sides=np.array([0 if n.startswith('DB') else 1 for n in names]);centers=[]
    for i,(name,side) in enumerate(zip(names,sides)):
        footprint=((w.nmj_chemical[w.neuron_names.index(name)]>0)@projection.matrix).reshape(48,2)[:,side]
        center=float(np.arange(48)@footprint/footprint.sum());centers.append(center)
        end=max(1,int(np.floor(center)));start=max(0,end-10);weights[i,start:end]=1/(end-start)
    return weights,sides,np.array(centers)


def run(a):
    out=ROOT/a.name;out.mkdir(exist_ok=False)
    spec=vars(a).copy();(out/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    started=time.monotonic()
    plan=json.loads((ROOT/'connection-plan.json').read_text())
    network=C1Network(ROOT/'kernels',plan,dt_ms=a.dt_ms)
    names,points,clamps,afferents,gaps,initial=configure(network,a)
    network.reset();w=load_anatomical_body_wiring();projection=PlanarMuscleProjection.rank_bins(w.muscle_names)
    weights,sides,centers=field(w,names)
    replay=None
    if a.replay:
        record=json.loads((ROOT/a.replay/'record.json').read_text())
        assert record['complete'] and record['spec']['duration']==a.duration
        assert record['trace_sha256']==hashlib.sha256((ROOT/a.replay/'trace.npz').read_bytes()).hexdigest()
        with np.load(ROOT/a.replay/'trace.npz') as z:replay=z['feedback_pA'].copy()
        assert replay.shape==(round(a.duration*1000)+1,18)
    rows=[];feedback_rows=[];failure=None;max_gap_imbalance=0.
    count=round(a.duration*1000)
    with BoyleMuscleBody('results/full-network-body-interface-2026-11-05/build/worker') as body:
        body.reset();rest=body.state.side_length.copy()
        for tick in range(count+1):
            t=tick/1000;s=network.state()
            strain=(body.state.side_length-rest)/rest
            bend=(strain[:,1]-strain[:,0])/2
            sensed=np.where(sides==0,1.,-1.)*(weights@bend)
            candidate=np.clip(a.feedback_gain*sensed,-a.feedback_clip,a.feedback_clip)
            actual=(replay[tick] if replay is not None else candidate if a.feedback=='live' else np.zeros(18))
            for clamp,value in zip(afferents,actual):clamp.amp=float(value)/1000
            feedback_rows.append(actual.copy())
            calcium=np.maximum(s['muscle_calcium_mM'],0)
            drive=projection.to_segments(.3*calcium/(calcium+2.5e-7))
            external=np.zeros((48,2))
            if a.perturb and 2.-1e-12<=t<2.3-1e-12:external[:8,0]=.2
            command=np.clip(drive+external,-1,1)
            invalid=s['neuron_voltage_mV'].min()<-100 or s['neuron_voltage_mV'].max()>60
            if tick%10==0 or invalid:
                rows.append(dict(**s,pose=body.state.pose.copy(),drive=drive.copy(),external_drive=external.copy(),speed=body.state.signed_speed_m_s,
                    body_muscle=body.state.muscle.copy(),side_length=body.state.side_length.copy(),sensed_strain=sensed,
                    B_gate=np.array([p.n for p in points]) if points else np.zeros(18)))
                max_gap_imbalance=max(max_gap_imbalance,max(abs(float(x.i+y.i)) for x,y in network.gap_connections))
            if invalid:failure='neuron voltage outside[-100,60]mV';break
            if tick==count:break
            body.step(1,muscle_drive=command);network.advance(1.)
            assert abs(body.state.time_s-float(network.h.t)/1000)<1e-9
            if (tick+1)%1000==0:print(a.name,(tick+1)/1000,'s',flush=True)
    values={k:np.array([r[k] for r in rows]) for k in rows[0]}
    values['exchange_time_s']=np.arange(len(feedback_rows))*.001;values['feedback_pA']=np.array(feedback_rows)
    values['B_names']=np.array(names);values['neuron_names']=np.array(w.neuron_names)
    values['feedback_weights']=weights;values['feedback_sides']=sides;values['nominal_centers']=centers
    np.savez_compressed(out/'trace.npz',**values)
    late=values['time_s']>=max(1.,a.duration/2)-1e-8
    bindex=[plan['neurons'].index(n) for n in names]
    result=dict(spec=spec,complete=failure is None,failure=failure,actual_duration_s=float(values['time_s'][-1]),
        neurons=302,muscles=95,chemical_connections=len(network.chemical_connections),gap_pairs=len(network.gap_connections),
        B_names=names,modified_AVB_gap_pairs=gaps,B_initial_voltage_mV=initial.tolist(),
        max_gap_pair_current_imbalance_nA=max_gap_imbalance,
        neuron_voltage_range_mV=[float(values['neuron_voltage_mV'].min()),float(values['neuron_voltage_mV'].max())],
        late_B_voltage_sd_mV=np.std(values['neuron_voltage_mV'][late][:,bindex],axis=0).tolist(),
        late_mean_speed_um_s=float(values['speed'][late].mean()*1e6),
        late_centroid_displacement_um=float(np.linalg.norm(values['pose'][-1,:,:2].mean(0)-values['pose'][np.flatnonzero(late)[0],:,:2].mean(0))*1e6),
        maximum_drive=float(values['drive'].max()),late_drive_sd=float(values['drive'][late].std(0).mean()),
        maximum_feedback_pA=float(abs(values['feedback_pA']).max()),wall_s=time.monotonic()-started,
        trace_sha256=hashlib.sha256((out/'trace.npz').read_bytes()).hexdigest())
    (out/'record.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True)
    if failure:raise RuntimeError(failure)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--model',choices=['prototype','generic_C1'],default='prototype')
    p.add_argument('--feedback',choices=['off','live'],default='off');p.add_argument('--replay')
    p.add_argument('--gap-nS',dest='gap_nS',type=float,default=.1)
    p.add_argument('--feedback-gain',type=float,default=100);p.add_argument('--feedback-clip',type=float,default=1)
    p.add_argument('--duration',type=float,default=4);p.add_argument('--dt-ms',type=float,default=.025)
    p.add_argument('--seed',type=int,default=13);p.add_argument('--avb-on',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--calcium-scale',type=float,default=1)
    p.add_argument('--perturb',action='store_true')
    p.add_argument('--nmj-nS',dest='nmj_nS',type=float,default=.09)
    run(p.parse_args())
