"""Full302 Cook adaptation of modWorm's graded Python model, independently of C1.

The delayed voltage-projection loop is NOT body strain feedback. No traveling
muscle waveform or prescribed translation is supplied. Lengths are reported in
body lengths: the upstream rod's numerical units are not converted to microns.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import integrate, interpolate, linalg

from modworm_python_bridge import load_python_backend
from thermotaxis.neuromechanics.anatomical_body import load_anatomical_body_wiring


ROOT = Path('results/full302-modworm-reference-2026-11-07')


def build(a):
    nv_cls, mb_cls, _ = load_python_backend(a.root/'upstream')
    from modWorm import network_dynamics as nd, network_interactions as ni
    w = load_anatomical_body_wiring()
    n = len(w.neuron_names)
    metadata = json.loads(Path('results/full302-operating-range-2026-11-07/connection-plan.json').read_text())
    signs = {row['name']: row['polarity'] for row in metadata['polarity_table']}
    polarity = np.array([-48. if signs[name]=='inh' else 0. for name in w.neuron_names])
    # Explicit portability extension of the upstream n=279 constructor.
    nv = nv_cls.CelegansWorm_NervousSystem_PPC.__new__(nv_cls.CelegansWorm_NervousSystem_PPC)
    nv.network_Size = n
    nv.neuron_C = nd.init_neuron_C(np.full(n,.0015))
    nv.neuron_Linear = nd.init_neuron_Linear(np.full(n,.01),np.full(n,-35.))
    nv.neuron_Chemical = nd.init_neuron_Chemical(np.full(n,1/1.5),np.full(n,5/1.5),np.full(n,.125))
    gap, chem = w.neuron_gap.copy(), w.neuron_chemical.copy()
    np.fill_diagonal(gap,0); np.fill_diagonal(chem,0)
    nv.network_Electrical = ni.init_network_Electrical(gap,np.full((n,n),.1),np.ones(n,bool))
    nv.network_Chemical = ni.init_network_Chemical(chem,np.full((n,n),.1),np.repeat(polarity[:,None],n,axis=1),np.ones(n,bool))
    nv.timescale = a.exchange
    nv.fdb_init_Int = round(1.18/a.exchange)
    nv.fdb_delay_Int = round(.6/a.exchange)
    # Upstream drops LU's P; use a pivot-aware solve and record its difference.
    ni.init_vth_Linear(nv)
    g = nv.network_Electrical
    s_eq = np.round(nv.neuron_Chemical[0]/(nv.neuron_Chemical[0]+2*nv.neuron_Chemical[1]),4)
    matrix = -np.diag(nv.neuron_Linear[0])+g-np.diag(g.sum(1))-np.diag((nv.network_Chemical[0]*s_eq).sum(1))
    lu = linalg.lu_factor(matrix)
    source_vth = nv.compute_Vth(np.zeros(n))
    nv.compute_Vth = lambda current: linalg.lu_solve(lu,nv.bb_vth-current)
    pivot_delta = float(abs(nv.compute_Vth(np.zeros(n))-source_vth).max())
    muscle_map = np.zeros((96,n))
    for mi,name in enumerate(w.muscle_names):
        row = (int(name[3:])-1)*4+{'MDL':0,'MDR':1,'MVL':2,'MVR':3}[name[:3]]
        muscle_map[row] = w.nmj_chemical[:,mi]*np.where(polarity<0,-1.,1.)
    mb = mb_cls.CelegansWorm_MuscleBody_PPC(muscle_map)
    mb.body_ForceScaling *= a.force_scale
    mb.fluid_Viscosity *= a.viscosity_scale
    if a.straight:
        mb.initcond[:] = 0
    if a.initial == 'equilibrium':
        nv.initcond = np.r_[nv.compute_Vth(np.zeros(n)),s_eq]
    else:
        rng = np.random.default_rng(7)
        nv.initcond = 1e-4*rng.normal(0,.94,2*n)
    if a.feedback == 'off': nv.proprioceptive_FDB = lambda *args: np.zeros(n)
    return nv,mb,w,dict(pivot_correction_max_mV=pivot_delta,
        neurons=n,readout_muscles=95,rod_segments=24,
        directed_chemical_nonzero=int(np.count_nonzero(chem)),gap_pairs=int(np.count_nonzero(np.triu(gap,1))),
        muscle_map_rank=int(np.linalg.matrix_rank(muscle_map)),missing_MVL24_zero=bool(np.all(muscle_map[94]==0)))


def run(a):
    folder = a.root/a.name; folder.mkdir(exist_ok=False)
    spec = vars(a).copy(); spec['root']=str(spec['root'])
    (folder/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    started=time.monotonic()
    nv,mb,w,audit = build(a)
    from modWorm import body_dynamics as bd
    count=round(a.duration/a.exchange)+1
    t=np.arange(count)*a.exchange
    inp=np.zeros((count,302))
    if a.stim != 'none':
        names={'PLM':['PLML','PLMR'],'AVB':['AVBL','AVBR'],'AFD':['AFDL','AFDR']}[a.stim]
        if a.preset:
            template=np.load(a.root/'upstream/modWorm/presets_input/input_mat_gentle_post_touch.npy')[:,276]
            wave=np.interp(t-a.onset,np.arange(len(template))*.01,template,left=0,right=0)*(a.current/template.max())
        else: wave=np.where(t>=a.onset,a.current,0.)
        for name in names:inp[:,w.neuron_names.index(name)] = wave
    nv.interp_i_Ext=interpolate.interp1d(t,inp,axis=0,fill_value='extrapolate')
    voltage=np.zeros((count,302)); syn=np.zeros_like(voltage); vth=np.zeros_like(voltage)
    body=np.zeros((count,len(mb.initcond))); force=np.zeros((count,48)); feedback=np.zeros_like(voltage)
    voltage[0]=nv.initcond[:302];syn[0]=nv.initcond[302:];body[0]=mb.initcond
    rn=integrate.ode(nv.forward_Network,nv.forward_network_Jacobian).set_integrator('vode',method='bdf',rtol=a.neural_rtol,atol=a.neural_atol,nsteps=10000)
    rb=integrate.ode(mb.forward_Body).set_integrator('dopri5',rtol=a.body_rtol,atol=a.body_atol,nsteps=10000,max_step=.001)
    rn.set_initial_value(nv.initcond,0);rb.set_initial_value(mb.initcond,0)
    failure=None
    for k in range(count):
        vth[k]=nv.compute_Vth(inp[k])
        force[k]=mb.forward_Muscles(voltage[k],vth[k])
        if a.force_off: force[k]=0
        mb.fR,mb.fL=force[k,:23],force[k,24:47]
        fb=nv.proprioceptive_FDB(mb,voltage,syn,vth,force,body[:,:26],k)
        feedback[k]=fb;nv.input_FDB=fb
        if not all(np.isfinite(x).all() for x in [voltage[k],syn[k],body[k],force[k]]):
            failure='nonfinite state';break
        if k==count-1:break
        rn.integrate(t[k+1]);rb.integrate(t[k+1])
        voltage[k+1]=rn.y[:302];syn[k+1]=rn.y[302:];body[k+1]=rb.y
        if not rn.successful() or not rb.successful():
            failure=f'integrator return codes neural={rn.get_return_code()}, body={rb.get_return_code()}'
            k+=1;break
        if (k+1)%100==0:print(a.name,round(t[k+1],2),'seconds',flush=True)
    keep=slice(None,k+1)
    # Raw coordinates at segment centers; no smoothing or display extrapolation.
    xy=[]
    for row in body[keep]:
        x,y,*_=bd.fwd_body_Coordinates(mb,row[0],row[1],row[26],row[27],row[2:26],row[28:52])
        xy.append(np.stack([x,y],axis=1))
    xy=np.array(xy)
    length=float(mb.body_SegLength.sum())
    mass=mb.body_Mass/mb.body_Mass.sum()
    centroid=np.einsum('tsc,s->tc',xy,mass)
    disp=np.linalg.norm(centroid[-1]-centroid[0])/length
    v=voltage[keep]; tail=t[keep]>=max(a.onset,a.duration/2)
    result=dict(spec=spec,audit=audit,complete=failure is None,failure=failure,
        actual_duration_s=float(t[k]),wall_s=time.monotonic()-started,
        voltage_range_mV=[float(v.min()),float(v.max())],
        centroid_displacement_BL=float(disp),
        late_path_speed_BL_s=float(np.linalg.norm(np.diff(centroid[tail],axis=0),axis=1).sum()/max(a.exchange,t[keep][tail][-1]-t[keep][tail][0])/length),
        late_force_sd=float(force[keep][tail].std(0).mean()),
        late_angle_sd_rad=float(body[keep][tail,2:26].std(0).mean()),
        units='upstream numerical length normalized by total segment length; not calibrated micrometers')
    np.savez_compressed(folder/'trace.npz',time_s=t[keep],input_pA=inp[keep],voltage_mV=v,synaptic_gate=syn[keep],
        threshold_mV=vth[keep],force=force[keep],feedback_mV=feedback[keep],body_state=body[keep],xy=xy,
        centroid=centroid,length=length,neuron_names=np.array(w.neuron_names))
    result['trace_sha256']=hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest()
    (folder/'record.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)
    if failure:raise RuntimeError(failure)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--name',required=True)
    p.add_argument('--stim',choices=['PLM','AVB','AFD','none'],default='PLM')
    p.add_argument('--current',type=float,default=3000)
    p.add_argument('--preset',action='store_true');p.add_argument('--onset',type=float,default=0)
    p.add_argument('--duration',type=float,default=4);p.add_argument('--exchange',type=float,default=.01)
    p.add_argument('--feedback',choices=['delayed_voltage','off'],default='delayed_voltage')
    p.add_argument('--initial',choices=['upstream_zero','equilibrium'],default='upstream_zero')
    p.add_argument('--straight',action='store_true');p.add_argument('--force-off',action='store_true')
    p.add_argument('--force-scale',type=float,default=1);p.add_argument('--viscosity-scale',type=float,default=1)
    p.add_argument('--neural-rtol',type=float,default=5e-6);p.add_argument('--neural-atol',type=float,default=1e-3)
    p.add_argument('--body-rtol',type=float,default=2.5e-6);p.add_argument('--body-atol',type=float,default=1e-3)
    run(p.parse_args())
