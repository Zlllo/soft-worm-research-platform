"""Declared posthoc current-order probe; no feedback or parameter refitting."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from run_full302_closed_loop import check, PARENT, DEFAULT
from run_full302_c1 import sha,write
from thermotaxis.neuromechanics.full302_c1 import C1Network
from thermotaxis.neuromechanics.anatomical_body import load_anatomical_body_wiring,PlanarMuscleProjection
from thermotaxis.neuromechanics.boyle_muscle_body import BoyleMuscleBody


def prepare(root):
    check(root);folder=root/'temporal-probe';folder.mkdir(exist_ok=False)
    source=root/'warm_live/trace.npz'
    with np.load(source) as z:slow=z['AFD_current_pA'].copy();times=z['exchange_time_s'].copy()
    early=int(np.flatnonzero(slow>0)[0]);front=slow.copy()
    front[early:-1]=np.sort(slow[early:-1])[::-1]
    assert np.array_equal(np.sort(slow[:-1]),np.sort(front[:-1]))
    np.savez_compressed(folder/'profiles.npz',time_s=times,slow=slow,frontloaded=front)
    write(folder/'design.json',dict(status='declared_before_temporal_probe_runs',posthoc=True,
        reason='slow warming reached near5pA yet generated negligible muscle activation; test temporal ordering',
        source_sha256=sha(source),profiles_sha256=sha(folder/'profiles.npz'),
        script_sha256=sha(Path(__file__)),dt_ms=.025,exchange_ms=1.,duration_s=2.,
        intervention='descending permutation of all command intervals after first positive sample; same values/histogram, peak, integral and RMS',
        model='same full302 C1 and calcium-to-body map, no mechanical feedback or other injected currents',
        scope='offline current replay diagnostic, not a new sensory model or navigation policy'))


def run(root,name):
    check(root);folder=root/'temporal-probe';d=json.loads((folder/'design.json').read_text())
    assert sha(Path(__file__))==d['script_sha256'] and sha(folder/'profiles.npz')==d['profiles_sha256']
    out=folder/name;out.mkdir(exist_ok=False)
    with np.load(folder/'profiles.npz') as z:profile=z[name].copy()
    plan=json.loads((PARENT/'connection-plan.json').read_text())
    started=time.monotonic();network=C1Network(PARENT/'kernels',plan,dt_ms=.025)
    clamps=[network.add_current(n,start_ms=0,duration_ms=1e9,amplitude_pA=0) for n in ['AFDL','AFDR']]
    network.reset();w=load_anatomical_body_wiring();projection=PlanarMuscleProjection.rank_bins(w.muscle_names)
    times=[];volt=[];calcium=[];poses=[];commands=[]
    with BoyleMuscleBody('results/full-network-body-interface-2026-11-05/build/worker') as body:
        body.reset()
        for tick in range(2001):
            state=network.state()
            if tick%10==0:
                times.append(tick*.001);volt.append(state['neuron_voltage_mV'])
                calcium.append(state['muscle_calcium_mM']);poses.append(body.state.pose.copy())
            if tick==2000:break
            for clamp in clamps:clamp.amp=float(profile[tick])/1000
            c=np.maximum(state['muscle_calcium_mM'],0)
            drive=projection.to_segments(.3*c/(c+2.5e-7))
            if tick%10==0:commands.append(drive)
            body.step(1,muscle_drive=drive);network.advance(1.)
            assert abs(body.state.time_s-float(network.h.t)/1000)<1e-10
    values=dict(time_s=np.array(times),neuron_voltage_mV=np.stack(volt),muscle_calcium_mM=np.stack(calcium),
        pose=np.stack(poses),commanded_drive=np.stack(commands),AFD_current_pA=profile)
    assert all(np.isfinite(v).all() for v in values.values())
    np.savez_compressed(out/'trace.npz',**values)
    write(out/'record.json',dict(name=name,neurons=302,muscles=95,complete=True,neural_fits=0,
        wall_s=time.monotonic()-started,trace_sha256=sha(out/'trace.npz'),
        current_integral_pA_s=float(profile[:-1].sum()*.001),peak_current_pA=float(profile.max())))
    check(root);print('Completed temporal probe',name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--name',choices=['slow','frontloaded']);a=p.parse_args()
    if a.stage=='prepare':prepare(a.root)
    else:run(a.root,a.name)
