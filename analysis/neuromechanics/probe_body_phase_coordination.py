"""Physical positive controls, explicitly supplied muscle waves, no neurons.

This establishes whether the existing body can translate when driven with
coordinated contractions. It is never counted as neural locomotion success.
"""
import hashlib
import argparse
import json
from pathlib import Path
import time

import numpy as np

from thermotaxis.neuromechanics.anatomical_body import load_anatomical_body_wiring, PlanarMuscleProjection
from thermotaxis.neuromechanics.boyle_muscle_body import BoyleMuscleBody


ROOT=Path('results/body-phase-coordination-2026-11-07')


def run(root=ROOT,worker='results/full-network-body-interface-2026-11-05/build/worker',case_names=None):
    root.mkdir(exist_ok=False)
    specs=[dict(name='zero',mode='zero'),dict(name='standing',mode='standing'),
           dict(name='head_to_tail',mode='travel',direction=1),
           dict(name='tail_to_head',mode='travel',direction=-1),
           dict(name='phase_shuffled',mode='shuffled',direction=1)]
    if case_names:specs=[s for s in specs if s['name'] in case_names.split(',')]
    duration=12.;dt=.001;frequency=.5;wavenumber=1.5;amplitude=.15
    design=dict(declared_before_execution=True,specs=specs,duration_s=duration,dt_s=dt,
        frequency_Hz=frequency,waves_per_body=wavenumber,drive_mean=amplitude,drive_amplitude=amplitude,
        neurons=False,training=False,interpretation='externally supplied 95 named muscle drives: physical positive controls only',
        metric_window_s=[6,12],seed=7,worker=str(worker))
    (root/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    w=load_anatomical_body_wiring();projection=PlanarMuscleProjection.rank_bins(w.muscle_names)
    phase=np.array([(int(n[3:])-.5)/24 for n in w.muscle_names])*2*np.pi*wavenumber
    # Shuffling longitudinal bins preserves each oscillator's amplitude/frequency.
    permutation=np.random.default_rng(7).permutation(24)
    phase_shuffled=np.array([(permutation[int(n[3:])-1]+.5)/24 for n in w.muscle_names])*2*np.pi*wavenumber
    side=np.array([1 if n[1]=='D' else -1 for n in w.muscle_names])
    for spec in specs:
        folder=root/spec['name'];folder.mkdir()
        rows=[];energy=np.zeros(95);dose=np.zeros(95);started=time.monotonic()
        with BoyleMuscleBody(design['worker']) as body:
            body.reset()
            for tick in range(round(duration/dt)+1):
                t=tick*dt
                p=(np.zeros(95) if spec['mode']=='standing' else phase_shuffled if spec['mode']=='shuffled' else phase)
                drive=amplitude*(1+side*np.sin(2*np.pi*frequency*t-spec.get('direction',1)*p))
                if spec['mode']=='zero':drive[:]=0
                command=projection.to_segments(drive)
                if tick%10==0:
                    rows.append(dict(time_s=t,pose=body.state.pose.copy(),velocity=body.state.velocity.copy(),
                        speed=body.state.signed_speed_m_s,drive=command.copy(),muscle=body.state.muscle.copy()))
                if tick==round(duration/dt):break
                energy+=drive**2*dt;dose+=drive*dt
                body.step(1,muscle_drive=command)
        out={k:np.array([r[k] for r in rows]) for k in rows[0]}
        np.savez_compressed(folder/'trace.npz',**out)
        late=out['time_s']>=6
        center=out['pose'][:,:,:2].mean(1)
        result=dict(spec=spec,complete=True,wall_s=time.monotonic()-started,
            late_signed_speed_um_s=float(out['speed'][late].mean()*1e6),
            late_net_displacement_um=float(np.linalg.norm(center[-1]-center[np.where(late)[0][0]])*1e6),
            late_centroid_path_um=float(np.linalg.norm(np.diff(center[late],axis=0),axis=1).sum()*1e6),
            muscle_drive_integral=dose.tolist(),muscle_drive_squared_integral=energy.tolist(),
            trace_sha256=hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest())
        (folder/'record.json').write_text(json.dumps(result,indent=2)+'\n')
        print(spec['name'],result['late_signed_speed_um_s'],result['late_net_displacement_um'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--worker',default='results/full-network-body-interface-2026-11-05/build/worker')
    p.add_argument('--cases')
    a=p.parse_args();run(a.root,a.worker,a.cases)
