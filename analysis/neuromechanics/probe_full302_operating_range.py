"""Controlled C1 release-range intervention; keep the original C1 model frozen.

Threshold changes are sensitivity experiments, not fitted biological parameters.
Every run starts in a fresh process. Full 302 neurons, 95 muscle cells and the
existing external-muscle Boyle worker are retained without imposed body motion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from thermotaxis.neuromechanics.full302_c1 import C1Network
from thermotaxis.neuromechanics.anatomical_body import load_anatomical_body_wiring, PlanarMuscleProjection
from thermotaxis.neuromechanics.boyle_muscle_body import BoyleMuscleBody


ROOT = Path('results/full302-operating-range-2026-11-07')


def run(a):
    folder = a.root / a.name
    folder.mkdir(exist_ok=False)
    spec = vars(a).copy(); spec['root'] = str(spec['root'])
    (folder/'spec.json').write_text(json.dumps(spec, indent=2)+'\n')
    plan = json.loads((a.root/'connection-plan.json').read_text())
    network = C1Network(a.root/a.kernels, plan, dt_ms=a.dt_ms)
    for (pre, post), syn in network.chemical_connections.items():
        syn.Vth = a.nmj_threshold if post in plan['muscles'] else a.nn_threshold
    if a.stim != 'none':
        cells = {'AFD': ['AFDL','AFDR'], 'AVB': ['AVBL','AVBR'],
                 'AS2': ['AS2'], 'PLM': ['PLML','PLMR']}[a.stim]
        for name in cells:
            network.add_current(name, start_ms=200, duration_ms=1000*a.duration-.2*1000,
                                amplitude_pA=a.current)
    network.reset()
    w = load_anatomical_body_wiring()
    projection = PlanarMuscleProjection.rank_bins(w.muscle_names)
    rows = []; started = time.monotonic(); failure = None
    with BoyleMuscleBody('results/full-network-body-interface-2026-11-05/build/worker') as body:
        body.reset()
        for tick in range(round(a.duration*1000)+1):
            state = network.state()
            c = np.maximum(state['muscle_calcium_mM'], 0)
            command = projection.to_segments(.3*c/(c+2.5e-7))
            invalid = (not all(np.isfinite(v).all() for v in state.values()) or
                state['neuron_voltage_mV'].min() < -100 or state['neuron_voltage_mV'].max() > 60)
            if tick % 10 == 0 or invalid:
                rows.append(dict(**state, pose=body.state.pose.copy(),
                    drive=command, speed=body.state.signed_speed_m_s,
                    syn_open=np.array([float(s.s) for s in network.chemical_connections.values()])))
            if invalid:
                failure = 'nonfinite or voltage outside prespecified [-100,60] mV guard'
                break
            if tick == round(a.duration*1000): break
            body.step(1, muscle_drive=command)
            network.advance(1.)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(folder/'trace.npz', **out)
    last = out['time_s'] >= out['time_s'][-1]/2
    nn = [i for i,(pre,post) in enumerate(network.chemical_connections) if post in plan['neurons']]
    nm = [i for i,(pre,post) in enumerate(network.chemical_connections) if post in plan['muscles']]
    result = dict(spec=spec, complete=failure is None, failure=failure,
        actual_duration_s=float(out['time_s'][-1]), wall_s=time.monotonic()-started,
        neurons=302, muscles=95, neuron_range_mV=[float(out['neuron_voltage_mV'].min()),float(out['neuron_voltage_mV'].max())],
        muscle_range_mV=[float(out['muscle_voltage_mV'].min()),float(out['muscle_voltage_mV'].max())],
        max_calcium_mM=float(out['muscle_calcium_mM'].max()), max_drive=float(out['drive'].max()),
        last_half_mean_nn_release=float(out['syn_open'][last][:,nn].mean()),
        last_half_mean_nmj_release=float(out['syn_open'][last][:,nm].mean()),
        last_half_signed_speed_um_s=float(out['speed'][last].mean()*1e6),
        centroid_displacement_um=float(np.linalg.norm(out['pose'][-1,:,:2].mean(0)-out['pose'][0,:,:2].mean(0))*1e6),
        trace_sha256=hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest())
    (folder/'record.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)
    if failure: raise RuntimeError(failure)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--name',required=True)
    p.add_argument('--nn-threshold',type=float,default=0)
    p.add_argument('--nmj-threshold',type=float,default=0)
    p.add_argument('--stim',choices=['none','AFD','AVB','AS2','PLM'],default='none')
    p.add_argument('--current',type=float,default=5)
    p.add_argument('--duration',type=float,default=2)
    p.add_argument('--dt-ms',type=float,default=.025)
    p.add_argument('--kernels',default='kernels')
    run(p.parse_args())
