"""Isolated B-prototype -> C1-muscle transfer; no physical body tuning."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import neuron


ROOT=Path('results/b-motor-mechanisms-2026-11-07')


def main(a):
    out=ROOT/f'nmj-{a.nS}nS';out.mkdir(exist_ok=False)
    before=Path.cwd();h=neuron.h
    try:
        os.chdir((ROOT/'kernels').resolve());assert neuron.load_mechanisms('.')
        h.load_file('stdrun.hoc');h.load_file('GenericMuscleCell.hoc')
    finally:os.chdir(before)
    p=json.loads((ROOT/'selected-parameters.json').read_text())['parameters']
    sec=h.Section();sec.L=5;sec.diam=5;sec.cm=p['capacitance_nF']*1000/(.01*h.area(.5,sec=sec))
    cell=h.BMotorPoint(sec(.5))
    for dest,src in [('gL','leak_nS'),('gCa','calcium_nS'),('gK','potassium_nS'),('EL','leak_reversal_mV'),('ECa','calcium_reversal_mV'),('EK','potassium_reversal_mV'),('midCa','calcium_midpoint_mV'),('widthCa','calcium_width_mV'),('midK','potassium_midpoint_mV'),('widthK','potassium_width_mV')]:setattr(cell,dest,p[src])
    cell.tauK=1000*p['potassium_tau_s']
    peer=h.Section();peer.L=5;peer.diam=5
    gap=h.neuron_to_neuron_elec_syn(sec(.5));gap.conductance=.0001;gap.weight=1;h.setpointer(peer(.5)._ref_v,'vpeer',gap)
    muscle=h.GenericMuscleCell()
    nmj=h.neuron_to_neuron_exc_syn(muscle.soma(.5));nmj.Vth=-30;nmj.conductance=a.nS/1000;nmj.weight=1
    h.setpointer(sec(.5)._ref_v,'vpeer',nmj)
    h.cai0_ca_ion=0;h.cao0_ca_ion=2
    def voltage_init():muscle.set_initial_v()
    def ions_init():muscle.set_initial_ion_properties()
    initializers=[h.FInitializeHandler(0,voltage_init),h.FInitializeHandler(1,ions_init)]
    h.dt=.025;h.cvode_active(0);h.finitialize(-60);peer.v=-20
    rows=[]
    for k in range(320001):
        if k%40==0:rows.append([float(h.t)/1000,sec(.5).v,muscle.soma(.5).v,muscle.soma(.5).cai,nmj.s])
        if k<320000:h.fadvance()
    values=np.array(rows);late=values[:,0]>=4
    ca=np.maximum(values[:,3],0);drive=.3*ca/(ca+2.5e-7)
    result=dict(nmj_nS=a.nS,neural_dt_ms=.025,duration_s=8,
        late_neuron_voltage_amplitude_mV=float(np.ptp(values[late,1])),
        late_muscle_voltage_amplitude_mV=float(np.ptp(values[late,2])),
        late_calcium_range_mM=[float(values[late,3].min()),float(values[late,3].max())],
        late_drive_range=[float(drive[late].min()),float(drive[late].max())],
        late_drive_sd=float(drive[late].std()))
    np.savez_compressed(out/'trace.npz',time_s=values[:,0],B_voltage_mV=values[:,1],muscle_voltage_mV=values[:,2],calcium_mM=values[:,3],gate=values[:,4],drive=drive)
    (out/'record.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--nS',type=float,default=.09);main(p.parse_args())
