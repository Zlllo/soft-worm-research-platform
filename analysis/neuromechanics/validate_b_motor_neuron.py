"""Independent NEURON/SciPy comparison before full302 body trials."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import neuron

from thermotaxis.neuromechanics.b_motor_oscillator import BMotorParameters,rhs,sigmoid


ROOT=Path('results/b-motor-mechanisms-2026-11-07')


def main(a):
    out=ROOT/f'component-dt{a.dt}';out.mkdir(exist_ok=False)
    before=Path.cwd()
    try:
        os.chdir((ROOT/'kernels').resolve());assert neuron.load_mechanisms('.')
    finally:os.chdir(before)
    h=neuron.h;h.load_file('stdrun.hoc')
    p=BMotorParameters(**json.loads((ROOT/'selected-parameters.json').read_text())['parameters'])
    sec=h.Section();sec.L=5;sec.diam=5;sec.cm=p.capacitance_nF*1000/(.01*h.area(.5,sec=sec))
    cell=h.BMotorPoint(sec(.5))
    for target,source in [('gL','leak_nS'),('gCa','calcium_nS'),('gK','potassium_nS'),('EL','leak_reversal_mV'),('ECa','calcium_reversal_mV'),('EK','potassium_reversal_mV'),('midCa','calcium_midpoint_mV'),('widthCa','calcium_width_mV'),('midK','potassium_midpoint_mV'),('widthK','potassium_width_mV')]:setattr(cell,target,getattr(p,source))
    cell.tauK=p.potassium_tau_s*1000
    peer=h.Section();peer.L=5;peer.diam=5
    gap=h.neuron_to_neuron_elec_syn(sec(.5));gap.conductance=.0001;gap.weight=1
    h.setpointer(peer(.5)._ref_v,'vpeer',gap)
    h.dt=a.dt;h.cvode_active(0);h.finitialize(-60);peer.v=-20
    rows=[]
    for k in range(round(8000/a.dt)+1):
        if k%round(1/a.dt)==0:rows.append([float(h.t)/1000,sec(.5).v,cell.n])
        if k<round(8000/a.dt):h.fadvance()
    values=np.array(rows);t=values[:,0]
    ref=solve_ivp(lambda t,y:rhs(t,y,p),[0,8.001],[-60,sigmoid(-60,p.potassium_midpoint_mV,p.potassium_width_mV)],t_eval=t,max_step=.001,rtol=1e-10,atol=1e-12)
    peaks,_=find_peaks(values[t>=4,1],prominence=5)
    tr=t[t>=4][peaks];freq=float(1/np.diff(tr).mean())
    refpeaks,_=find_peaks(ref.y[0,t>=4],prominence=5);trref=t[t>=4][refpeaks]
    result=dict(dt_ms=a.dt,max_voltage_error_mV=float(abs(values[:,1]-ref.y[0]).max()),
        voltage_RMS_error_mV=float(np.sqrt(np.mean((values[:,1]-ref.y[0])**2))),frequency_Hz=freq,
        reference_frequency_Hz=float(1/np.diff(trref).mean()),neuron_capacitance_pF=float(sec.cm*.01*h.area(.5,sec=sec)))
    np.savez_compressed(out/'trace.npz',time_s=t,voltage_mV=values[:,1],gate=values[:,2],reference_voltage_mV=ref.y[0],reference_gate=ref.y[1])
    (out/'record.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    assert result['max_voltage_error_mV']<.2 and abs(result['frequency_Hz']/result['reference_frequency_Hz']-1)<.01


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dt',type=float,default=.025);main(p.parse_args())
