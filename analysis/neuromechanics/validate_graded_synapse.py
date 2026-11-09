"""Constant presynaptic voltage vs the exact graded-synapse gate solution."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import neuron


def run(a):
    before=Path.cwd()
    try:
        os.chdir(a.kernels.resolve())
        assert neuron.load_mechanisms('.')
    finally:
        os.chdir(before)
    h=neuron.h;h.load_file('stdrun.hoc');h.dt=a.dt_ms;h.cvode_active(0)
    pre=h.Section();post=h.Section()
    syn=h.neuron_to_neuron_exc_syn(post(.5))
    syn.Vth=a.threshold;syn.weight=0
    h.setpointer(pre(.5)._ref_v,'vpeer',syn)
    h.finitialize(a.voltage)
    pre.v=a.voltage;syn.s=0
    inf=1/(1+np.exp((a.threshold-a.voltage)/5))
    tau=(1-inf)/.025
    rows=[]
    for step in range(round(a.duration_ms/a.dt_ms)+1):
        exact=inf*(1-np.exp(-float(h.t)/tau))
        rows.append([float(h.t),float(syn.s),float(exact)])
        if step<round(a.duration_ms/a.dt_ms):h.fadvance()
    values=np.array(rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output.with_suffix('.npz'),values=values)
    result=dict(kernels=str(a.kernels),voltage_mV=a.voltage,threshold_mV=a.threshold,
        dt_ms=a.dt_ms,tau_ms=float(tau),max_absolute_gate_error=float(abs(values[:,1]-values[:,2]).max()),
        gate_range=[float(values[:,1].min()),float(values[:,1].max())],
        within_probability_bounds=bool(np.all((values[:,1]>=0)&(values[:,1]<=1))))
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kernels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--voltage',type=float,default=12);p.add_argument('--threshold',type=float,default=-30)
    p.add_argument('--dt-ms',type=float,default=.025);p.add_argument('--duration-ms',type=float,default=5)
    run(p.parse_args())
