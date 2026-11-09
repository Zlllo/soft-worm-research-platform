"""Independent fixed-point stability and time-domain oscillator checks."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.signal import find_peaks

from thermotaxis.neuromechanics.b_motor_oscillator import BMotorParameters,rhs,sigmoid


ROOT=Path('results/b-motor-mechanisms-2026-11-07')


def main():
    folder=ROOT/'bifurcation';folder.mkdir(exist_ok=False)
    p=BMotorParameters(**json.loads((ROOT/'selected-parameters.json').read_text())['parameters'])
    rows=[]
    for gap in [0,.025,.05,.075,.1,.125,.15,.2]:
        def scalar(v):return rhs(0,[v,sigmoid(v,p.potassium_midpoint_mV,p.potassium_width_mV)],p,gap_nS=gap)[0]
        grid=np.linspace(-100,60,1601);roots=[]
        for lo,hi in zip(grid[:-1],grid[1:]):
            if scalar(lo)*scalar(hi)>=0:continue
            v=brentq(scalar,lo,hi);m=sigmoid(v,p.calcium_midpoint_mV,p.calcium_width_mV);n=sigmoid(v,p.potassium_midpoint_mV,p.potassium_width_mV)
            dm=m*(1-m)/p.calcium_width_mV;dn=n*(1-n)/p.potassium_width_mV
            jac=np.array([[(-p.leak_nS-p.calcium_nS*(dm*(v-p.calcium_reversal_mV)+m)-p.potassium_nS*n-gap)/p.capacitance_nF,-p.potassium_nS*(v-p.potassium_reversal_mV)/p.capacitance_nF],[dn/p.potassium_tau_s,-1/p.potassium_tau_s]])
            eig=np.linalg.eigvals(jac)
            roots.append(dict(voltage_mV=float(v),eigenvalue_real_per_s=eig.real.tolist(),eigenvalue_imag_per_s=eig.imag.tolist(),stable=bool(eig.real.max()<0)))
        t=np.arange(0,16.0001,.001)
        sol=solve_ivp(lambda t,y:rhs(t,y,p,gap_nS=gap),[0,16],[-60,sigmoid(-60,p.potassium_midpoint_mV,p.potassium_width_mV)],t_eval=t,max_step=.005,rtol=1e-8,atol=1e-10)
        assert sol.success;late=t>=8;peaks,_=find_peaks(sol.y[0,late],prominence=5)
        times=t[late][peaks]
        result=dict(gap_nS=gap,fixed_points=roots,late_amplitude_mV=float(np.ptp(sol.y[0,late])),peaks=len(peaks),
            frequency_Hz=float(1/np.diff(times).mean()) if len(peaks)>1 else None)
        rows.append(result);np.savez_compressed(folder/f'gap{gap}.npz',time_s=t,voltage_mV=sol.y[0],gate=sol.y[1]);print(result,flush=True)
    (folder/'results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
