"""Select a prototype only on isolated cell criteria, before body trials."""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

from thermotaxis.neuromechanics.b_motor_oscillator import BMotorParameters,rhs,sigmoid


ROOT=Path('results/b-motor-mechanisms-2026-11-07')


def main(series):
    root=ROOT/series;root.mkdir(exist_ok=False)
    if series=='isolated':
        candidates=[replace(BMotorParameters(),calcium_nS=g,potassium_tau_s=t) for g in [.2,.3,.4] for t in [.1,.25]]
    else:
        values=[.2,.3,.4] if series=='narrow-gates' else [.08,.1,.12]
        candidates=[replace(BMotorParameters(),calcium_nS=g,calcium_width_mV=5.,potassium_midpoint_mV=-15.,potassium_width_mV=5.) for g in values]
    design=dict(status='declared_before_execution',candidate_order=[p.as_dict() for p in candidates],
        criterion='last8s: no-gap voltage sd<.1mV; gap.1nS to-20mV >=5 peaks, amplitude>5mV, periodCV<.1, voltages[-100,60]mV',
        selection='first candidate satisfying all criteria, no body result inspected',duration_s=16,maximum_step_s=.005,
        source='Xu2018 Eq5 structure and2019 corrected nonlinear C3pF; other parameters engineering choices, not SI reproduction',
        links=['https://www.pnas.org/doi/10.1073/pnas.1717022115','https://www.pnas.org/doi/10.1073/pnas.1902630116'])
    (root/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    rows=[];selected=None
    for i,p in enumerate(candidates):
        results=[]
        for gap in [0.,.1]:
            t=np.arange(0,16.0001,.001)
            sol=solve_ivp(lambda t,y:rhs(t,y,p,gap_nS=gap),[0,16],[-60.,sigmoid(-60,-25,10)],
                t_eval=t,max_step=.005,rtol=1e-8,atol=1e-10)
            assert sol.success and len(sol.t)==len(t)
            late=t>=8;v=sol.y[0,late];times=t[late]
            peaks,_=find_peaks(v,prominence=5)
            periods=np.diff(times[peaks]);cv=float(periods.std()/periods.mean()) if len(periods)>1 else None
            d=dict(gap_nS=gap,late_sd_mV=float(v.std()),late_amplitude_mV=float(v.max()-v.min()),
                peaks=len(peaks),frequency_Hz=float(1/periods.mean()) if len(periods) else None,
                period_cv=cv,voltage_range_mV=[float(sol.y[0].min()),float(sol.y[0].max())])
            np.savez_compressed(root/f'candidate{i}-gap{gap}.npz',time_s=t,voltage_mV=sol.y[0],potassium_gate=sol.y[1])
            results.append(d)
        passed=(results[0]['late_sd_mV']<.1 and results[1]['peaks']>=5 and results[1]['late_amplitude_mV']>5
            and results[1]['period_cv'] is not None and results[1]['period_cv']<.1
            and all(d['voltage_range_mV'][0]>=-100 and d['voltage_range_mV'][1]<=60 for d in results))
        rows.append(dict(candidate=i,parameters=p.as_dict(),results=results,passed=passed))
        print(i,results,passed,flush=True)
        if passed and selected is None:selected=dict(candidate=i,parameters=p.as_dict(),source='first passing isolated-only candidate')
    (root/'calibration.json').write_text(json.dumps(dict(results=rows,selected=selected),indent=2)+'\n')
    if selected is not None:(ROOT/'selected-parameters.json').write_text(json.dumps(selected,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--series',choices=['isolated','narrow-gates','lower-calcium'],default='isolated')
    main(p.parse_args().series)
