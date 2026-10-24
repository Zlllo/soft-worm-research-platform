"""Offline linear-identification checks; this is not a neural performance test.

Known local motion is granted to this analysis, never to the controllers. A
linear ambient trend is a separate model assumption from the sine-wave probe.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def positions(times,turn_rate):
    speed=.0002;offset=.0005;amplitude=.00015
    angle=turn_rate*times
    if turn_rate==0:
        x=speed*times;y=np.zeros_like(times)
    else:
        x=speed*np.sin(angle)/turn_rate;y=speed*(1-np.cos(angle))/turn_rate
    phase=np.pi*times+.2
    return x+offset*np.cos(angle)-amplitude*np.sin(phase)*np.sin(angle), y+offset*np.sin(angle)+amplitude*np.sin(phase)*np.cos(angle)


def design_stats(matrix):
    normalized=matrix.copy()
    for j in range(1,matrix.shape[1]):
        c=matrix[:,j]-matrix[:,j].mean()
        normalized[:,j]=c/np.linalg.norm(c)
    normalized[:,0]/=np.linalg.norm(normalized[:,0])
    singular=np.linalg.svd(normalized,compute_uv=False)
    rank=int(np.linalg.matrix_rank(normalized))
    full=rank==matrix.shape[1]
    uncertainty=None
    if full:
        inverse=np.linalg.pinv(matrix)
        # Measurement noise 0.002K, unconstrained linear coefficients.
        uncertainty=float(.002*np.linalg.norm(inverse[1])*40.)
    return dict(rank=rank,columns=matrix.shape[1],full_rank=full,
                normalized_condition_number=float(singular[0]/singular[-1]) if full else None,
                unconstrained_gradient_x_standard_error_K_m=uncertainty)


def main():
    out=Path('results/moving-front-identifiability-2026-10-24')
    if out.exists():raise RuntimeError('identification output exists')
    out.mkdir(parents=True)
    plan=dict(status='offline_linear_algebra_not_neural_training',windows_s=[4.,20.,60.],
        sampling_dt_s=.1,head_frequency_Hz=.5,speed_m_s=.0002,turn_rate_rad_s=[0.,.08],
        models=['unknown linear ambient trend','unknown sine/cosine coefficients, known60s period'],
        scope='Regression treats gradient components as unconstrained; condition numbers are not an impossibility theorem for networks with extra priors.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'design.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(design=plan,rows=[])
    for window in plan['windows_s']:
        t=np.arange(round(window/.1)+1)*.1
        for turn in plan['turn_rate_rad_s']:
            x,y=positions(t,turn)
            for ambient in ['linear','sine']:
                columns=[np.ones(len(t)),x*40.,y*40.]
                if ambient=='linear':columns.append(t*.008)
                else:columns.extend([.05*np.sin(2*np.pi*t/60),.05*np.cos(2*np.pi*t/60)])
                stats=design_stats(np.column_stack(columns))
                if turn==0 and ambient=='linear':assert stats['rank']==3
                else:assert stats['full_rank']
                result['rows'].append(dict(window_s=window,turn_rate_rad_s=turn,ambient=ambient,**stats))
    t=np.arange(41)*.1;x,y=positions(t,0.)
    gy=np.sqrt(40.**2-20.**2)
    first=20*x+gy*y-.004*t-.01
    second=-20*x+gy*y+.004*t+.01
    maximum=float(np.max(np.abs(first-second)))
    assert maximum<1e-14
    result['counterexample']=dict(maximum_head_temperature_difference_K=maximum,
        gradient_vectors_K_m=[[20.,float(gy)],[-20.,float(gy)]],
        ambient_linear_rates_K_s=[-.004,.004],offsets_K=[-.01,.01],
        center_temperature_errors_K=[-.01,.01],both_gradient_norms_K_m=40.,
        limit='Exact for unknown linear ambient trends and straight motion; not asserted for fixed-phase sine probes.')
    shutil.copyfile(__file__,out/'source.py')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
