"""Dynamics port of Ji et al. eLife2021 Fig7 public MATLAB model.

Original source/provenance: data/reference/elife68848. A/M states and spatial
units are normalized source-model units, never SI worm/body calibration.
Preserves source Euler dt=200/1500, 1499 steps, delayed indexing and noise*dt.
Run segmentation here uses raw M>0, not MATLAB's smooth/bwlabeln plotting code.
"""
import numpy as np


def simulate_feedback(feedback,noise,turn_draws,initial_angles,dt=200/1500,pulse=None):
    noise=np.asarray(noise,dtype=float);turn_draws=np.asarray(turn_draws,dtype=float)
    angles=np.asarray(initial_angles,dtype=float).copy()
    if noise.shape!=turn_draws.shape or noise.ndim!=2 or angles.shape!=(noise.shape[1],):raise ValueError('inconsistent source draws')
    if not np.isfinite(noise).all() or not np.isfinite(turn_draws).all() or np.any((turn_draws<0)|(turn_draws>1)):raise ValueError('invalid draws')
    steps,batch=noise.shape;hist=np.zeros((steps+1,batch,4));hist[0,:,:2]=1.
    headings=np.zeros((steps+1,batch));headings[0]=angles
    delay=int(np.floor(.5/dt+.5))
    logistic=lambda x:1/(1+np.exp(-np.clip(x,-700,700)))
    for j in range(1,steps+1):
        a,m,x,y=hist[j-1].T
        prev_m=hist[max(0,j-2),:,1]
        flag=(prev_m>=0).astype(int)*10+(m>=0)
        angles=np.where(flag==10,angles+np.pi,angles)
        angles=np.where((flag==10)&(angles>2*np.pi),angles-2*np.pi,angles)
        angles=np.where(flag==1,turn_draws[j-1]*2*np.pi,angles)
        lag=max(0,j-delay)
        sensory=(x-hist[lag,:,2])*(-1.5)*(m>0)
        if pulse is not None:
            onset,duration,amplitude=pulse
            sensory=sensory-amplitude*((j-1)*dt>=onset)*((j-1)*dt<onset+duration)*(m>0)
        da=1.5*(sensory+.5+2*noise[j-1]+feedback*logistic(15*m)-a)*dt
        dm=1.5*(logistic(5*(a-1.5))-.7*logistic(-5*(a-1.5))-m)*dt
        hist[j,:,0]=a+da;hist[j,:,1]=m+dm
        hist[j,:,2]=x+np.cos(angles)*dt;hist[j,:,3]=y+np.sin(angles)*dt
        headings[j]=angles
    return hist,headings


def feedback_metrics(hist,headings,dt=200/1500):
    motor=hist[:-1,:,1]>0
    runs=[]
    for i in range(motor.shape[1]):
        padded=np.pad(motor[:,i].astype(int),(1,1));starts=np.flatnonzero(np.diff(padded)==1);ends=np.flatnonzero(np.diff(padded)==-1)
        runs.extend([(end-start)*dt for start,end in zip(starts,ends)])
    return dict(mean_forward_run_s=float(np.mean(runs)) if runs else 0.,forward_fraction=float(motor.mean()),
        directed_forward_bias=float((-np.cos(headings[:-1])*motor).sum()/max(1,motor.sum())),
        mean_final_x_source_units=float(hist[-1,:,2].mean()),median_final_x_source_units=float(np.median(hist[-1,:,2])),
        forward_runs=len(runs),raw_motor_segmentation=True)
