"""Independent functional mechanism tasks and original-model replication.

No connectome ranking is inferred from these positive/negative controls.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from thermotaxis.sensory_adaptation import ThresholdAdaptation
from thermotaxis.motor_feedback import simulate_feedback,feedback_metrics


def main():
    root=Path('results/navigation-mechanisms-2026-10-25')
    if root.exists():raise RuntimeError('preserve existing mechanism study')
    root.mkdir(parents=True)
    sources=['analysis/navigation_mechanism_tasks.py','src/thermotaxis/sensory_adaptation.py','src/thermotaxis/motor_feedback.py']
    plan=dict(status='functional_mechanism_audits_not_connectome_advantage',
        adaptation=asdict(ThresholdAdaptation()),temperature_steps_K=[-5.,5.],step_duration_s=2400.,dt_s=.1,
        backgrounds_K=[288.15,293.15,298.15],probe_increment_K=.02,
        motor_feedback=[-1.,0.,1.],simulation_seeds=[7301,7302,7303],agents_per_seed=256,
        source_steps=1499,source_dt_s=200/1500,cooling_durations_s=[1.,7.,15.,30.],pulse_amplitude_source_units=1.,pulse_onset_s=60.,
        limitations=['AFD threshold time constants from published fits; no measured calcium-trace fit.',
            'Sensor response range is a declared working value; thermal preference is separate.',
            'Motor model is the published two-state mechanism model, not the project8 graph or SI body.',
            'Preserve source delayed indexing/noise scaling; segmentation uses raw motor sign.',
            'Matched streams across gains; source MATLAB RNG is not available.'],
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources})
    (root/'design.json').write_text(json.dumps(plan,indent=2)+'\n')
    for n in sources:
        q=root/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,q)
    sensor=ThresholdAdaptation();adapt=[];traces={}
    for step in plan['temperature_steps_K']:
        b=293.15;target=b+step;trace=[]
        for j in range(24001):
            if j%100==0:trace.append([j*.1,float(b)])
            if j<24000:b=sensor.update(target,b,.1)
        traces[str(step)]=trace
        fraction=(b-293.15)/step
        adapt.append(dict(step_K=step,adapted_fraction_after_2400s=float(fraction),final_threshold_K=float(b),
            predicted_fraction=float(1-np.exp(-2400/(45 if step>0 else 840)))))
    detection=[]
    for background in plan['backgrounds_K']:
        increment=plan['probe_increment_K']
        detection.append(dict(background_K=background,adaptive_increment_response=float(sensor.response(background+increment,background)-sensor.response(background,background)),
            fixed_increment_response=float(sensor.response(background+increment,293.15)-sensor.response(background,293.15))))
    motor=[]
    for seed in plan['simulation_seeds']:
        rng=np.random.default_rng(seed);noise=rng.normal(size=(1499,256));turns=rng.uniform(size=(1499,256));angle=rng.uniform(0,2*np.pi,256)
        for gain in plan['motor_feedback']:
            hist,heads=simulate_feedback(gain,noise,turns,angle)
            motor.append(dict(seed=seed,feedback=gain,task='source_static_gradient',metrics=feedback_metrics(hist,heads)))
            for duration in plan['cooling_durations_s']:
                h,p=simulate_feedback(gain,noise,turns,angle,pulse=(60.,duration,1.))
                before=int(60/(200/1500));after=min(1499,int((60+duration+2)/(200/1500)))
                initially_forward=h[before,:,1]>0
                ended=np.any(h[before:after,:,1]<=0,axis=0)
                motor.append(dict(seed=seed,feedback=gain,task='cooling_pulse',duration_s=duration,
                    eligible=int(initially_forward.sum()),termination_fraction=float(ended[initially_forward].mean()) if initially_forward.any() else None,
                    baseline_termination_fraction=float(np.any(hist[before:after,:,1]<=0,axis=0)[initially_forward].mean()) if initially_forward.any() else None,
                    metrics=feedback_metrics(h,p)))
    result=dict(design=plan,adaptation=adapt,pulse_detection=detection,threshold_traces=traces,motor=motor,
        source_hashes_unchanged={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in plan['source_sha256'].items()})
    assert all(result['source_hashes_unchanged'].values())
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('AFD',adapt,'detection',detection,flush=True)
    for gain in plan['motor_feedback']:
        rows=[r['metrics'] for r in motor if r['task']=='source_static_gradient' and r['feedback']==gain]
        print('MOTOR',gain,{k:float(np.mean([r[k] for r in rows])) for k in ['mean_forward_run_s','directed_forward_bias','mean_final_x_source_units']},flush=True)


if __name__=='__main__':main()
