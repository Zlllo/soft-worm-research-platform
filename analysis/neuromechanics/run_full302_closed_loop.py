"""Bounded full302 thermal/mechanical coupling and motor-capability probes."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from run_full302_c1 import check as check_parent, sha, write, REPO
from thermotaxis.neuromechanics.full302_c1 import C1Network
from thermotaxis.neuromechanics.full302_feedback import LocalAFD,BendingFeedback
from thermotaxis.neuromechanics.anatomical_body import load_anatomical_body_wiring,PlanarMuscleProjection
from thermotaxis.neuromechanics.boyle_muscle_body import BoyleMuscleBody


PARENT=Path('results/full302-c1-dynamics-2026-11-05')
DEFAULT=Path('results/full302-closed-loop-2026-11-05')


def specifications():
    rows=[]
    # Explicit dict construction allows each override to replace a default.
    def spec(name,**kw):
        row=dict(name=name,duration_s=2.,thermal='flat',feedback='off',noise_K=0.,
            frozen_head=False,bend=None,motor_pA=0.,replay=None)
        row.update(kw);rows.append(row)
    spec('flat_off')
    spec('warm_off',thermal='warming')
    spec('warm_live',thermal='warming',feedback='live')
    spec('warm_replay',thermal='warming',feedback='replay',replay='warm_live')
    spec('warm_frozen_head',thermal='warming',feedback='live',frozen_head=True)
    spec('warm_noisy',thermal='warming',feedback='live',noise_K=.001)
    spec('bend_D_off',duration_s=3.,bend='D')
    spec('bend_D_live',duration_s=3.,bend='D',feedback='live')
    spec('bend_D_replay',duration_s=3.,bend='D',feedback='replay',replay='bend_D_live')
    spec('bend_V_live',duration_s=3.,bend='V',feedback='live')
    spec('bend_V_replay_D',duration_s=3.,bend='V',feedback='replay',replay='bend_D_live')
    spec('motor_AVB_off',duration_s=4.,motor_pA=5.)
    spec('motor_AVB_live',duration_s=4.,motor_pA=5.,feedback='live')
    return rows


def prepare(root):
    if root.exists():raise FileExistsError('preserve previous results; choose a new root')
    check_parent(PARENT);root.mkdir(parents=True)
    feedback=BendingFeedback.anatomical_bins()
    write(root/'feedback-map.json',dict(names=list(feedback.names),sides=feedback.sides.tolist(),
        weights=feedback.weights.tolist(),nominal_output_centers=feedback.nominal_output_centers.tolist(),
        scope='ipsilateral NMJ footprint used only to choose a nominal anterior ten-segment field, not measured receptor anatomy'))
    sources=['analysis/neuromechanics/run_full302_closed_loop.py',
        'src/thermotaxis/neuromechanics/full302_feedback.py','tests/test_full302_feedback.py',
        'src/thermotaxis/sensory_adaptation.py','src/thermotaxis/neuromechanics/full302_c1.py',
        'src/thermotaxis/neuromechanics/boyle_muscle_body.py','src/thermotaxis/neuromechanics/boyle_body.py',
        'src/thermotaxis/neuromechanics/boyle_output.py','src/thermotaxis/neuromechanics/anatomical_body.py']
    inputs=[PARENT/'design.json',PARENT/'connection-plan.json',PARENT/'component-pair.json',PARENT/'kernel-build.json',
        Path('results/full-network-body-interface-2026-11-05/build/build.json'),
        Path('results/full-network-body-interface-2026-11-05/build/worker')]
    inputs += [p for p in (PARENT/'kernels').rglob('*') if p.is_file() and p.suffix in ('.mod','.hoc','.so','.dylib')]
    d=dict(status='declared_before_all_closed_loop_runs',parent=str(PARENT),specs=specifications(),
        neurons=302,muscles=95,neural_dt_ms=.025,exchange_ms=1.,save_ms=10.,
        temperature=dict(reference_K=293.15,gradient_K_m=[0.,40.],warming_K=.05,on_s=.2,off_s=1.2,
            sensor='actual head position in environment, plus registered temporal warming; controller sees scalar temperature only',
            initial_threshold='clean initial local temperature',warming_adaptation_tau_s=45.,cooling_adaptation_tau_s=840.,
            filter_tau_s=.1,response_width_K=.05,maximum_AFD_current_pA=5.,noise_seed=7,
            uncalibrated='width, lowpass, gain are engineering values; not fitted AFD physiology or cultivation preference'),
        mechanical=dict(targets=list(feedback.names),gain_pA_per_strain=500.,clip_pA=5.,
            signal='ipsilateral shortening contrast: (opposite_strain-own_strain)/2, normalized by accepted initial relaxed length',
            field='ten segments immediately anterior to nominal ipsilateral NMJ-output center; truncated near head',
            assumptions='field, projection and gain explicit; no corollary-discharge or mechanosensitive channel claim'),
        muscle_mapping='same calcium->drive formula and planar projection as parent',
        bend_intervention=dict(drive=.2,segments=list(range(8)),on_s=.2,off_s=.6,
            scope='external unilateral muscle pulse, not network-generated movement'),
        motor_probe=dict(cells=['AVBL','AVBR'],amplitude_pA=5.,
            scope='declared tonic command-cell injection, not hidden drive or autonomous navigation',
            gate='last2s mean signed forward speed>=10um/s and forward progression>=20um; no post-selection'),
        replay='past completed reference currents at exactly matching exchange times; no future target state',
        train=False,source_sha256={p:sha(REPO/p) for p in sources},
        inputs_sha256={str(p):sha(REPO/p) for p in inputs})
    write(root/'design.json',d)


def check(root):
    d=json.loads((root/'design.json').read_text());check_parent(PARENT)
    for p,h in {**d['source_sha256'],**d['inputs_sha256']}.items():
        if sha(REPO/p)!=h:raise RuntimeError('changed source/input: '+p)
    return d


def run(root,name,*,neural_dt_ms=None,exchange_ms=None,suffix=''):
    d=check(root);spec=next(s for s in d['specs'] if s['name']==name)
    dt=d['neural_dt_ms'] if neural_dt_ms is None else neural_dt_ms
    exchange=d['exchange_ms'] if exchange_ms is None else exchange_ms
    count=round(1000*spec['duration_s']/exchange)
    if abs(count*exchange-1000*spec['duration_s'])>1e-10 or exchange not in (1.,2.):
        raise ValueError('use registered 1/2ms exchange on native 1ms body ticks')
    folder=root/(name+suffix);folder.mkdir(exist_ok=False)
    replay=None
    if spec['replay']:
        if exchange!=1.:raise ValueError('replay reference uses1ms exchange')
        reference=root/spec['replay'];record=json.loads((reference/'record.json').read_text())
        assert record['complete'] and sha(reference/'trace.npz')==record['trace_sha256']
        with np.load(reference/'trace.npz') as z:replay=z['feedback_current_pA'].copy()
        assert len(replay)==count+1
    plan=json.loads((PARENT/'connection-plan.json').read_text())
    started=time.monotonic();network=C1Network(PARENT/'kernels',plan,dt_ms=dt)
    thermal=[network.add_current(n,start_ms=0,duration_ms=1e9,amplitude_pA=0) for n in ['AFDL','AFDR']]
    mapping=BendingFeedback.anatomical_bins()
    afferents=[network.add_current(n,start_ms=0,duration_ms=1e9,amplitude_pA=0) for n in mapping.names]
    if spec['motor_pA']:
        for n in ['AVBL','AVBR']:
            network.add_current(n,start_ms=0,duration_ms=1e9,amplitude_pA=spec['motor_pA'])
    network.reset();w=load_anatomical_body_wiring();projection=PlanarMuscleProjection.rank_bins(w.muscle_names)
    rng=np.random.default_rng(7);all_rows=[];saved=[];closure_error=0.;max_geometry_error=0.
    from thermotaxis.neuromechanics.boyle_output import side_lengths
    with BoyleMuscleBody('results/full-network-body-interface-2026-11-05/build/worker') as body:
        state=body.reset();rest=state.side_length.copy();initial_head=state.head_xy_m.copy()
        sensor=LocalAFD(293.15+np.array(d['temperature']['gradient_K_m'])@initial_head)
        for tick in range(count+1):
            time_s=tick*exchange/1000
            assert abs(state.time_s-time_s)<1e-10 and abs(float(network.h.t)/1000-time_s)<1e-10
            nstate=network.state()
            warming=.05 if spec['thermal']=='warming' and .2-1e-12<=time_s<1.2-1e-12 else 0.
            gradient=np.array(d['temperature']['gradient_K_m']) if spec['thermal']=='warming' else np.zeros(2)
            actual_temperature=293.15+gradient@state.head_xy_m+warming
            sensor_position=initial_head if spec['frozen_head'] else state.head_xy_m
            received_temperature=293.15+gradient@sensor_position+warming
            noise=float(rng.normal(0,spec['noise_K']))
            received_temperature+=noise
            if tick==0:temperature_current=0.
            else:temperature_current=sensor.advance(float(received_temperature),exchange/1000)
            actual_feedback=mapping.currents(state.side_length,rest)
            feedback=(actual_feedback if spec['feedback']=='live' else
                      replay[tick] if spec['feedback']=='replay' else np.zeros(18))
            for stim in thermal:stim.amp=temperature_current/1000
            for stim,current in zip(afferents,feedback):stim.amp=float(current)/1000
            c=np.maximum(nstate['muscle_calcium_mM'],0)
            neural_command=projection.to_segments(.3*c/(c+2.5e-7))
            external=np.zeros((48,2))
            if spec['bend'] and .2-1e-12<=time_s<.6-1e-12:
                external[:8,0 if spec['bend']=='D' else 1]=.2
            command=np.clip(neural_command+external,-1,1)
            all_rows.append(dict(time_s=time_s,head=state.head_xy_m.copy(),temperature=actual_temperature,
                received=received_temperature,noise=noise,afd=temperature_current,threshold=sensor.threshold_K,
                filtered=sensor.filtered_K,feedback=feedback.copy(),actual_feedback=actual_feedback,
                speed=state.signed_speed_m_s))
            if tick%round(10/exchange)==0:
                saved.append(dict(time_s=time_s,**{k:v for k,v in nstate.items() if k!='time_s'},
                    pose=state.pose.copy(),muscle=state.muscle.copy(),side=state.side_length.copy(),
                    neural_command=neural_command.copy(),external=external.copy()))
            closure_error=max(closure_error,abs(actual_temperature-(293.15+gradient@state.head_xy_m+warming)))
            max_geometry_error=max(max_geometry_error,float(abs(state.side_length-side_lengths(state.pose)).max()))
            if tick==count:break
            # Inputs use only accepted t observations, held during [t,t+exchange].
            state=body.step(round(exchange),muscle_drive=command)
            network.advance(exchange)
            if tick%1000==0:print(name,'completed_ms',tick*exchange,flush=True)
    a=all_rows;s=saved
    values=dict(time_s=np.array([r['time_s'] for r in s]),
        neuron_voltage_mV=np.stack([r['neuron_voltage_mV'] for r in s]),
        muscle_voltage_mV=np.stack([r['muscle_voltage_mV'] for r in s]),
        muscle_calcium_mM=np.stack([r['muscle_calcium_mM'] for r in s]),
        pose=np.stack([r['pose'] for r in s]),body_muscle=np.stack([r['muscle'] for r in s]),
        side_length=np.stack([r['side'] for r in s]),neural_command=np.stack([r['neural_command'] for r in s]),
        external_command=np.stack([r['external'] for r in s]),
        exchange_time_s=np.array([r['time_s'] for r in a]),head_xy_m=np.stack([r['head'] for r in a]),
        actual_temperature_K=np.array([r['temperature'] for r in a]),received_temperature_K=np.array([r['received'] for r in a]),
        noise_K=np.array([r['noise'] for r in a]),AFD_current_pA=np.array([r['afd'] for r in a]),
        threshold_K=np.array([r['threshold'] for r in a]),filtered_temperature_K=np.array([r['filtered'] for r in a]),
        feedback_current_pA=np.stack([r['feedback'] for r in a]),
        actual_feedback_candidate_pA=np.stack([r['actual_feedback'] for r in a]),
        signed_speed_m_s=np.array([r['speed'] for r in a]))
    if not all(np.isfinite(v).all() for v in values.values()):raise RuntimeError('nonfinite closed-loop trace')
    assert max_geometry_error<1e-17
    np.savez_compressed(folder/'trace.npz',**values)
    write(folder/'record.json',dict(spec=spec,neurons=302,muscles=95,neural_dt_ms=dt,exchange_ms=exchange,
        trace_sha256=sha(folder/'trace.npz'),complete=True,neural_fits=0,wall_s=time.monotonic()-started,
        sampling_equation_error_K=closure_error,geometry_error_m=max_geometry_error,
        max_AFD_current_pA=float(values['AFD_current_pA'].max()),
        max_feedback_current_pA=float(abs(values['feedback_current_pA']).max()),
        replay_reference_sha256=sha(root/spec['replay']/'trace.npz') if spec['replay'] else None))
    check(root);print('Completed',name,suffix,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run'])
    p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--name')
    p.add_argument('--neural-dt-ms',type=float);p.add_argument('--exchange-ms',type=float);p.add_argument('--suffix',default='')
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.root)
    else:run(a.root,a.name,neural_dt_ms=a.neural_dt_ms,exchange_ms=a.exchange_ms,suffix=a.suffix)
