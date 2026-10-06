"""Independent contracts for the finite-response body/action bridge."""
from dataclasses import asdict, replace
import math
import numpy as np
import pytest
from thermotaxis.phase2_control import (ControlConfig, FiniteDrive, PolicyObservation,
                                       action_heading, head_position, run_four_action)
from thermotaxis.phase2_body import solve_motion
from thermotaxis.legacy_policy import legacy_action_heading


def test_action_mapping_and_strict_config():
    for action in range(4):
        assert math.cos(action_heading(action)) == pytest.approx(math.cos(legacy_action_heading(action,y_axis_up=True)))
        assert math.sin(action_heading(action)) == pytest.approx(math.sin(legacy_action_heading(action,y_axis_up=True)))
    for bad in (True, np.bool_(True), 1.0, -1, 4):
        with pytest.raises(ValueError): action_heading(bad)
    for changes in ({'schema_version':1.0},{'physics_dt_s':.03},{'drive_tau_s':0},{'initially_active':1}):
        with pytest.raises(ValueError): replace(ControlConfig(),**changes)
    config=ControlConfig()
    assert ControlConfig.from_dict(asdict(config)) == config
    with pytest.raises(ValueError): ControlConfig.from_dict({**asdict(config),'extra':1})


def test_exact_relaxation_phase_derivative_and_command_continuity():
    config=ControlConfig(); drive=FiniteDrive(config)
    drive.targets=(.6,1.,-.4)
    t=.37; d=drive.at(t); eps=1e-6
    before,after=drive.at(t-eps),drive.at(t+eps)
    for field,rate in (('amplitude_rad','amplitude_rate_rad_s'),('phase_rad','phase_rate_rad_s'),('bias_rad','bias_rate_rad_s')):
        assert (getattr(after,field)-getattr(before,field))/(2*eps)==pytest.approx(getattr(d,rate),rel=1e-8,abs=1e-10)
    assert d.phase_rad==pytest.approx(2*math.pi*(t-.5*(1-math.exp(-t/.5))))
    split=FiniteDrive(config);split.targets=drive.targets
    for _ in range(37): split.advance(.01)
    np.testing.assert_allclose(list(asdict(split.at(0)).values()),list(asdict(d).values()),rtol=1e-12,atol=1e-12)
    split.targets=(0.,0.,.7)
    new=split.at(0)
    for field in ('amplitude_rad','phase_rad','bias_rad'):
        assert getattr(new,field)==pytest.approx(getattr(d,field))


def test_local_information_sampling_and_pose_contract():
    config=replace(ControlConfig(),duration_s=.4,sensor_dt_s=.2,control_dt_s=.1)
    observations=[];calls=[]
    def sampler(head,t):
        calls.append((head,t));return 290.+t
    def policy(obs):
        observations.append(obs);return 0
    result=run_four_action(config,policy,temperature_sampler=sampler)
    assert set(asdict(observations[0]))=={'time_s','sample_time_s','temperature_K','previous_action'}
    assert all(isinstance(o,PolicyObservation) for o in observations)
    assert [o.sample_time_s for o in observations]==pytest.approx([0,0,.2,.2])
    assert [o.temperature_K for o in observations]==pytest.approx([290,290,290.2,290.2])
    assert [t for _,t in calls]==pytest.approx([0,.2,.4])
    first=result['trajectory'][0]
    assert (first['x_m'],first['y_m'],first['orientation_rad'])==(0,0,0)
    assert first['head_x_m']==pytest.approx(-config.body.length_m/2)
    assert first['amplitude_rad']==0
    assert first['bias_rad']==0
    with pytest.raises(ValueError): run_four_action(config,policy,legacy_observer=lambda o,p:p)
    with pytest.raises(ValueError): run_four_action(replace(config,observation_mode='legacy_privileged'),policy)


def test_logged_power_head_and_energy_from_actual_drive():
    config=replace(ControlConfig(),duration_s=.4)
    result=run_four_action(config,lambda o:0)
    assert result['diagnostics']['max_force_residual_N'] < 1e-17
    assert result['diagnostics']['max_torque_residual_N_m'] < 1e-20
    from thermotaxis.phase2_body import DriveState
    energies=[]
    for row in result['trajectory'][::5]:
        d=DriveState(**{k:row[k] for k in DriveState.__dataclass_fields__})
        pose=np.array([row['x_m'],row['y_m'],row['orientation_rad']])
        m=solve_motion(config.body,row['time_s'],pose[2],d)
        assert row['power_W']==pytest.approx(m.power_W,rel=1e-12,abs=1e-25)
        np.testing.assert_allclose(head_position(config,d,row['time_s'],pose),[row['head_x_m'],row['head_y_m']],atol=1e-15)
        energies.append(row['dissipation_J'])
    assert np.all(np.diff(energies)>=0)
    assert energies[-1]>0


def test_never_started_rest_and_finite_stop():
    config=replace(ControlConfig(),duration_s=.8)
    stopped=run_four_action(config,lambda o:None)
    assert all(row['power_W']==0 and row['dissipation_J']==0 and row['x_m']==0 and row['y_m']==0 for row in stopped['trajectory'])
    active=run_four_action(config,lambda o:2 if o.time_s<.4 else None)
    row=active['trajectory'][20]
    assert row['action'] is None
    assert row['amplitude_rad']>0 and row['phase_rate_rad_s']>0
    assert active['trajectory'][-1]['amplitude_rad']<row['amplitude_rad']
    assert active['trajectory'][-1]['dissipation_J']>row['dissipation_J']
