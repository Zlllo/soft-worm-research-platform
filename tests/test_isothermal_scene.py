from dataclasses import replace
import math

import numpy as np
import pytest

from thermotaxis.isothermal_scene import (TrackingConfig,RegressionControlConfig,SweepReading,
    PhaseRegressionController,BlindTrackingController,advance_pose,head_position,run_tracking)


def test_config_rejects_aliasing_and_mismatched_clocks():
    with pytest.raises(ValueError,match='undersampled'):
        TrackingConfig(sensor_dt_s=1.)
    with pytest.raises(ValueError,match='integer physics'):
        TrackingConfig(sensor_dt_s=.115)
    with pytest.raises(ValueError,match='schema'):
        TrackingConfig(schema_version=2.)


def test_phase_regression_recovers_analytic_local_gradient_and_phase_sign():
    c=RegressionControlConfig()
    # Global gradient (40,0), straight heading .6rad. Center begins at x=.001m.
    heading=.6
    along=40*math.cos(heading)
    lateral=-40*math.sin(heading)
    times=np.arange(0,4.00001,.1)
    for offset,factor in [(0.,1.),(math.pi/2,0.),(math.pi,-1.)]:
        controller=PhaseRegressionController(c)
        for t in times:
            phi=math.pi*t+.3
            temp=c.preferred_K+40*.001+along*c.speed_m_s*t+along*c.head_forward_offset_m+lateral*c.head_lateral_amplitude_m*math.sin(phi)
            controller.observe(SweepReading(float(t),temp,phi+offset))
        estimate=controller.estimate()
        assert estimate['longitudinal_K_m']==pytest.approx(along,abs=1e-8)
        assert estimate['lateral_K_m']==pytest.approx(lateral*factor,abs=1e-8)
        assert estimate['estimated_center_error_K']==pytest.approx(.04+along*c.speed_m_s*times[-1],abs=1e-10)


def test_singular_phase_window_does_not_create_a_turn_command():
    control=PhaseRegressionController()
    for t in np.arange(0,4.01,.1):
        control.observe(SweepReading(float(t),293.15+.01*t,0.))
    assert control.command(4.)==0.
    assert control.last_estimate is None
    assert control.invalid_windows==1


def test_head_geometry_and_exact_angular_relaxation():
    c=TrackingConfig()
    hx,hy=head_position(c,0.,0.,0.,math.pi/2)
    assert (hx,hy)==pytest.approx((c.length_m/2,c.head_lateral_amplitude_m))
    x,y,theta,omega=advance_pose(0.,0.,0.,0.,.3,.2,.0002,.5)
    assert omega==pytest.approx(.3*(1-math.exp(-.4)))
    assert theta==pytest.approx(.3*(.2-.5*(1-math.exp(-.4))))
    assert math.hypot(x,y)==pytest.approx(.0002*.2)
    # Exact angle/omega are invariant to time subdivision, position converges separately.
    a=advance_pose(0.,0.,0.,0.,.3,.1,.0002,.5)
    b=advance_pose(*a,.3,.1,.0002,.5)
    assert b[2:]==pytest.approx((theta,omega))


def test_reported_phase_does_not_modify_physics_or_sensor_noise():
    c=TrackingConfig(duration_s=4.)
    correct=run_tracking(c,BlindTrackingController(),101,record_trajectory=True)
    scrambled=run_tracking(c,BlindTrackingController(),101,phase_mode='scrambled',record_trajectory=True)
    assert np.array_equal(correct['trajectory'],scrambled['trajectory'])
    assert np.array_equal(correct['sensor_samples'][:,[0,1,3]],scrambled['sensor_samples'][:,[0,1,3]])
    assert not np.array_equal(correct['sensor_samples'][:,2],scrambled['sensor_samples'][:,2])


def test_physics_refinement_preserves_sensor_clock_and_noise_stream():
    c=TrackingConfig(duration_s=4.)
    coarse=run_tracking(c,BlindTrackingController(),101,record_trajectory=True)
    refined=run_tracking(replace(c,physics_dt_s=.005),BlindTrackingController(),101,record_trajectory=True)
    assert coarse['clocks']['sensor_samples']==refined['clocks']['sensor_samples']==41
    assert coarse['clocks']['control_updates']==refined['clocks']['control_updates']==21
    assert np.allclose(coarse['sensor_samples'],refined['sensor_samples'],atol=1e-10,rtol=0.)


def test_controller_observation_has_no_truth_geometry_and_simulation_repeats():
    assert set(SweepReading.__dataclass_fields__)=={'sample_time_s','temperature_K','motor_phase_rad'}
    c=TrackingConfig(duration_s=8.)
    a=run_tracking(c,PhaseRegressionController(),101)
    b=run_tracking(c,PhaseRegressionController(),101)
    assert a==b
    assert a['metrics']['max_abs_turn_rate_rad_s']<=c.max_turn_rate_rad_s
    assert a['metrics']['prescribed_forward_distance_m']==pytest.approx(c.duration_s*c.speed_m_s)
