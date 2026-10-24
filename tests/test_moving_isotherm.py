from dataclasses import replace
import math

import numpy as np
import pytest

from thermotaxis.isothermal_scene import TrackingConfig,BlindTrackingController,run_tracking
from thermotaxis.moving_isotherm import isotherm_normal_speed,moving_tracking_metrics


@pytest.mark.parametrize('field',['linear','radial'])
def test_moving_metric_reduces_to_original_metric_for_static_field(field):
    config=TrackingConfig(field=field,duration_s=4.)
    row=run_tracking(config,BlindTrackingController(),9501,record_trajectory=True)
    result=moving_tracking_metrics(config,row)
    assert np.array_equal(isotherm_normal_speed(config,[0,1,2]),[0,0,0])
    assert result['moving_tracking_fraction']==row['metrics']['tracking_fraction']
    assert result['static_tangent_tracking_fraction']==row['metrics']['tracking_fraction']
    assert result['moving_tracking_tangential_distance_m']==pytest.approx(row['metrics']['tracking_tangential_distance_m'])


@pytest.mark.parametrize('amplitude,period',[(.05,60.),(.1,120.)])
def test_exact_moving_front_reference_is_valid_even_when_static_tangent_rule_fails(amplitude,period):
    config=TrackingConfig(uniform_wave_amplitude_K=amplitude,uniform_wave_period_s=period)
    t=np.arange(round(config.duration_s/config.physics_dt_s)+1)*config.physics_dt_s
    vx=isotherm_normal_speed(config,t)
    vy=np.sqrt(config.speed_m_s**2-vx**2)
    x=-amplitude*np.sin(2*math.pi*t/period)/config.gradient_K_m
    y=np.r_[0,np.cumsum((vy[1:]+vy[:-1])*.5*config.physics_dt_s)]
    heading=np.arctan2(vy,vx)
    omega=np.gradient(heading,config.physics_dt_s)
    trace=np.column_stack((t,x,y,heading,omega,x,y,np.zeros(len(t))))
    row=dict(trajectory=trace,initial_state={'field_angle_rad':0.})
    metrics=moving_tracking_metrics(config,row)
    assert metrics['moving_tracking_fraction']==1
    assert metrics['static_tangent_tracking_fraction']<.5
    assert metrics['center_error_rms_K']<1e-12
    assert metrics['normal_velocity_error_rms_m_s']<1e-12
    assert metrics['necessary_speed_feasible_fraction']==1
    assert np.max(np.abs(omega))<config.max_turn_rate_rad_s
    assert metrics['maximum_front_speed_m_s']==pytest.approx(.0001308996939,rel=1e-8)
