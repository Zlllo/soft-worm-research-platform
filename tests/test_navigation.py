from dataclasses import replace
import math

import numpy as np
import pytest

from thermotaxis.navigation import NavigationBatch,NavigationConfig
from thermotaxis.sensory_adaptation import ThresholdAdaptation


def config(**kw):
    values=dict(duration_s=30.,event_s=15.,event_jitter_s=1.,stable_s=2.);values.update(kw)
    return replace(NavigationConfig(),**values)


def test_initial_states_are_outside_band_on_both_sides_and_packets_have_no_geometry():
    env=NavigationBatch(config(),range(100,164))
    e=env.field(env.x,env.y,0.)
    assert np.all(np.abs(e)>.05) and np.all(np.abs(e)<=.15+1e-12)
    assert np.any(e>0) and np.any(e<0)
    packet=env.initial_packet();assert packet.shape==(2,64,5)
    assert np.all(packet[0]==0) and np.all(packet[-1,:,-1]==1)
    assert np.std(env.heading)>1.


def test_field_event_and_sensor_streams_are_paired_independent_of_action():
    c=config(noise_K=0.)
    a=NavigationBatch(c,[51,52]);b=NavigationBatch(c,[51,52])
    assert np.array_equal(a.initial_packet(),b.initial_packet())
    for _ in range(5):
        p,_,_=a.step(np.zeros((2,2)));q,_,_=b.step(np.zeros((2,2)))
        assert np.array_equal(p,q)
    assert np.array_equal(a.event,b.event)
    x=np.array([0.,c.radius_m]);y=np.zeros(2)
    before=a.field(x,y,0.);after=a.field(x,y,30.)
    assert np.max(np.abs(after-before))>0
    assert np.all(a.event/c.control_dt_s==np.round(a.event/c.control_dt_s))


def test_constant_turn_integrates_circle_and_refined_clock_agrees():
    c=config(response_tau_s=0.,noise_K=0.)
    a=NavigationBatch(c,[51]);b=NavigationBatch(replace(c,physics_dt_s=.01),[51])
    x,y,h=a.x[0],a.y[0],a.heading[0];omega=.15
    for _ in range(5):
        a.step(np.array([[.5,1.]]));b.step(np.array([[.5,1.]]))
    expected=[x+c.speed_m_s/omega*(math.sin(h+omega)-math.sin(h)),
        y-c.speed_m_s/omega*(math.cos(h+omega)-math.cos(h))]
    assert np.allclose([a.x[0],a.y[0]],expected,atol=1e-9,rtol=0)
    assert np.allclose([a.x[0],a.y[0]],[b.x[0],b.y[0]],atol=1e-9,rtol=0)


def test_failure_times_remain_censored_and_empty_completion_is_not_success():
    env=NavigationBatch(config(),[11,12])
    for _ in range(150):env.step(np.zeros((2,2)))
    for row in env.metrics():
        assert isinstance(row['complete_success'],bool)
        for k in ['first_entry_s','pre_stable_s','reentry_s','post_stable_s']:
            assert row[k] is None or row[k]>=0
        assert row['prescribed_distance_m']==pytest.approx(.0002*30.)


def test_asymmetric_adaptation_matches_exponential_targets_and_retains_threshold_history():
    sensor=ThresholdAdaptation()
    assert sensor.update(1.,0.,45.)==pytest.approx(1-math.exp(-1))
    assert sensor.update(0.,1.,840.)==pytest.approx(math.exp(-1))
    b=0.
    for _ in range(1200):b=sensor.update(1.,b,.1)
    assert b==pytest.approx(1-math.exp(-120/45)) and b>.9
    assert sensor.response([.02,1.02],[0.,1.])==pytest.approx([.4,.4])


def test_invalid_navigation_contracts_are_rejected():
    for changes in [dict(initial_error_min_K=.01),dict(radius_m=.002),dict(physics_dt_s=.03),dict(event_s=239.)]:
        with pytest.raises(ValueError):NavigationConfig(**changes)


def test_future_field_schedule_is_not_in_pre_event_observations():
    a=NavigationBatch(config(event_jitter_s=0.),[13,14])
    b=NavigationBatch(replace(config(event_jitter_s=0.),event_s=20.),[13,14])
    assert np.array_equal(a.initial_packet(),b.initial_packet())
    for _ in range(20):
        p,_,_=a.step(np.zeros((2,2)));q,_,_=b.step(np.zeros((2,2)))
        assert np.array_equal(p,q)


def test_stage_metrics_distinguish_entry_from_stability_and_recovery():
    c=config(event_jitter_s=0.);env=NavigationBatch(c,[13]);env.event[:]=15.
    times=np.arange(151)*.2;h=np.zeros((151,1,8));h[:,0,0]=times
    h[:,0,1]=.1;h[:,0,2]=0;h[:,0,7]=.0002
    h[((times>=4)&(times<14))|((times>=21)&(times<29)),0,1]=0.
    env.history=list(h);row=env.metrics()[0]
    assert row['first_entry_s']==pytest.approx(4.) and row['pre_stable_s']==pytest.approx(4.)
    assert row['reentry_s']==pytest.approx(6.) and row['post_stable_s']==pytest.approx(6.)
    assert row['complete_success']
