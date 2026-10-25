from dataclasses import replace
import math
import numpy as np

from thermotaxis.navigation import NavigationBatch,NavigationConfig


def test_original_rotation_can_make_recovery_impossible_even_from_old_isotherm():
    c=replace(NavigationConfig(),event_s=140.,event_jitter_s=0.)
    env=NavigationBatch(c,[31],fields=[0]);normal=env.angle[0]
    point=.028*np.array([-math.sin(normal),math.cos(normal)])
    before=env.field(np.array([point[0]]),np.array([point[1]]),0.)[0]
    after=env.field(np.array([point[0]]),np.array([point[1]]),140.)[0]
    assert abs(before)<1e-12
    lower=max(0,abs(after)-c.comfort_K)/c.gradient_K_m/c.speed_m_s+c.stable_s
    assert lower>100.


def test_small_rotation_speed_bound_does_not_rule_out_old_band_states_with_travel_limit():
    c=replace(NavigationConfig(),event_s=140.,event_jitter_s=0.,field_turn_rad=math.pi/12)
    env=NavigationBatch(c,[31],fields=[0]);angle=env.angle[0]
    n=np.array([math.cos(angle),math.sin(angle)]);t=np.array([-math.sin(angle),math.cos(angle)])
    limits=[]
    for normal in [-c.comfort_K/c.gradient_K_m,0.,c.comfort_K/c.gradient_K_m]:
        for tangent in np.linspace(-.028,.028,101):
            point=normal*n+tangent*t
            error=env.field(np.array([point[0]]),np.array([point[1]]),140.)[0]
            lower=max(0,abs(error)-c.comfort_K)/c.gradient_K_m/c.speed_m_s+c.stable_s
            limits.append(lower)
    assert max(limits)<60. <100.
