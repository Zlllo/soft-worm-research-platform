from dataclasses import replace
import math

import numpy as np
import pytest

from thermotaxis.isothermal_scene import TrackingConfig,BlindTrackingController,PhaseRegressionController,SweepReading,run_tracking,field_error
from thermotaxis.moving_isotherm import moving_tracking_metrics
from thermotaxis.phased_field import PhasedTemperatureField,phased_tracking_metrics


class Capture(BlindTrackingController):
    def observe(self,reading):self.reading=reading


def test_radial_phase_control_rejects_collapsing_target_and_accepts_fixed_mean_initialization():
    config=TrackingConfig(field='radial',uniform_wave_amplitude_K=.1,uniform_wave_period_s=120.)
    phase=3*math.pi/2
    with pytest.raises(ValueError,match='reaches origin'):
        PhasedTemperatureField(BlindTrackingController(),config,phase)
    initialized=replace(config,target_radius_m=config.target_radius_m-.1*math.sin(phase)/40)
    wrapper=PhasedTemperatureField(BlindTrackingController(),initialized,phase)
    assert wrapper.correction(0.)==0


def test_phase_is_world_temperature_only_and_initial_error_is_unchanged():
    config=TrackingConfig(uniform_wave_amplitude_K=.05)
    recorder=Capture();wrapper=PhasedTemperatureField(recorder,config,math.pi/2)
    for t in [0.,1.,2.]:
        original=SweepReading(t,293.16,.8)
        wrapper.observe(original)
        expected=.05*(np.sin(2*np.pi*t/60+math.pi/2)-np.sin(2*np.pi*t/60)-1)
        assert recorder.reading.temperature_K==pytest.approx(original.temperature_K+expected)
        assert recorder.reading.sample_time_s==original.sample_time_s
        assert recorder.reading.motor_phase_rad==original.motor_phase_rad
    assert wrapper.correction(0.)==0


@pytest.mark.parametrize('field',['linear','radial'])
def test_zero_phase_is_bitwise_same_simulation_and_offline_metric(field):
    config=TrackingConfig(field=field,duration_s=8.,uniform_wave_amplitude_K=.05)
    original=run_tracking(config,PhaseRegressionController(),10101,record_trajectory=True)
    wrapped=run_tracking(config,PhasedTemperatureField(PhaseRegressionController(),config,0.),10101,record_trajectory=True)
    assert np.array_equal(original['trajectory'],wrapped['trajectory'])
    assert np.array_equal(original['sensor_samples'],wrapped['sensor_samples'])
    assert phased_tracking_metrics(config,wrapped,0.)==moving_tracking_metrics(config,original)


@pytest.mark.parametrize('field',['linear','radial'])
def test_offline_metric_matches_explicit_phased_field_temperature(field):
    config=TrackingConfig(field=field,duration_s=4.,uniform_wave_amplitude_K=.1,uniform_wave_period_s=120.)
    phase=math.pi/2
    wrapper=PhasedTemperatureField(BlindTrackingController(),config,phase)
    row=run_tracking(config,wrapper,10102,record_trajectory=True)
    t,x,y=row['trajectory'][:,:3].T
    actual=field_error(config,x,y,t,row['initial_state']['field_angle_rad'])+wrapper.correction(t)
    metrics=phased_tracking_metrics(config,row,phase)
    assert metrics['center_error_rms_K']==pytest.approx(float(np.sqrt(np.mean(actual[:-1]**2))),abs=1e-12)
    assert metrics['maximum_front_speed_m_s']==pytest.approx(float(np.max(np.abs(-.1*2*np.pi/120*np.cos(2*np.pi*t[:-1]/120+phase)/40))),abs=1e-12)
