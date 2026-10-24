"""Phase/initial-temperature controls without changing archived scene code.

The wrapper constructs a different physical temperature field, not a policy
feature. Its local readings are T_spatial+A[sin(wt+phi)-sin(phi)]+noise.
The subtraction holds initial error/radius fixed across phases. In radial
fields this necessarily shifts the mean target radius; that is reported.
"""
from dataclasses import replace

import numpy as np

from .isothermal_scene import SweepReading
from .moving_isotherm import moving_tracking_metrics


class PhasedTemperatureField:
    def __init__(self, controller, config, phase_rad):
        self.controller=controller
        self.config=config
        self.phase_rad=float(phase_rad)
        if not np.isfinite(self.phase_rad):raise ValueError('phase must be finite')
        if config.field=='radial':
            effective_mean=config.target_radius_m+config.uniform_wave_amplitude_K*np.sin(self.phase_rad)/config.gradient_K_m
            if effective_mean-config.uniform_wave_amplitude_K/config.gradient_K_m<=1e-12:
                raise ValueError('phase-anchored radial isotherm reaches origin; adjust initial radius')

    def correction(self,times):
        c=self.config;wt=2*np.pi*np.asarray(times)/c.uniform_wave_period_s
        return c.uniform_wave_amplitude_K*(np.sin(wt+self.phase_rad)-np.sin(wt)-np.sin(self.phase_rad))

    def observe(self,reading):
        self.controller.observe(SweepReading(reading.sample_time_s,
            reading.temperature_K+float(self.correction(reading.sample_time_s)),reading.motor_phase_rad))

    def command(self,time_s):return self.controller.command(time_s)

    def diagnostics(self):
        return dict(controller=self.controller.diagnostics(),field_phase_rad=self.phase_rad,
                    field_initial_temperature_anchored=True,global_phase_forwarded_to_policy=False)


def phased_tracking_metrics(config,result,phase_rad):
    """Equivalent coordinate/reference transformation for offline metrics only."""
    trace=np.array(result['trajectory'],copy=True)
    trace[:,0]+=phase_rad*config.uniform_wave_period_s/(2*np.pi)
    offset=config.uniform_wave_amplitude_K*np.sin(phase_rad)/config.gradient_K_m
    adjusted=config
    if config.field=='linear':
        angle=result['initial_state']['field_angle_rad']
        trace[:,1]-=offset*np.cos(angle)
        trace[:,2]-=offset*np.sin(angle)
    else:
        adjusted=replace(config,target_radius_m=config.target_radius_m+offset)
    return moving_tracking_metrics(adjusted,dict(result,trajectory=trace))
