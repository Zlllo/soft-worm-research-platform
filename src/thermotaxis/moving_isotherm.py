"""Offline evaluation relative to a moving isotherm; never policy input."""
import math

import numpy as np

from .isothermal_scene import field_error


def isotherm_normal_speed(config, times):
    """Level-set normal speed -partial_t(T)/|grad T|, known only offline."""
    times=np.asarray(times,dtype=float)
    frequency=2*math.pi/config.uniform_wave_period_s
    return -config.uniform_wave_amplitude_K*frequency*np.cos(frequency*times)/config.gradient_K_m


def moving_tracking_metrics(config, result):
    trajectory=np.asarray(result['trajectory'])
    t,x,y,heading=trajectory[:,:4].T
    angle=result['initial_state']['field_angle_rad']
    normal=np.full(len(t),angle) if config.field=='linear' else np.arctan2(y,x)
    normal_speed=config.speed_m_s*np.cos(heading-normal)
    tangent_speed=config.speed_m_s*np.sin(heading-normal)
    front_speed=isotherm_normal_speed(config,t)
    residual=normal_speed-front_speed
    relative_angle=np.arctan2(np.abs(residual),np.abs(tangent_speed))
    relative_angle=np.where(np.hypot(residual,tangent_speed)>1e-12,relative_angle,math.pi/2)
    static_angle=np.arcsin(np.clip(np.abs(np.cos(heading-normal)),0,1))
    error=field_error(config,x,y,t,angle)
    comfort=np.abs(error)<=config.comfort_half_width_K
    moving=comfort & (relative_angle<=config.tracking_angle_tolerance_rad)
    static=comfort & (static_angle<=config.tracking_angle_tolerance_rad)
    return dict(moving_tracking_fraction=float(moving[:-1].mean()),
        static_tangent_tracking_fraction=float(static[:-1].mean()),
        center_error_rms_K=float(np.sqrt(np.mean(error[:-1]**2))),
        normal_velocity_error_rms_m_s=float(np.sqrt(np.mean(residual[:-1]**2))),
        relative_tangent_error_mean_rad=float(relative_angle[:-1].mean()),
        moving_tracking_tangential_distance_m=float(config.physics_dt_s*np.sum(np.abs(tangent_speed[:-1])*moving[:-1])),
        maximum_front_speed_m_s=float(np.max(np.abs(front_speed[:-1]))),
        necessary_speed_feasible_fraction=float((np.abs(front_speed[:-1])<=config.speed_m_s).mean()))
