"""2D active-sampling benchmark with a prescribed kinematic head sweep.

This is NOT an RFT, elastic-body, crawling-contact or metabolic-energy model.
Forward speed is prescribed; only angular response is dynamically constrained.
Policy observations are local noisy temperature, sample time and motor phase.
True position, heading and field derivatives are used only for offline metrics.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TrackingConfig:
    schema_version: int = 2
    field: str = 'linear'
    preferred_K: float = 293.15
    gradient_K_m: float = 40.
    target_radius_m: float = .005
    uniform_wave_amplitude_K: float = 0.
    uniform_wave_period_s: float = 60.
    length_m: float = .001
    head_lateral_amplitude_m: float = .00015
    head_frequency_Hz: float = .5
    speed_m_s: float = .0002
    max_turn_rate_rad_s: float = .3
    turn_response_tau_s: float = .3
    physics_dt_s: float = .01
    sensor_dt_s: float = .1
    control_dt_s: float = .2
    duration_s: float = 120.
    noise_std_K: float = .002
    comfort_half_width_K: float = .05
    tracking_angle_tolerance_rad: float = math.radians(25)
    initial_error_fraction: float = .4
    initial_heading_jitter_rad: float = .35

    def __post_init__(self):
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version,int) or self.schema_version != 2:
            raise ValueError('tracking schema_version must be 2')
        if self.field not in ('linear','radial'):
            raise ValueError('field must be linear or radial')
        for key,value in asdict(self).items():
            if key in ('field','schema_version'):
                continue
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'{key} must be finite numeric')
        for key in ('gradient_K_m','target_radius_m','uniform_wave_period_s','length_m',
                    'head_lateral_amplitude_m','head_frequency_Hz','speed_m_s',
                    'max_turn_rate_rad_s','physics_dt_s','sensor_dt_s','control_dt_s',
                    'duration_s','comfort_half_width_K','tracking_angle_tolerance_rad'):
            if getattr(self,key)<=0:
                raise ValueError(f'{key} must be positive')
        for key in ('uniform_wave_amplitude_K','turn_response_tau_s','noise_std_K',
                    'initial_error_fraction','initial_heading_jitter_rad'):
            if getattr(self,key)<0:
                raise ValueError(f'{key} must be nonnegative')
        for key in ('sensor_dt_s','control_dt_s','duration_s'):
            ratio=getattr(self,key)/self.physics_dt_s
            if ratio<1 or abs(ratio-round(ratio))>1e-9:
                raise ValueError(f'{key} must be integer physics steps')
        if self.sensor_dt_s*self.head_frequency_Hz>=.5:
            raise ValueError('head phase is undersampled at the sensor clock')
        if not 0<=self.initial_error_fraction<1:
            raise ValueError('initial error fraction must be in [0,1)')
        if self.target_radius_m<=self.initial_error_fraction*self.comfort_half_width_K/self.gradient_K_m:
            raise ValueError('radial initial radius must remain positive')


@dataclass(frozen=True)
class SweepReading:
    sample_time_s: float
    temperature_K: float
    motor_phase_rad: float


@dataclass(frozen=True)
class RegressionControlConfig:
    preferred_K: float = 293.15
    comfort_scale_K: float = .05
    speed_m_s: float = .0002
    head_forward_offset_m: float = .0005
    head_lateral_amplitude_m: float = .00015
    window_s: float = 4.
    reference_gradient_K_m: float = 40.
    error_gain_rad_s: float = .2
    alignment_gain_rad_s: float = .4
    max_turn_rate_rad_s: float = .3
    max_condition_number: float = 100.

    def __post_init__(self):
        for key,value in asdict(self).items():
            if isinstance(value,bool) or not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
            if key not in ('preferred_K','head_forward_offset_m') and value<=0:
                raise ValueError(f'{key} must be positive')
        if self.head_forward_offset_m<0:
            raise ValueError('head forward offset must be nonnegative')


class BlindTrackingController:
    def observe(self, reading):
        pass

    def command(self, time_s):
        return 0.

    def diagnostics(self):
        return dict(controller='blind_constant_heading')


class PhaseRegressionController:
    """Local least-squares heuristic, NOT a biological circuit reconstruction.

    Fit T=b0+b1*(t-now)/window+b2*sin(phi)+b3*cos(phi).
    Under a locally straight, static, slowly turning path, b1/window/speed
    estimates the longitudinal gradient and b2/head_amplitude the lateral one.
    Global thermal drift, curvature and head-forward motion can bias estimates.
    """
    def __init__(self,config=RegressionControlConfig()):
        self.config=config
        self.history=deque()
        self.invalid_windows=0
        self.valid_windows=0
        self.last_estimate=None
        self.last_condition=None

    def observe(self,reading):
        if self.history and reading.sample_time_s<=self.history[-1].sample_time_s:
            raise ValueError('sensor timestamps must strictly increase')
        if not all(math.isfinite(v) for v in asdict(reading).values()):
            raise ValueError('sensor reading must be finite')
        self.history.append(reading)
        cutoff=reading.sample_time_s-self.config.window_s
        while self.history and self.history[0].sample_time_s<cutoff-1e-10:
            self.history.popleft()

    def estimate(self):
        c=self.config
        if len(self.history)<12 or self.history[-1].sample_time_s-self.history[0].sample_time_s<.9*c.window_s:
            return None
        newest=self.history[-1].sample_time_s
        t=np.array([r.sample_time_s-newest for r in self.history])/c.window_s
        phi=np.array([r.motor_phase_rad for r in self.history])
        matrix=np.column_stack((np.ones(len(t)),t,np.sin(phi),np.cos(phi)))
        condition=float(np.linalg.cond(matrix))
        self.last_condition=condition
        if not math.isfinite(condition) or condition>c.max_condition_number:
            return None
        measured=np.array([r.temperature_K-c.preferred_K for r in self.history])
        weights,_,rank,_=np.linalg.lstsq(matrix,measured,rcond=None)
        if rank!=4 or not np.all(np.isfinite(weights)):
            return None
        longitudinal=float(weights[1]/c.window_s/c.speed_m_s)
        lateral=float(weights[2]/c.head_lateral_amplitude_m)
        centre_error=float(weights[0]-c.head_forward_offset_m*longitudinal)
        return dict(longitudinal_K_m=longitudinal,lateral_K_m=lateral,
                    estimated_center_error_K=centre_error,condition_number=condition)

    def command(self,time_s):
        del time_s
        estimate=self.estimate()
        self.last_estimate=estimate
        if estimate is None:
            self.invalid_windows+=1
            return 0.
        self.valid_windows+=1
        c=self.config
        longitudinal=estimate['longitudinal_K_m']/c.reference_gradient_K_m
        lateral=estimate['lateral_K_m']/c.reference_gradient_K_m
        error=estimate['estimated_center_error_K']/c.comfort_scale_K
        rate=-c.error_gain_rad_s*error*lateral-c.alignment_gain_rad_s*longitudinal*lateral
        return float(np.clip(rate,-c.max_turn_rate_rad_s,c.max_turn_rate_rad_s))

    def diagnostics(self):
        return dict(controller='local_phase_regression_heuristic',valid_windows=self.valid_windows,
                    invalid_windows=self.invalid_windows,last_estimate=self.last_estimate)


def advance_pose(x,y,heading,omega,target,dt,speed,tau):
    """Exact held-target angular relaxation; midpoint forward-position step."""
    if tau==0:
        theta_mid=heading+target*dt/2
        next_heading=heading+target*dt
        next_omega=target
    else:
        angular_step=lambda h: target*h+(omega-target)*tau*(-math.expm1(-h/tau))
        theta_mid=heading+angular_step(dt/2)
        next_heading=heading+angular_step(dt)
        next_omega=target+(omega-target)*math.exp(-dt/tau)
    return x+speed*dt*math.cos(theta_mid),y+speed*dt*math.sin(theta_mid),next_heading,next_omega


def head_position(config,x,y,heading,phase):
    lateral=config.head_lateral_amplitude_m*math.sin(phase)
    forward=config.length_m/2
    return (x+forward*math.cos(heading)-lateral*math.sin(heading),
            y+forward*math.sin(heading)+lateral*math.cos(heading))


def field_error(config,x,y,time_s,field_angle):
    if config.field=='linear':
        spatial=config.gradient_K_m*(x*np.cos(field_angle)+y*np.sin(field_angle))
    else:
        spatial=config.gradient_K_m*(np.hypot(x,y)-config.target_radius_m)
    return spatial+config.uniform_wave_amplitude_K*np.sin(2*np.pi*np.asarray(time_s)/config.uniform_wave_period_s)


def _longest_true_duration(mask,dt):
    longest=active=0
    for value in mask:
        active=active+1 if value else 0
        longest=max(longest,active)
    return longest*dt


def run_tracking(config,controller,seed,*,sensor_location='head',phase_mode='correct',record_trajectory=False):
    if sensor_location not in ('head','center') or phase_mode not in ('correct','quarter_offset','scrambled'):
        raise ValueError('invalid sensor/phase condition')
    streams=[np.random.default_rng(np.random.SeedSequence([seed,i])) for i in range(4)]
    environment_rng,initial_rng,sensor_rng,phase_rng=streams
    field_angle=float(environment_rng.uniform(-math.pi,math.pi))
    offset_K=float(initial_rng.uniform(-1,1)*config.initial_error_fraction*config.comfort_half_width_K)
    direction=float(initial_rng.choice([-1,1]))
    initial_phase=float(initial_rng.uniform(-math.pi,math.pi))
    if config.field=='linear':
        angle=field_angle
        radius=offset_K/config.gradient_K_m
    else:
        angle=float(initial_rng.uniform(-math.pi,math.pi))
        radius=config.target_radius_m+offset_K/config.gradient_K_m
    x,y=radius*math.cos(angle),radius*math.sin(angle)
    heading=angle+direction*math.pi/2+float(initial_rng.uniform(-config.initial_heading_jitter_rad,config.initial_heading_jitter_rad))
    initial_state=dict(x_m=x,y_m=y,heading_rad=heading,motor_phase_rad=initial_phase,field_angle_rad=field_angle)
    steps=round(config.duration_s/config.physics_dt_s)
    sensor_every=round(config.sensor_dt_s/config.physics_dt_s)
    control_every=round(config.control_dt_s/config.physics_dt_s)
    # Columns: time, center x/y, heading, angular rate, head x/y, physical phase.
    trajectory=np.empty((steps+1,8))
    omega=target=0.
    samples=[]
    controls=[]
    for step in range(steps+1):
        time=step*config.physics_dt_s
        phase=initial_phase+2*math.pi*config.head_frequency_Hz*time
        hx,hy=head_position(config,x,y,heading,phase)
        if step%sensor_every==0:
            sx,sy=(hx,hy) if sensor_location=='head' else (x,y)
            truth=config.preferred_K+float(field_error(config,sx,sy,time,field_angle))
            measured=truth+float(sensor_rng.normal(0,config.noise_std_K))
            reported_phase=phase
            if phase_mode=='quarter_offset':
                reported_phase+=math.pi/2
            elif phase_mode=='scrambled':
                reported_phase=float(phase_rng.uniform(-math.pi,math.pi))
            controller.observe(SweepReading(time,measured,reported_phase))
            samples.append((time,measured,reported_phase,truth))
        if step%control_every==0:
            target=float(controller.command(time))
            if not math.isfinite(target) or abs(target)>config.max_turn_rate_rad_s+1e-12:
                raise ValueError('controller commanded an invalid turn rate')
            controls.append((time,target))
        trajectory[step]=(time,x,y,heading,omega,hx,hy,phase)
        if step<steps:
            x,y,heading,omega=advance_pose(x,y,heading,omega,target,config.physics_dt_s,config.speed_m_s,config.turn_response_tau_s)
    t,xs,ys,angles,rates,hxs,hys,_=trajectory.T
    center_error=field_error(config,xs,ys,t,field_angle)
    head_error=field_error(config,hxs,hys,t,field_angle)
    if config.field=='linear':
        normal_angle=np.full(len(t),field_angle)
    else:
        normal_angle=np.arctan2(ys,xs)
    # Both tangent directions count; true direction is offline evaluation only.
    tangent_error=np.arcsin(np.clip(np.abs(np.cos(angles-normal_angle)),0,1))
    comfort=np.abs(center_error)<=config.comfort_half_width_K
    tracking=comfort & (tangent_error<=config.tracking_angle_tolerance_rad)
    dt=config.physics_dt_s
    # Left endpoint quadrature; t=duration has zero measure, no duplicate duration.
    active=tracking[:-1]
    tangential_speed=config.speed_m_s*np.abs(np.sin(angles[:-1]-normal_angle[:-1]))
    signed_tangential_speed=config.speed_m_s*np.sin(angles[:-1]-normal_angle[:-1])
    metrics=dict(center_residence_fraction=float(comfort[:-1].mean()),
        head_residence_fraction=float((np.abs(head_error[:-1])<=config.comfort_half_width_K).mean()),
        tracking_fraction=float(active.mean()),longest_tracking_s=_longest_true_duration(active,dt),
        center_error_rms_K=float(np.sqrt(np.mean(center_error[:-1]**2))),
        center_error_mae_K=float(np.abs(center_error[:-1]).mean()),
        head_error_rms_K=float(np.sqrt(np.mean(head_error[:-1]**2))),
        tangent_error_mean_rad=float(tangent_error[:-1].mean()),
        tracking_tangential_distance_m=float(dt*np.sum(tangential_speed*active)),
        tracking_net_tangential_distance_m=float(abs(dt*np.sum(signed_tangential_speed*active))),
        prescribed_forward_distance_m=config.speed_m_s*config.duration_s,
        net_displacement_m=float(math.hypot(xs[-1]-xs[0],ys[-1]-ys[0])),
        integrated_abs_turn_rad=float(dt*np.abs(rates[:-1]).sum()),
        max_abs_turn_rate_rad_s=float(np.abs(rates).max()))
    result=dict(config=asdict(config),seed=seed,sensor_location=sensor_location,phase_mode=phase_mode,
        initial_state=initial_state,metrics=metrics,controller_diagnostics=controller.diagnostics(),
        clocks=dict(physics_steps=steps,sensor_samples=len(samples),control_updates=len(controls)),
        observation_contract='tracking_local_temperature_and_motor_phase_v2',
        model_scope='prescribed 2D kinematics; not RFT, elasticity, contact or metabolic cost')
    if record_trajectory:
        result['trajectory']=trajectory
        result['sensor_samples']=np.asarray(samples)
        result['control_commands']=np.asarray(controls)
    return result
