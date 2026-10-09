"""Four historical direction actions executed by a finite-response RFT body.

SI laboratory y points up. A requested direction is a command, never a pose edit.
The low-level direction servo uses its body's proprioceptive axis theta + pi.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import math
from typing import Callable
import numpy as np
from .phase2_body import BodyConfig, DriveState, shape_geometry, solve_motion

ACTION_HEADINGS = (math.pi / 2, -math.pi / 2, math.pi, 0.)
ACTION_NAMES = ('up', 'down', 'left', 'right')


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def action_heading(action):
    if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < 4:
        raise ValueError('action must be integer 0..3, or None to relax drive')
    return ACTION_HEADINGS[int(action)]


@dataclass(frozen=True)
class ControlConfig:
    schema_version: int = 1
    body: BodyConfig = field(default_factory=BodyConfig)
    physics_dt_s: float = .02
    sensor_dt_s: float = .1
    control_dt_s: float = .2
    duration_s: float = 24.
    drive_tau_s: float = .5
    bias_tau_s: float = .5
    heading_gain: float = 2.
    max_bias_rad: float = 2.
    initial_x_m: float = 0.
    initial_y_m: float = 0.
    initial_orientation_rad: float = 0.
    initially_active: bool = False
    observation_mode: str = 'local_temperature'

    def __post_init__(self):
        if self.schema_version != 1 or isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise ValueError('schema_version must be 1')
        if not isinstance(self.body, BodyConfig) or not isinstance(self.initially_active, bool):
            raise ValueError('body/initially_active type invalid')
        if self.observation_mode not in ('local_temperature', 'legacy_privileged'):
            raise ValueError('unknown observation_mode')
        for key, value in asdict(self).items():
            if key in ('body', 'schema_version', 'initially_active', 'observation_mode'):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{key} must be finite numeric')
        for key in ('physics_dt_s','sensor_dt_s','control_dt_s','duration_s','drive_tau_s','bias_tau_s','heading_gain','max_bias_rad'):
            if getattr(self,key) <= 0:
                raise ValueError(f'{key} must be positive')
        for key in ('sensor_dt_s','control_dt_s','duration_s'):
            ratio = getattr(self,key)/self.physics_dt_s
            if abs(ratio-round(ratio)) > 1e-9 or ratio < 1:
                raise ValueError(f'{key} must be positive integer physics steps')
        b = self.body
        if any(getattr(b,k) != 0 for k in ('amplitude_modulation_rad','phase_modulation_rad','bias_rad','bias_modulation_rad')):
            raise ValueError('four-action servo requires unmodulated zero-bias reference body')
        if b.amplitude_rad < 0:
            raise ValueError('reference amplitude must be nonnegative')

    @classmethod
    def from_dict(cls, data):
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError('control config keys must exactly match schema')
        data = dict(data)
        data['body'] = BodyConfig.from_dict(data['body'])
        return cls(**data)


@dataclass(frozen=True)
class PolicyObservation:
    time_s: float
    sample_time_s: float
    temperature_K: float | None
    previous_action: int | None


class FiniteDrive:
    """Exact held-target exponential relaxation, including integrated phase."""
    def __init__(self, config):
        self.config = config
        self.amplitude = config.body.amplitude_rad if config.initially_active else 0.
        self.frequency = config.body.frequency_Hz if config.initially_active else 0.
        self.bias = 0.
        self.phase = 0.
        self.targets = (self.amplitude, self.frequency, 0.)

    def at(self, elapsed_s):
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError('elapsed drive time must be finite nonnegative')
        a_target, f_target, b_target = self.targets
        tau, btau = self.config.drive_tau_s, self.config.bias_tau_s
        ea, eb = math.exp(-elapsed_s/tau), math.exp(-elapsed_s/btau)
        a = a_target + (self.amplitude-a_target)*ea
        f = f_target + (self.frequency-f_target)*ea
        b = b_target + (self.bias-b_target)*eb
        phase = self.phase + 2*math.pi*(f_target*elapsed_s + (self.frequency-f_target)*tau*(1-ea))
        return DriveState(a,(a_target-a)/tau,phase,2*math.pi*f,b,(b_target-b)/btau)

    def advance(self, elapsed_s):
        d = self.at(elapsed_s)
        self.amplitude, self.frequency, self.bias, self.phase = d.amplitude_rad,d.phase_rate_rad_s/(2*math.pi),d.bias_rad,d.phase_rad


def head_position(config, drive, time_s, pose):
    g = shape_geometry(config.body,time_s,s_m=[0.],drive=drive)
    c,s = math.cos(pose[2]),math.sin(pose[2])
    return pose[:2] + g.position_m[0] @ np.array([[c,s],[-s,c]])


def run_four_action(config: ControlConfig, policy: Callable[[PolicyObservation], int | None], *, temperature_sampler=None, legacy_observer=None, terminal_condition=None):
    """No boundary clipping/contact. Separate sampling/control ticks include t=0.

    temperature_sampler((head_x, head_y), time) may inject noisy local readings.
    Optional terminal_condition(true_head_xy,time) ends evaluation on a
    physics tick without projecting pose; it is an evaluator, not policy input.
    Local policy receives no true coordinates or pose. In legacy_privileged
    mode an explicit legacy_observer(local_observation, true_head_xy) produces
    the old policy input. This does not imply equal information between modes.
    """
    if (config.observation_mode == 'legacy_privileged') != (legacy_observer is not None):
        raise ValueError('legacy_privileged mode requires explicit legacy_observer; local mode forbids it')
    drive = FiniteDrive(config)
    pose = np.array([config.initial_x_m,config.initial_y_m,config.initial_orientation_rad])
    rows, controls, samples = [],[],[]
    steps = round(config.duration_s/config.physics_dt_s)
    sensor_stride = round(config.sensor_dt_s/config.physics_dt_s)
    control_stride = round(config.control_dt_s/config.physics_dt_s)
    energy = 0.
    termination = None
    last_action = None
    reading = None
    sample_time = 0.
    max_force,max_torque = 0.,0.
    for step in range(steps+1):
        time = step*config.physics_dt_s
        d = drive.at(0.)
        head = head_position(config,d,time,pose)
        if step % sensor_stride == 0:
            sample_time = time
            reading = None if temperature_sampler is None else float(temperature_sampler(tuple(head),time))
            if reading is not None and not math.isfinite(reading):
                raise ValueError('temperature sample must be finite')
            samples.append({'time_s':time,'head_x_m':float(head[0]),'head_y_m':float(head[1]),'temperature_K':reading})
        terminated = terminal_condition is not None and bool(terminal_condition(tuple(head),time))
        if terminated:
            termination = {'time_s':time,'head_x_m':float(head[0]),'head_y_m':float(head[1]),'reason':'explicit_terminal_condition'}
        if step < steps and not terminated and step % control_stride == 0:
            obs = PolicyObservation(time,sample_time,reading,last_action)
            policy_input = obs if legacy_observer is None else legacy_observer(obs,tuple(head))
            action = policy(policy_input)
            if action is None:
                heading,error = None,None
                drive.targets = (0.,0.,0.)
            else:
                heading = action_heading(action)
                action = int(action)
                error = wrap_angle(heading-(pose[2]+math.pi))
                bias = float(np.clip(-config.heading_gain*error,-config.max_bias_rad,config.max_bias_rad))
                drive.targets = (config.body.amplitude_rad,config.body.frequency_Hz,bias)
            last_action = action
            controls.append({'time_s':time,'action':action,'desired_heading_rad':heading,'axis_error_rad':error,'amplitude_target_rad':drive.targets[0],'frequency_target_Hz':drive.targets[1],'bias_target_rad':drive.targets[2]})
        # Power after command changes includes parameter derivatives immediately.
        d = drive.at(0.)
        motion = solve_motion(config.body,time,pose[2],d)
        max_force=max(max_force,float(np.linalg.norm(motion.force_N)))
        max_torque=max(max_torque,abs(motion.torque_N_m))
        row={'time_s':time,'x_m':float(pose[0]),'y_m':float(pose[1]),'orientation_rad':float(pose[2]),'axis_heading_rad':wrap_angle(pose[2]+math.pi),'head_x_m':float(head[0]),'head_y_m':float(head[1]),'dissipation_J':energy,'power_W':motion.power_W,'action':last_action}
        row.update(asdict(d)); rows.append(row)
        if step == steps or terminated:
            break
        h=config.physics_dt_s
        mid=solve_motion(config.body,time+h/2,pose[2]+h*motion.angular_velocity_rad_s/2,drive.at(h/2))
        pose[:2]+=h*mid.translation_m_s
        pose[2]+=h*mid.angular_velocity_rad_s
        energy+=h*mid.power_W
        drive.advance(h)
    return {'trajectory':rows,'controls':controls,'samples':samples,'diagnostics':{'max_force_residual_N':max_force,'max_torque_residual_N_m':max_torque},'config':asdict(config),'termination':termination}
