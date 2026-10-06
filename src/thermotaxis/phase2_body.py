"""Prescribed inextensible planar body in local linear drag; SI working model.

The reference point is the uniform material-arclength centre, not the first node.
This module has no temperature observation, contact or elastic constitutive law.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from functools import lru_cache

@dataclass(frozen=True)
class BodyConfig:
    schema_version: int = 1
    length_m: float = 1e-3
    drag_parallel_N_s_m2: float = 1e-3
    drag_ratio: float = 2.0
    waves: float = 1.0
    amplitude_rad: float = 0.6
    frequency_Hz: float = 1.0
    amplitude_modulation_rad: float = 0.0
    phase_modulation_rad: float = 0.0
    bias_rad: float = 0.0
    bias_modulation_rad: float = 0.0
    modulation_frequency_Hz: float = 0.25
    quadrature_order: int = 48
    integration_order: int = 32
    dt_s: float = 0.01
    duration_s: float = 4.0
    initial_orientation_rad: float = 0.0

    def __post_init__(self):
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version != 1:
            raise ValueError('schema_version must be 1')
        for key in ('quadrature_order', 'integration_order'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 4:
                raise ValueError(f'{key} must be integer >= 4')
        for key, value in asdict(self).items():
            if key not in ('schema_version', 'quadrature_order', 'integration_order'):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f'{key} must be finite numeric')
        for key in ('length_m', 'drag_parallel_N_s_m2', 'drag_ratio', 'waves', 'dt_s', 'duration_s'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        for key in ('frequency_Hz', 'modulation_frequency_Hz'):
            if getattr(self, key) < 0:
                raise ValueError(f'{key} must be nonnegative')
        if self.duration_s / self.dt_s < 1:
            raise ValueError('duration must be at least one time step')

    @classmethod
    def from_dict(cls, data):
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError('body config keys must exactly match schema')
        return cls(**data)

@dataclass(frozen=True)
class DriveState:
    amplitude_rad: float
    amplitude_rate_rad_s: float
    phase_rad: float
    phase_rate_rad_s: float
    bias_rad: float
    bias_rate_rad_s: float

@dataclass(frozen=True)
class Geometry:
    s_m: np.ndarray
    weights_m: np.ndarray | None
    position_m: np.ndarray
    deformation_velocity_m_s: np.ndarray
    tangent: np.ndarray

@dataclass(frozen=True)
class Motion:
    translation_m_s: np.ndarray
    angular_velocity_rad_s: float
    power_W: float
    force_N: np.ndarray
    torque_N_m: float
    geometry: Geometry
    material_velocity_m_s: np.ndarray

@lru_cache(maxsize=32)
def _gauss(order):
    return leggauss(order)

def drive_state(config: BodyConfig, time_s: float) -> DriveState:
    if not math.isfinite(time_s):
        raise ValueError('time must be finite')
    q = 2 * math.pi * config.modulation_frequency_Hz
    z = q * time_s
    return DriveState(config.amplitude_rad + config.amplitude_modulation_rad * math.sin(z),
                      config.amplitude_modulation_rad * q * math.cos(z),
                      2 * math.pi * config.frequency_Hz * time_s + config.phase_modulation_rad * math.sin(z),
                      2 * math.pi * config.frequency_Hz + config.phase_modulation_rad * q * math.cos(z),
                      config.bias_rad + config.bias_modulation_rad * math.sin(z),
                      config.bias_modulation_rad * q * math.cos(z))

def _angle(config, drive, s):
    z = 2 * math.pi * config.waves * s / config.length_m - drive.phase_rad
    a = drive.amplitude_rad * np.sin(z) + drive.bias_rad * (s / config.length_m - .5)
    adot = (drive.amplitude_rate_rad_s * np.sin(z) - drive.amplitude_rad * drive.phase_rate_rad_s * np.cos(z)
            + drive.bias_rate_rad_s * (s / config.length_m - .5))
    return a, adot

def _uncentred_geometry(config, drive, s):
    nodes, weights = _gauss(config.integration_order)
    samples = s[:, None] * (nodes[None, :] + 1) / 2
    a, adot = _angle(config, drive, samples)
    tangents = np.stack((np.cos(a), np.sin(a)), axis=-1)
    tangent_rates = np.stack((-np.sin(a) * adot, np.cos(a) * adot), axis=-1)
    position = np.einsum('j,ijk,i->ik', weights, tangents, s / 2)
    velocity = np.einsum('j,ijk,i->ik', weights, tangent_rates, s / 2)
    a_at_s, _ = _angle(config, drive, s)
    return position, velocity, np.stack((np.cos(a_at_s), np.sin(a_at_s)), axis=-1)

def shape_geometry(config: BodyConfig, time_s: float, s_m=None, drive: DriveState | None = None) -> Geometry:
    drive = drive_state(config, time_s) if drive is None else drive
    if any(not math.isfinite(x) for x in asdict(drive).values()):
        raise ValueError('drive state must be finite')
    nodes, weights = _gauss(config.quadrature_order)
    quadrature_s = (nodes + 1) * config.length_m / 2
    weights_m = weights * config.length_m / 2
    reference_r, reference_v, reference_t = _uncentred_geometry(config, drive, quadrature_s)
    centre = weights_m @ reference_r / config.length_m
    centre_rate = weights_m @ reference_v / config.length_m
    if s_m is None:
        s, r, v, tangent = quadrature_s, reference_r, reference_v, reference_t
    else:
        s = np.asarray(s_m, dtype=float)
        if s.ndim != 1 or not np.all(np.isfinite(s)) or np.any(s < 0) or np.any(s > config.length_m):
            raise ValueError('s_m must be one-dimensional finite material coordinates in [0,L]')
        r, v, tangent = _uncentred_geometry(config, drive, s)
        weights_m = None
    return Geometry(s, weights_m, r - centre, v - centre_rate, tangent)

def solve_motion(config: BodyConfig, time_s: float, orientation_rad: float = 0., drive=None) -> Motion:
    if not math.isfinite(orientation_rad):
        raise ValueError('orientation must be finite')
    g = shape_geometry(config, time_s, drive=drive)
    c, sn = math.cos(orientation_rad), math.sin(orientation_rad)
    rotation = np.array([[c, -sn], [sn, c]])
    r, vshape, tangent = (x @ rotation.T for x in (g.position_m, g.deformation_velocity_m_s, g.tangent))
    g = Geometry(g.s_m, g.weights_m, r, vshape, tangent)
    # D is a positive resistance tensor; actual medium force is -D v.
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    D = config.drag_parallel_N_s_m2 * (np.einsum('ni,nj->nij', tangent, tangent)
           + config.drag_ratio * np.einsum('ni,nj->nij', normal, normal))
    basis = np.zeros((len(r), 2, 3))
    basis[:, 0, 0], basis[:, 1, 1] = 1, 1
    # Scale angular column by L so all solve unknowns have velocity units.
    basis[:, :, 2] = np.stack((-r[:, 1], r[:, 0]), axis=1) / config.length_m
    matrix = np.einsum('n,nij,nik,nkl->jl', g.weights_m, basis, D, basis)
    rhs = -np.einsum('n,nij,nik,nk->j', g.weights_m, basis, D, vshape)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        raise ValueError('nonfinite resistance system')
    eigenvalues = np.linalg.eigvalsh(matrix)
    condition = np.linalg.cond(matrix)
    if eigenvalues[0] <= 0 or not np.isfinite(condition) or condition > 1e12:
        raise ValueError('ill-conditioned scaled resistance system; refine geometry or parameters')
    solution = np.linalg.solve(matrix, rhs)
    v = vshape + np.einsum('nij,j->ni', basis, solution)
    force_density = -np.einsum('nij,nj->ni', D, v)
    force = g.weights_m @ force_density
    torque = float(g.weights_m @ (r[:, 0] * force_density[:, 1] - r[:, 1] * force_density[:, 0]))
    power = float(np.einsum('n,ni,nij,nj->', g.weights_m, v, D, v))
    if not np.all(np.isfinite(solution)) or not math.isfinite(power) or power < 0:
        raise ValueError('nonfinite motion or invalid dissipation')
    return Motion(solution[:2], float(solution[2] / config.length_m), power, force, torque, g, v)

def simulate(config: BodyConfig):
    """Explicit midpoint pose/energy integration; exact final-time remainder."""
    count = int(math.ceil(config.duration_s / config.dt_s))
    pose = np.array([0., 0., config.initial_orientation_rad])
    elapsed, energy = 0., 0.
    rows = [[elapsed, *pose, energy, solve_motion(config, 0., pose[2]).power_W]]
    for step in range(count):
        h = min(config.dt_s, config.duration_s - elapsed)
        if h <= 0:
            break
        start = solve_motion(config, elapsed, pose[2])
        middle = solve_motion(config, elapsed + h / 2, pose[2] + h * start.angular_velocity_rad_s / 2)
        pose[:2] += h * middle.translation_m_s
        pose[2] += h * middle.angular_velocity_rad_s
        energy += h * middle.power_W
        elapsed = min(config.duration_s, elapsed + h)
        rows.append([elapsed, *pose, energy, solve_motion(config, elapsed, pose[2]).power_W])
    return np.asarray(rows)
