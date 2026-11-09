"""Read the published 49-rod WormSim output and reconstruct side geometry.

Coordinates are metres; the third rod variable is cross-section orientation,
not a world-heading command. The 48 dorsal/ventral actuators are effective
segments, not individually named anatomical muscle cells.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_body(path: str | Path):
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] != 148 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("expected finite time plus 49 rods with three variables each")
    time = data[:, 0]
    if time[0] < 0 or not np.all(np.diff(time) > 0):
        raise ValueError("timestamps must be nonnegative and strictly increasing")
    return time, data[:, 1:].reshape(-1, 49, 3)


def rod_radii():
    index = np.arange(49)
    return 40e-6 * np.sqrt(1 - ((index - 24) / 24.2) ** 2)


def side_lengths(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape[-2:] != (49, 3) or not np.isfinite(pose).all():
        raise ValueError("expected finite (...,49,3) rod poses")
    normal = np.stack([np.cos(pose[..., 2]), np.sin(pose[..., 2])], axis=-1)
    arm = rod_radii()[:, None] * normal
    dorsal = pose[..., :2] + arm
    ventral = pose[..., :2] - arm
    return np.stack([np.linalg.norm(np.diff(dorsal, axis=-2), axis=-1),
                     np.linalg.norm(np.diff(ventral, axis=-2), axis=-1)], axis=-1)


def curvature(pose):
    """Signed interior centerline curvature (m^-1), 47 sample positions."""
    xy = np.asarray(pose)[..., :2]
    tangent = np.diff(xy, axis=-2)
    length = np.linalg.norm(tangent, axis=-1)
    if np.any(length <= 0):
        raise ValueError("zero-length centerline segment")
    angle = np.unwrap(np.arctan2(tangent[..., 1], tangent[..., 0]), axis=-1)
    return np.diff(angle, axis=-1) / ((length[..., 1:] + length[..., :-1]) / 2)


def body_metrics(time, pose, warmup_s=3.):
    time, pose = np.asarray(time), np.asarray(pose)
    selected = time >= warmup_s - 1e-9
    if np.count_nonzero(selected) < 3:
        raise ValueError("insufficient post-warmup frames")
    t = time[selected]
    center = pose[selected, :, :2].mean(axis=1)
    displacement = np.diff(center, axis=0)
    lengths = np.linalg.norm(np.diff(pose[..., :2], axis=1), axis=2).sum(axis=1)
    k = curvature(pose)
    head_signal = k[selected, :5].mean(axis=1)
    spectrum = np.abs(np.fft.rfft(head_signal-head_signal.mean())) ** 2
    frequencies = np.fft.rfftfreq(len(t), d=float(np.median(np.diff(t))))
    frequency = None if np.std(head_signal) < 1e-6 else float(frequencies[1+np.argmax(spectrum[1:])])
    return dict(saved_end_s=float(time[-1]), frames=len(time), warmup_s=warmup_s,
                center_path_m=float(np.linalg.norm(displacement, axis=1).sum()),
                mean_center_speed_m_s=float(np.linalg.norm(displacement, axis=1).sum()/(t[-1]-t[0])),
                net_center_displacement_m=float(np.linalg.norm(center[-1]-center[0])),
                material_centerline_length_min_m=float(lengths.min()),
                material_centerline_length_max_m=float(lengths.max()),
                head_curvature_std_per_m=float(np.std(head_signal)),
                head_curvature_dominant_frequency_Hz=frequency,
                scope="geometric diagnostics of source model, not measured-animal calibration")
