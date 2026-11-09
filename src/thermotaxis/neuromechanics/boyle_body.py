"""Persistent process adapter for the published Boyle/Berri/Cohen body.

Build with analysis/neuromechanics/build_boyle_worker.py. Coordinates are m,
time is s. This adapter exposes effective drive, not anatomical synapses.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time

import numpy as np

from .boyle_output import curvature


@dataclass(frozen=True)
class BoyleState:
    time_s: float
    microsteps: int
    internal_time_s: float
    pose: np.ndarray
    velocity: np.ndarray
    neural: np.ndarray
    muscle: np.ndarray
    stretch: np.ndarray
    side_length: np.ndarray

    @property
    def head_xy_m(self):
        return self.pose[0, :2].copy()

    @property
    def signed_speed_m_s(self):
        axis = self.pose[0, :2] - self.pose[-1, :2]
        return float(self.velocity[:, :2].mean(axis=0) @ axis / np.linalg.norm(axis))

    @property
    def head_curvature_per_m(self):
        return float(curvature(self.pose)[:5].mean())


class BoyleBody:
    """A resettable worker with a fixed 1 ms neuro/muscle clock.

    Each new control affects the neural update after the next physical tick.
    ``causal=False`` retains the upstream integrator's possible lookahead for
    explicit compatibility comparisons. Use the default for new experiments.
    Only initial reset applies a rigid pose transform; step never sets heading.
    """

    def __init__(self, binary: str | Path, *, causal=True, timeout_s=30.):
        self.binary = Path(binary).resolve()
        meta = json.loads(self.binary.with_name("build.json").read_text())
        if hashlib.sha256(self.binary.read_bytes()).hexdigest() != meta["binary_sha256"]:
            raise ValueError("worker binary does not match build metadata")
        self.dt_s = float(meta["dt_s"])
        self.causal = bool(causal)
        self.timeout_s = float(timeout_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout must be positive and finite")
        self._process = None
        self._stderr = None
        self._buffer = b""
        self.state = None

    def reset(self, *, head_xy_m=(0., 0.), rotation_rad=0.):
        xy = np.asarray(head_xy_m, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or not math.isfinite(rotation_rad):
            raise ValueError("initial pose must be finite")
        self.close()
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen([str(self.binary), *map(str, xy), str(rotation_rad), str(int(self.causal))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr, bufsize=0)
        self._buffer = b""
        try:
            self.state = self._receive()
        except Exception:
            self.close()
            raise
        return self.state

    def _receive(self):
        deadline = time.monotonic() + self.timeout_s
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self._process.stdout], [], [], max(0, remaining))[0]:
                raise TimeoutError("body worker response timed out")
            chunk = os.read(self._process.stdout.fileno(), 65536)
            if not chunk:
                self._stderr.seek(0)
                raise RuntimeError("body worker stopped: " + self._stderr.read()[-2000:].decode(errors="replace"))
            self._buffer += chunk
            if len(self._buffer) > 200000:
                raise RuntimeError("oversize body worker response")
        line, self._buffer = self._buffer.split(b"\n", 1)
        parts = line.split()
        if len(parts) != 682 or parts[0] != b"OK":
            raise RuntimeError("invalid worker packet: " + line[:120].decode(errors="replace"))
        values = np.array([float(v) for v in parts[1:]])
        if not np.isfinite(values).all() or values[1] < 0 or values[1] != int(values[1]):
            raise RuntimeError("invalid worker state")
        arrays = [values[3:150].reshape(49, 3), values[150:297].reshape(49, 3)]
        arrays += [values[297+i*96:393+i*96].reshape(48, 2) for i in range(4)]
        for array in arrays:
            array.setflags(write=False)
        return BoyleState(float(values[0]), int(values[1]), float(values[2]), *arrays)

    def step(self, microsteps=1, *, drive=.675, head_bias=0.):
        if isinstance(microsteps, bool) or not isinstance(microsteps, (int, np.integer)) or not 1 <= microsteps <= 1000000:
            raise ValueError("microsteps must be an integer in [1,1000000]")
        if not math.isfinite(drive) or not 0 <= drive <= 1 or not math.isfinite(head_bias) or abs(head_bias) > .1:
            raise ValueError("drive must be in [0,1], head bias in [-.1,.1]")
        if self._process is None:
            raise RuntimeError("reset body before stepping")
        expected = self.state.microsteps + microsteps
        try:
            self._process.stdin.write(f"STEP {microsteps} {drive:.17g} {head_bias:.17g}\n".encode())
            state = self._receive()
            if state.microsteps != expected or abs(state.time_s-expected*self.dt_s) > 1e-10:
                raise RuntimeError("worker clock did not match requested interval")
            if self.causal and state.internal_time_s > state.time_s + 1e-10:
                raise RuntimeError("solver integrated beyond control clock")
        except Exception:
            self.close()
            raise
        self.state = state
        return state

    def advance(self, duration_s, **controls):
        count = self.microstep_count(duration_s)
        return self.step(count, **controls)

    def microstep_count(self, duration_s):
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration must be positive and finite")
        count = round(duration_s/self.dt_s)
        if count < 1 or abs(count*self.dt_s-duration_s) > 1e-10:
            raise ValueError("duration must be a whole number of 1 ms ticks")
        return count

    def close(self):
        if self._process is not None:
            process, self._process = self._process, None
            if process.poll() is None:
                try:
                    process.stdin.write(b"QUIT\n")
                    process.wait(timeout=1)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait()
            process.stdin.close()
            process.stdout.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        self.state = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
