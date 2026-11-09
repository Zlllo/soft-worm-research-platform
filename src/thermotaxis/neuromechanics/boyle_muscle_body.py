"""Direct 48x2 muscle-drive interface; legacy locomotor neurons are bypassed."""
import json

import numpy as np

from .boyle_body import BoyleBody


class BoyleMuscleBody(BoyleBody):
    def __init__(self, binary, **kwargs):
        super().__init__(binary, **kwargs)
        meta = json.loads(self.binary.with_name('build.json').read_text())
        if meta.get('control_mode') != 'external_muscle_drive':
            raise ValueError('use a separately built external-muscle worker')

    def step(self, microsteps=1, *, muscle_drive):
        if isinstance(microsteps, bool) or not isinstance(microsteps, (int, np.integer)) or not 1 <= microsteps <= 1000000:
            raise ValueError('microsteps must be an integer in [1,1000000]')
        values = np.asarray(muscle_drive, dtype=float)
        if values.shape != (48, 2) or not np.isfinite(values).all() or np.any(abs(values) > 1):
            raise ValueError('muscle_drive must be finite 48x2 values in [-1,1]')
        if self._process is None:
            raise RuntimeError('reset body before stepping')
        expected = self.state.microsteps + microsteps
        command = ' '.join(format(float(v), '.17g') for v in values.flat)
        try:
            self._process.stdin.write(f'MUSCLE {microsteps} {command}\n'.encode())
            state = self._receive()
            if state.microsteps != expected or abs(state.time_s - expected * self.dt_s) > 1e-10:
                raise RuntimeError('muscle worker clock did not match requested interval')
            if self.causal and state.internal_time_s > state.time_s + 1e-10:
                raise RuntimeError('solver integrated beyond muscle control clock')
        except Exception:
            self.close()
            raise
        self.state = state
        return state
