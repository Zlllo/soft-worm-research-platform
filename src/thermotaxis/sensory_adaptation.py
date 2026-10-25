"""A phenomenological AFD-threshold module, separate from preference memory.

Threshold relaxation times 45 s (warming) and 840 s (cooling) follow the fits
in Hawk et al. 2018 Fig1J. Rectification width and response amplitude below are
working values, not fitted calcium traces or measured channel kinetics.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ThresholdAdaptation:
    warming_tau_s: float = 45.
    cooling_tau_s: float = 840.
    response_width_K: float = .05

    def __post_init__(self):
        if any(not math.isfinite(x) or x<=0 for x in (self.warming_tau_s,self.cooling_tau_s,self.response_width_K)):
            raise ValueError('positive finite adaptation scales required')

    def update(self,temperature_K,threshold_K,dt_s):
        if not math.isfinite(dt_s) or dt_s<=0:raise ValueError('positive dt required')
        t=np.asarray(temperature_K);b=np.asarray(threshold_K)
        tau=np.where(t>=b,self.warming_tau_s,self.cooling_tau_s)
        return b+(-np.expm1(-dt_s/tau))*(t-b)

    def response(self,temperature_K,threshold_K):
        return np.clip((np.asarray(temperature_K)-np.asarray(threshold_K))/self.response_width_K,0.,1.)
