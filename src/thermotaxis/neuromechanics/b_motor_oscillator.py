"""Explicit phenomenological B-cell Ca/K prototype, inspired by Xu2018 Eq5.

This is not a reconstruction of the inaccessible complete SI parameter set.
Only C=3pF and the100pS AVB coupling reference use verified corrected values;
other parameters are declared engineering choices, selected on isolated cells.
Units: mV, seconds, nS, nF and pA (nS*mV = pA = nF*mV/s).
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BMotorParameters:
    capacitance_nF: float = .003
    leak_nS: float = .1
    calcium_nS: float = .3
    potassium_nS: float = .5
    leak_reversal_mV: float = -60.
    calcium_reversal_mV: float = 40.
    potassium_reversal_mV: float = -80.
    calcium_midpoint_mV: float = -30.
    calcium_width_mV: float = 10.
    potassium_midpoint_mV: float = -25.
    potassium_width_mV: float = 10.
    potassium_tau_s: float = .25

    def as_dict(self):return asdict(self)


def sigmoid(v,mid,width):return 1/(1+np.exp(np.clip((mid-v)/width,-700,700)))


def rhs(t,y,p,*,gap_nS=.1,avb_mV=-20.,feedback_pA=0.):
    v,n=y
    m=sigmoid(v,p.calcium_midpoint_mV,p.calcium_width_mV)
    ninf=sigmoid(v,p.potassium_midpoint_mV,p.potassium_width_mV)
    current=(-p.leak_nS*(v-p.leak_reversal_mV)
        -p.calcium_nS*m*(v-p.calcium_reversal_mV)
        -p.potassium_nS*n*(v-p.potassium_reversal_mV)
        +gap_nS*(avb_mV-v)+feedback_pA)
    return np.array([current/p.capacitance_nF,(ninf-n)/p.potassium_tau_s])
