TITLE Explicit phenomenological B-cell Ca/K currents (Xu2018 Eq5 family)

COMMENT
Engineering prototype, not an exact SI reconstruction or fitted B-cell physiology.
Total cell capacitance is configured separately to3pF. Conductances in nS,
membrane voltage in mV, current in nA, gate time constant in ms.
ENDCOMMENT

NEURON {
    POINT_PROCESS BMotorPoint
    NONSPECIFIC_CURRENT i
    RANGE gL, gCa, gK, EL, ECa, EK, midCa, widthCa, midK, widthK, tauK, i, n
}

UNITS { (nA) = (nanoamp) (mV) = (millivolt) (nS) = (nanosiemens) }

PARAMETER {
    gL = .1 (nS)
    gCa = .08 (nS)
    gK = .5 (nS)
    EL = -60 (mV)
    ECa = 40 (mV)
    EK = -80 (mV)
    midCa = -30 (mV)
    widthCa = 5 (mV)
    midK = -15 (mV)
    widthK = 5 (mV)
    tauK = 250 (ms)
}

ASSIGNED { v (mV) i (nA) m ninf }
STATE { n }

INITIAL { n = 1/(1+exp((midK-v)/widthK)) }

BREAKPOINT {
    SOLVE states METHOD cnexp
    m = 1/(1+exp((midCa-v)/widthCa))
    i = .001 * (gL*(v-EL)+gCa*m*(v-ECa)+gK*n*(v-EK))
}

DERIVATIVE states {
    ninf = 1/(1+exp((midK-v)/widthK))
    n' = (ninf-n)/tauK
}
