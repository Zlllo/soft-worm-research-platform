"""Frozen functional interventions, kept separate from trained model sources."""
import copy
import math

from .tracking_circuit import CircuitTrackingNetwork


def with_time_constant(model, tau_s):
    """Return an independent model with unchanged learned tensors and new tau.

    This changes the discrete rate relaxation coefficient only. The finite
    input history and preprocessing stay fixed; it is not a training ablation.
    """
    if not isinstance(model, CircuitTrackingNetwork):
        raise TypeError('time-constant intervention requires a circuit model')
    if isinstance(tau_s, bool) or not math.isfinite(tau_s) or tau_s <= 0:
        raise ValueError('tau must be finite and positive')
    result = copy.deepcopy(model)
    result.tau_s = float(tau_s)
    result.alpha = 1 - math.exp(-result.config.sensor_dt_s / result.tau_s)
    result.eval()
    return result
