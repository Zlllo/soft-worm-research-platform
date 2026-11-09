"""Causal scalar thermal encoding and declared B-cell mechanical feedback.

These are explicit phenomenological interfaces, not measured receptor fields.
The controller side receives only local temperature and accepted side lengths.
"""
from dataclasses import dataclass
import math

import numpy as np

from thermotaxis.sensory_adaptation import ThresholdAdaptation
from .anatomical_body import load_anatomical_body_wiring, PlanarMuscleProjection


@dataclass
class LocalAFD:
    initial_temperature_K: float
    response_width_K: float = .05
    filter_tau_s: float = .1
    maximum_current_pA: float = 5.

    def __post_init__(self):
        if not math.isfinite(self.initial_temperature_K) or any(not math.isfinite(v) or v<=0 for v in
                (self.response_width_K,self.filter_tau_s,self.maximum_current_pA)):
            raise ValueError('finite temperature and positive sensory scales required')
        self.model=ThresholdAdaptation(response_width_K=self.response_width_K)
        self.filtered_K=float(self.initial_temperature_K)
        self.threshold_K=float(self.initial_temperature_K)

    def advance(self, temperature_K, dt_s):
        if not math.isfinite(temperature_K) or not math.isfinite(dt_s) or dt_s<=0:
            raise ValueError('finite local temperature and positive dt required')
        self.filtered_K+=-math.expm1(-dt_s/self.filter_tau_s)*(temperature_K-self.filtered_K)
        current=self.maximum_current_pA*float(self.model.response(self.filtered_K,self.threshold_K))
        self.threshold_K=float(self.model.update(self.filtered_K,self.threshold_K,dt_s))
        return current


@dataclass(frozen=True)
class BendingFeedback:
    names: tuple[str,...]
    sides: np.ndarray
    weights: np.ndarray
    nominal_output_centers: np.ndarray
    gain_pA_per_strain: float = 500.
    maximum_current_pA: float = 5.

    @classmethod
    def anatomical_bins(cls):
        w=load_anatomical_body_wiring();projection=PlanarMuscleProjection.rank_bins(w.muscle_names)
        names=tuple(n for n in w.neuron_names if n.startswith(('DB','VB')) and n[2:].isdigit())
        assert len(names)==18
        sides=np.array([0 if n.startswith('DB') else 1 for n in names])
        weights=np.zeros((18,48));centers=[]
        for i,(name,side) in enumerate(zip(names,sides)):
            footprint=((w.nmj_chemical[w.neuron_names.index(name)]>0)@projection.matrix).reshape(48,2)[:,side]
            if footprint.sum()<=0:
                raise ValueError('B-cell has no ipsilateral muscle footprint: '+name)
            center=float(np.arange(48)@footprint/footprint.sum());centers.append(center)
            # Nominal preceding ~200um of a1mm/48-segment body; not receptor anatomy.
            end=max(1,int(np.floor(center)))
            start=max(0,end-10)
            weights[i,start:end]=1/(end-start)
        for a in (sides,weights):a.setflags(write=False)
        return cls(names,sides,weights,np.array(centers))

    def currents(self, actual_side_lengths, relaxed_side_lengths):
        actual,rest=np.asarray(actual_side_lengths),np.asarray(relaxed_side_lengths)
        if actual.shape!=(48,2) or rest.shape!=(48,2) or not np.isfinite(actual).all() or not np.isfinite(rest).all() or np.any(rest<=0):
            raise ValueError('finite actual/relaxed 48x2 lengths and positive rest required')
        strain=(actual-rest)/rest
        # Positive on the compressed/inside side; uniform extension cancels.
        bending=(strain[:,1]-strain[:,0])/2
        signed=np.where(self.sides==0,1.,-1.)*(self.weights@bending)
        return np.clip(self.gain_pA_per_strain*signed,-self.maximum_current_pA,self.maximum_current_pA)
