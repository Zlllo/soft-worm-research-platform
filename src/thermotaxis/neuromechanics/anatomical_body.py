"""Named full-neuron/body-wall wiring and an explicit, lossy planar projection.

EM counts are anatomical data, never automatically interpreted as efficacy.
No neuronal dynamics or neurotransmitter signs are assigned by this module.
"""
from dataclasses import dataclass
import json

import numpy as np

from thermotaxis.connectome import DATA_ROOT, load_connectome
from .muscle_mapping import body_wall_slots


@dataclass(frozen=True)
class AnatomicalBodyWiring:
    neuron_names: tuple[str, ...]
    muscle_names: tuple[str, ...]
    neuron_chemical: np.ndarray
    neuron_gap: np.ndarray
    nmj_chemical: np.ndarray
    neuron_muscle_gap: np.ndarray
    muscle_chemical: np.ndarray
    muscle_gap: np.ndarray
    provenance: dict

    def chemical_muscle_input(self, activity, *, efficacy):
        """Apply explicitly supplied [neuron,muscle] signed efficacies on EM mask.

        This algebraic test interface does not turn EM counts into conductance.
        It omits NMJ electrical coupling, which needs a muscle-voltage model.
        """
        activity, efficacy = np.asarray(activity, dtype=float), np.asarray(efficacy, dtype=float)
        if activity.ndim < 1 or activity.shape[-1] != len(self.neuron_names):
            raise ValueError('activity must follow all named neurons')
        if efficacy.shape != self.nmj_chemical.shape:
            raise ValueError('efficacy must follow [neuron,muscle]')
        if not np.isfinite(activity).all() or not np.isfinite(efficacy).all():
            raise ValueError('activity and efficacy must be finite')
        if np.any(efficacy[self.nmj_chemical == 0] != 0):
            raise ValueError('efficacy creates a connection absent from the anatomical mask')
        return activity @ efficacy


def load_anatomical_body_wiring():
    graph = load_connectome()
    neurons = tuple(json.loads((DATA_ROOT / 'herm_neurons.json').read_text())['names'])
    muscles = tuple(n for n in graph.names if n in body_wall_slots())
    if len(neurons) != 302 or len(muscles) != 95:
        raise ValueError('unexpected anatomical population')
    ni, mi = [graph.names.index(n) for n in neurons], [graph.names.index(n) for n in muscles]
    arrays = [graph.chemical[np.ix_(ni, ni)], graph.gap[np.ix_(ni, ni)],
              graph.chemical[np.ix_(ni, mi)], graph.gap[np.ix_(ni, mi)],
              graph.chemical[np.ix_(mi, mi)], graph.gap[np.ix_(mi, mi)]]
    for a in arrays:
        a.setflags(write=False)
    return AnatomicalBodyWiring(neurons, muscles, *arrays, graph.provenance)


@dataclass(frozen=True)
class PlanarMuscleProjection:
    """Provisional rank-bin reduction, not measured muscle locations/geometry.

    Anatomical L/R counterparts share a nominal longitudinal bin. Average
    the two quadrant slots, then hold on two of the 48 physical segments.
    Missing MVL24 contributes zero, with fixed denominator two. This loses
    L/R differences and has rank 48; it must not be used to study torsion.
    """
    muscle_names: tuple[str, ...]
    matrix: np.ndarray

    @classmethod
    def rank_bins(cls, muscle_names):
        names = tuple(muscle_names)
        expected = set(body_wall_slots()) - {'MVL24'}
        if len(names) != 95 or set(names) != expected:
            raise ValueError('projection requires the named 95 body-wall muscles')
        matrix = np.zeros((95, 96))
        for i, name in enumerate(names):
            side = 0 if name[1] == 'D' else 1
            segment = 2 * (int(name[3:]) - 1)
            matrix[i, 2 * segment + side] = .5
            matrix[i, 2 * (segment + 1) + side] = .5
        matrix.setflags(write=False)
        return cls(names, matrix)

    def to_segments(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim < 1 or values.shape[-1] != 95 or not np.isfinite(values).all():
            raise ValueError('provide finite drives following muscle_names')
        return (values @ self.matrix).reshape(*values.shape[:-1], 48, 2)
