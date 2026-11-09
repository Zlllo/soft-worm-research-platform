"""One-sided excitation avoids unbounded hyperpolarizing current clamps.

This replaces the signed-current prototype after its negative currents drove
generic C1 cells below the declared operating range. Spatial fields are kept.
"""
import numpy as np

from .full302_feedback import BendingFeedback


class ExcitatoryBendingFeedback(BendingFeedback):
    def currents(self, actual_side_lengths, relaxed_side_lengths):
        return np.maximum(0.,super().currents(actual_side_lengths,relaxed_side_lengths))
