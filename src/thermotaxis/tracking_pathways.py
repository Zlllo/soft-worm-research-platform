"""Explicit functional pathway masks without modifying archived rate models."""
import copy

import torch

from .connectome import load_connectome
from .tracking_circuit import CircuitTrackingNetwork


MODES = ('intact', 'no_chemical', 'no_gap', 'isolated')


class PathwayCircuit(CircuitTrackingNetwork):
    def __init__(self, *args, pathway_mode='intact', **kwargs):
        if pathway_mode not in MODES:
            raise ValueError('invalid pathway intervention')
        super().__init__(*args, **kwargs)
        self.pathway_mode = pathway_mode

    def effective_matrices(self):
        chemical, gap = super().effective_matrices()
        if self.pathway_mode == 'no_chemical':
            chemical = chemical * 0
        elif self.pathway_mode == 'isolated':
            # Preserve chemical self-loops and intrinsic leaky memory.
            chemical = chemical * torch.eye(self.size, device=chemical.device, dtype=chemical.dtype)
        if self.pathway_mode in ('no_gap', 'isolated'):
            gap = gap * 0
        return chemical, gap


def with_pathways(model, mode):
    if not isinstance(model, CircuitTrackingNetwork):
        raise TypeError('pathway intervention requires a circuit model')
    result = PathwayCircuit(load_connectome().select(model.names), config=model.config,
        topology=model.topology, tau_s=model.tau_s, interface=model.interface, pathway_mode=mode)
    result.load_state_dict(model.state_dict())
    result.rewiring = copy.deepcopy(model.rewiring)
    result.eval()
    return result
