"""Simplified anatomical controllers for controlled topology experiments.

These are trainable rate models, not reconstructed living neural dynamics.
Chemical signs/efficacies are learned. Gap junctions are tied, nonnegative,
diffusive conductances; their anatomical weights are not treated as efficacy.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

from .connectome import rewire_directed


class CircuitPolicy(nn.Module):
    def __init__(self, graph, variant='anatomical', topology_seed=0, dt_s=1., tau_s=2.):
        super().__init__()
        if variant not in ('anatomical', 'rewired', 'dense'):
            raise ValueError('unknown circuit variant')
        if dt_s <= 0 or tau_s <= 0:
            raise ValueError('time scales must be positive')
        self.names = graph.names
        self.size = len(graph.names)
        self.alpha = 1 - math.exp(-dt_s / tau_s)
        self.dt_s = dt_s
        mask = graph.chemical > 0
        self.rewiring = None
        if variant == 'rewired':
            mask, self.rewiring = rewire_directed(mask, topology_seed)
        elif variant == 'dense':
            mask = np.ones_like(mask)
        pre, post = np.nonzero(mask)
        self.register_buffer('pre', torch.tensor(pre, dtype=torch.long))
        self.register_buffer('post', torch.tensor(post, dtype=torch.long))
        self.chemical_weights = nn.Parameter(torch.randn(len(pre)) / math.sqrt(self.size))
        # Gap topology is identical across the primary chemical-topology ablation.
        gi, gj = np.nonzero(np.triu(graph.gap > 0, 1))
        self.register_buffer('gap_i', torch.tensor(gi, dtype=torch.long))
        self.register_buffer('gap_j', torch.tensor(gj, dtype=torch.long))
        self.gap_logits = nn.Parameter(torch.zeros(len(gi)))
        self.gap_scale = .2 / max(1, int(np.max(np.sum(graph.gap > 0, axis=0))))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.input_gain = nn.Parameter(torch.ones(2))
        self.register_buffer('input_index', torch.tensor([self.names.index(n) for n in ('AFDL', 'AFDR')]))
        outputs = ('RIAL', 'RIAR') if 'RIAL' in self.names else ('AIYL', 'AIYR')
        self.register_buffer('output_index', torch.tensor([self.names.index(n) for n in outputs]))
        self.actor = nn.Linear(2, 1)
        self.critic = nn.Linear(2, 1)
        nn.init.zeros_(self.actor.weight); nn.init.zeros_(self.actor.bias)

    def effective_matrices(self):
        chemical = self.bias.new_zeros(self.size, self.size)
        chemical = chemical.index_put((self.pre, self.post), self.chemical_weights)
        gap = self.bias.new_zeros(self.size, self.size)
        g = self.gap_scale * torch.sigmoid(self.gap_logits)
        gap = gap.index_put((self.gap_i, self.gap_j), g)
        gap = gap.index_put((self.gap_j, self.gap_i), g)
        return chemical, gap

    def forward(self, temperature_error, hidden):
        chemical, gap = self.effective_matrices()
        drive = hidden @ chemical + self.bias
        inputs = temperature_error[:, None] * self.input_gain[None, :]
        drive = drive.index_add(1, self.input_index, inputs)
        diffusion = hidden @ gap - hidden * gap.sum(dim=0)
        hidden = hidden + self.alpha * (torch.tanh(drive) - hidden + diffusion)
        readout = hidden[:, self.output_index]
        # Same bounded reversal rate for every architecture, converted to dt probability.
        rate = .01 + .49 * torch.sigmoid(self.actor(readout).squeeze(-1))
        probability = -torch.expm1(-self.dt_s * rate)
        return probability, self.critic(readout).squeeze(-1), hidden

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class GenericGRUPolicy(nn.Module):
    """Generic learned memory: two hidden units, 36 trainable parameters.

    No anatomical input/output bottlenecks. This is a capacity-near comparator,
    not an exact computational-cost or neuronal-state-count match.
    """
    def __init__(self, dt_s=1.):
        super().__init__()
        self.size = 2
        self.dt_s = dt_s
        self.rewiring = None
        self.memory = nn.GRUCell(1, self.size)
        self.actor = nn.Linear(self.size, 1)
        self.critic = nn.Linear(self.size, 1)
        nn.init.zeros_(self.actor.weight); nn.init.zeros_(self.actor.bias)

    def forward(self, temperature_error, hidden):
        hidden = self.memory(temperature_error[:, None], hidden)
        rate = .01 + .49 * torch.sigmoid(self.actor(hidden).squeeze(-1))
        return -torch.expm1(-self.dt_s * rate), self.critic(hidden).squeeze(-1), hidden

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
