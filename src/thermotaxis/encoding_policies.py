"""Matched sensory encoding controls, not reconstructed AFD cell physiology.

Both conditions retain signed temperature level. The adaptive condition adds
a causal signed fast-minus-slow temperature signal, scaled by 0.05 K. Raw
controls compute identical filter states but replace this channel by zero.
No absolute-error feature or teacher state enters these models.
"""
import math

import torch
from torch import nn

from .connectome_policy import CircuitPolicy


class FeatureCircuit(CircuitPolicy):
    """Same anatomical rate dynamics, with two input features to AFD only."""
    def __init__(self, graph, variant, topology_seed):
        super().__init__(graph, variant, topology_seed=topology_seed)
        # Rows are the two sensory cells; columns are level and auxiliary signal.
        self.input_gain = nn.Parameter(torch.ones(2, 2))
        del self.critic

    def forward(self, features, hidden):
        chemical, gap = self.effective_matrices()
        drive = hidden @ chemical + self.bias
        drive = drive.index_add(1, self.input_index, features @ self.input_gain.T)
        diffusion = hidden @ gap - hidden * gap.sum(0)
        hidden = hidden + self.alpha * (torch.tanh(drive) - hidden + diffusion)
        rate = .01 + .49 * torch.sigmoid(self.actor(hidden[:, self.output_index]).squeeze(-1))
        return -torch.expm1(-self.dt_s * rate), torch.zeros_like(rate), hidden


class FeatureGRU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.size = hidden_size
        self.memory = nn.GRUCell(2, hidden_size)
        self.actor = nn.Linear(hidden_size, 1)
        self.rewiring = None
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, features, hidden):
        hidden = self.memory(features, hidden)
        rate = .01 + .49 * torch.sigmoid(self.actor(hidden).squeeze(-1))
        return -torch.expm1(-rate), torch.zeros_like(rate), hidden


class EncodedPolicy(nn.Module):
    def __init__(self, core, encoding='raw', fast_tau_s=.5, slow_tau_s=5.):
        super().__init__()
        if encoding not in ('raw', 'adaptive'):
            raise ValueError('unknown sensory encoding')
        if not 0 < fast_tau_s < slow_tau_s:
            raise ValueError('require 0 < fast tau < slow tau')
        self.core = core
        self.encoding = encoding
        self.size = core.size + 3
        self.rewiring = core.rewiring
        self.register_buffer('alpha', torch.tensor([1-math.exp(-1/fast_tau_s), 1-math.exp(-1/slow_tau_s)]))

    def encode(self, temperature_error, filter_state):
        # Incoming observation is signed local temperature deviation / 1.5 K.
        signed_K = temperature_error * 1.5
        filtered = filter_state[:, :2] + self.alpha * (signed_K[:, None]-filter_state[:, :2])
        filtered = torch.where(filter_state[:, 2:3] > .5, filtered, signed_K[:, None].expand(-1, 2))
        adaptive = (filtered[:, 0]-filtered[:, 1])/.05
        if self.encoding == 'raw':
            adaptive = torch.zeros_like(adaptive)
        features = torch.stack((temperature_error, adaptive), dim=1)
        new_state = torch.cat((filtered, torch.ones_like(signed_K[:, None])), dim=1)
        return features, new_state

    def forward(self, temperature_error, hidden):
        features, filtered = self.encode(temperature_error, hidden[:, self.core.size:])
        p, value, neural = self.core(features, hidden[:, :self.core.size])
        return p, value, torch.cat((neural, filtered), dim=1)


ARCHITECTURES = ('anatomical', 'rewired', 'dense', 'gru2', 'gru16')


def make_encoded_policy(graph, architecture, encoding, topology_seed=0):
    if architecture in ('gru2', 'gru16'):
        core = FeatureGRU(2 if architecture == 'gru2' else 16)
    elif architecture in ('anatomical', 'rewired', 'dense'):
        core = FeatureCircuit(graph, architecture, topology_seed)
    else:
        raise ValueError('unknown architecture')
    return EncodedPolicy(core, encoding)
