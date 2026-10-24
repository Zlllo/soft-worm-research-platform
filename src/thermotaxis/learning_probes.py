"""Trainability probes; explicit error filters are engineering positive controls.

The filter controls use the known task preference and absolute error. They are
NOT models of AFD sensory adaptation and are not connectome-efficiency evidence.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class MemoryProbePolicy(nn.Module):
    def __init__(self, hidden_size=16, absolute_input=False, dt_s=1.):
        super().__init__()
        self.size = hidden_size
        self.absolute_input = absolute_input
        self.dt_s = dt_s
        self.memory = nn.GRUCell(1, hidden_size)
        self.actor = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, temperature_error, hidden):
        obs = temperature_error.abs() if self.absolute_input else temperature_error
        hidden = self.memory(obs[:, None], hidden)
        rate = .01 + .49 * torch.sigmoid(self.actor(hidden).squeeze(-1))
        return -torch.expm1(-self.dt_s * rate), torch.zeros_like(rate), hidden


class ErrorFilterProbePolicy(nn.Module):
    """Two known error filters plus a fitted linear rate readout (2 parameters).

    With learn_taus=True, two additional time constants are learned, initialized
    to .8 and 3 seconds (teacher uses .5 and 5). The third hidden entry is an
    initialization flag, never an extra observation or teacher state.
    """
    def __init__(self, learn_taus=False, dt_s=1.):
        super().__init__()
        self.size = 3
        self.dt_s = dt_s
        self.bias = nn.Parameter(torch.tensor(.255))
        self.gain = nn.Parameter(torch.tensor(0.))
        initial = (.8, 3.) if learn_taus else (.5, 5.)
        raw = torch.tensor([math.log(math.expm1(t - .05)) for t in initial])
        if learn_taus:
            self.raw_taus = nn.Parameter(raw)
        else:
            self.register_buffer('raw_taus', raw)

    @property
    def taus_s(self):
        return .05 + F.softplus(self.raw_taus)

    def forward(self, temperature_error, hidden):
        error_K = temperature_error.abs() * 1.5
        alpha = -torch.expm1(-self.dt_s / self.taus_s)
        filters = hidden[:, :2] + alpha * (error_K[:, None] - hidden[:, :2])
        filters = torch.where(hidden[:, 2:3] > .5, filters, error_K[:, None].expand(-1, 2))
        hidden = torch.cat((filters, torch.ones_like(error_K[:, None])), dim=1)
        # Divide the history difference by .05 K to put its scale near unity.
        feature = (filters[:, 1] - filters[:, 0]) / .05
        rate = (self.bias + self.gain * feature).clamp(.01, .5)
        return -torch.expm1(-self.dt_s * rate), torch.zeros_like(rate), hidden


PROBES = ('signed_gru2', 'signed_gru16', 'absolute_gru16', 'fixed_error_filters', 'learned_error_filters')


def make_probe(name):
    if name == 'signed_gru2':
        return MemoryProbePolicy(2)
    if name == 'signed_gru16':
        return MemoryProbePolicy(16)
    if name == 'absolute_gru16':
        return MemoryProbePolicy(16, absolute_input=True)
    if name == 'fixed_error_filters':
        return ErrorFilterProbePolicy()
    if name == 'learned_error_filters':
        return ErrorFilterProbePolicy(learn_taus=True)
    raise ValueError('unknown probe')
