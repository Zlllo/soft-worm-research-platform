"""Causal rate-network tracking models for chemical-topology comparisons.

Only anatomical masks are used. All efficacies/signs are learned; fixed gap
topology is shared by real/rewired chemical graphs. Phase injection into AIY
and RIA readout are explicit engineering hypotheses, not reconstructed biology.
"""
import math

import numpy as np
import torch
from torch import nn

from .connectome import rewire_directed
from .tracking_learning import TrackingNetwork, TrackingNetworkConfig


class CircuitTrackingNetwork(TrackingNetwork):
    """Reuse the ordinary network's exact preprocessing and controller API."""

    def __init__(self, graph, config=None, topology='anatomical', topology_seed=0,
                 tau_s=.3, interface='restricted'):
        nn.Module.__init__(self)
        self.config = config or TrackingNetworkConfig(hidden_size=len(graph.names))
        if topology not in ('anatomical', 'rewired', 'dense'):
            raise ValueError('invalid chemical topology')
        if interface not in ('restricted', 'full'):
            raise ValueError('invalid input/output interface')
        if not math.isfinite(tau_s) or tau_s <= 0:
            raise ValueError('tau must be finite and positive')
        self.names = tuple(graph.names)
        self.size = len(self.names)
        self.topology = topology
        self.interface = interface
        self.tau_s = tau_s
        self.alpha = 1 - math.exp(-self.config.sensor_dt_s / tau_s)
        mask = graph.chemical > 0
        self.rewiring = None
        if topology == 'rewired':
            mask, self.rewiring = rewire_directed(mask, topology_seed)
        elif topology == 'dense':
            mask = np.ones_like(mask)
        pre, post = np.nonzero(mask)
        self.register_buffer('pre', torch.tensor(pre, dtype=torch.long))
        self.register_buffer('post', torch.tensor(post, dtype=torch.long))
        self.chemical_weights = nn.Parameter(torch.randn(len(pre)) / math.sqrt(self.size))
        gi, gj = np.nonzero(np.triu(graph.gap > 0, 1))
        self.register_buffer('gap_i', torch.tensor(gi, dtype=torch.long))
        self.register_buffer('gap_j', torch.tensor(gj, dtype=torch.long))
        self.gap_logits = nn.Parameter(torch.zeros(len(gi)))
        degree = np.sum(graph.gap > 0, axis=0)
        self.gap_scale = .2 / max(1, int(degree.max()))
        # Small random bias and nonzero readout avoid the prior exact symmetry.
        self.bias = nn.Parameter(.02 * torch.randn(self.size))
        if interface == 'restricted':
            inputs = [self.names.index(n) for n in ('AFDL', 'AFDR', 'AIYL', 'AIYR')]
            self.register_buffer('input_index', torch.tensor(inputs, dtype=torch.long))
            self.input_weights = nn.Parameter(.5 * torch.randn(4, 2))
            outputs = [self.names.index(n) for n in ('RIAL', 'RIAR')]
        else:
            self.input_weights = nn.Parameter(.5 * torch.randn(4, self.size))
            outputs = list(range(self.size))
        self.register_buffer('output_index', torch.tensor(outputs, dtype=torch.long))
        self.head = nn.Linear(len(outputs), 1)
        nn.init.normal_(self.head.weight, std=.02)
        nn.init.zeros_(self.head.bias)

    def effective_matrices(self):
        chemical = self.bias.new_zeros(self.size, self.size)
        chemical = chemical.index_put((self.pre, self.post), self.chemical_weights)
        gap = self.bias.new_zeros(self.size, self.size)
        conductance = self.gap_scale * torch.sigmoid(self.gap_logits)
        gap = gap.index_put((self.gap_i, self.gap_j), conductance)
        gap = gap.index_put((self.gap_j, self.gap_i), conductance)
        return chemical, gap

    def encode(self, windows):
        features = self.features(windows)
        if torch.any(features[:, :, 4].sum(1) < 1):
            raise ValueError('empty history window')
        chemical, gap = self.effective_matrices()
        if self.interface == 'restricted':
            temperature = features[:, :, :2] @ self.input_weights[:2].T
            phase = features[:, :, 2:4] @ self.input_weights[2:].T
            injection = features.new_zeros(*features.shape[:2], self.size)
            injection = injection.index_add(2, self.input_index, torch.cat((temperature, phase), dim=2))
        else:
            injection = features[:, :, :4] @ self.input_weights
        hidden = features.new_zeros(features.shape[0], self.size)
        for t in range(features.shape[1]):
            drive = hidden @ chemical + self.bias + injection[:, t]
            diffusion = hidden @ gap - hidden * gap.sum(0)
            updated = hidden + self.alpha * (torch.tanh(drive) - hidden + diffusion)
            hidden = torch.where(features[:, t, 4:5] > 0, updated, hidden)
        return hidden

    def forward(self, windows):
        hidden = self.encode(windows)
        return self.config.max_turn_rate_rad_s * torch.tanh(
            self.head(hidden[:, self.output_index]).squeeze(-1))

    def structural_metadata(self):
        chemical, gap = self.effective_matrices()
        mask = np.zeros((self.size, self.size), dtype=bool)
        mask[self.pre.cpu().numpy(), self.post.cpu().numpy()] = True
        return dict(names=list(self.names), topology=self.topology, interface=self.interface,
                    chemical_mask=mask.astype(int).tolist(), rewiring=self.rewiring,
                    gap_mask=(gap.detach().cpu().numpy() > 0).astype(int).tolist(),
                    parameters=sum(p.numel() for p in self.parameters()),
                    tau_s=self.tau_s, alpha=self.alpha,
                    temperature_injection='AFDL/R' if self.interface == 'restricted' else 'all nodes',
                    phase_injection='AIYL/R' if self.interface == 'restricted' else 'all nodes',
                    readout=[self.names[i] for i in self.output_index.cpu().tolist()])
