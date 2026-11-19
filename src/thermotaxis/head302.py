"""Topology-only head steering with the same sensory and readout ports for every RNN."""
import json

import numpy as np
import torch
from torch import nn

from .connectome import DATA_ROOT, load_connectome, rewire_directed


def neuron_graph():
    names = json.loads((DATA_ROOT / "herm_neurons.json").read_text())["names"]
    if len(names) != 302:
        raise ValueError("expected the 302-neuron hermaphrodite graph")
    return load_connectome().select(names)


def rewire_gap(mask, seed):
    """Undirected double-edge swaps, holding the labelled degrees and loops fixed."""
    rng = np.random.default_rng(seed)
    result = mask.copy()
    edges = np.argwhere(np.triu(mask, 1))
    for a, b in rng.integers(len(edges), size=(20 * len(edges), 2)):
        i, j = edges[a]
        k, l = edges[b]
        if rng.integers(2):
            k, l = l, k
        if len({i, j, k, l}) < 4 or result[i, l] or result[k, j]:
            continue
        result[i, j] = result[j, i] = result[k, l] = result[l, k] = False
        result[i, l] = result[l, i] = result[k, j] = result[j, k] = True
        edges[a], edges[b] = (i, l), (k, j)
    return result


def masks(graph, topology, graph_seed=0):
    chemical, gap = graph.chemical > 0, graph.gap > 0
    if topology == "dense":
        return np.ones_like(chemical), np.zeros_like(gap)
    if topology == "rewired":
        chemical, _ = rewire_directed(chemical, graph_seed)
        gap = rewire_gap(gap, graph_seed + 1)
    elif topology != "real":
        raise ValueError(topology)
    return chemical, gap


class HeadSteering(nn.Module):
    """Six recurrent updates per scan, then discard state before the next scan."""

    def __init__(self, names, chemical, gap):
        super().__init__()
        self.size = len(names)
        self.inputs = [names.index(name) for name in ("AFDL", "AFDR")]
        self.outputs = [names.index(name) for name in ("RIAL", "RIAR")]
        pre, post = np.nonzero(chemical)
        self.register_buffer("pre", torch.from_numpy(pre).long())
        self.register_buffer("post", torch.from_numpy(post).long())
        fan_in = torch.tensor(chemical.sum(0)[post], dtype=torch.float32)
        self.weight = nn.Parameter(torch.randn(len(pre)) / fan_in.sqrt())
        # Fixed symmetric electrical coupling; self-entries cancel in the Laplacian.
        conductance = gap.astype(np.float32)
        np.fill_diagonal(conductance, 0)
        degree = conductance.sum(0)
        laplacian = (conductance - np.diag(degree)) * (.2 / max(1., degree.max()))
        self.register_buffer("electrical", torch.from_numpy(laplacian))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.gain = nn.Parameter(torch.ones(2))
        self.readout = nn.Linear(2, 1)

    def recurrent_matrix(self):
        return self.electrical.index_put((self.pre, self.post), self.weight, accumulate=True)

    def forward(self, scan, matrix=None):
        if matrix is None:
            matrix = self.recurrent_matrix()
        drive = self.bias.expand(len(scan), -1).clone()
        drive[:, self.inputs] += scan * self.gain
        hidden = torch.tanh(drive)
        for _ in range(5):
            hidden = torch.tanh(hidden @ matrix + drive)
        return self.readout(hidden[:, self.outputs]).squeeze(1).tanh()
