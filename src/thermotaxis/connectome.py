"""Versioned anatomical data and explicitly selected candidate subnetworks.

All adjacency arrays use [source, target]. Anatomical weights are EM-section
counts, NOT synaptic efficacies, signs, or measured conductances.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


CORE = ('AFDL', 'AFDR', 'AIYL', 'AIYR', 'AIZL', 'AIZR', 'RIAL', 'RIAR')
AFD_AIY = CORE[:4]
SENSORY_EXTENSION = CORE + ('AWCL', 'AWCR', 'ASIL', 'ASIR', 'AIAL', 'AIAR', 'AIBL', 'AIBR')
DATA_ROOT = Path(__file__).resolve().parents[2] / 'data' / 'connectome'


@dataclass
class Connectome:
    names: tuple
    chemical: np.ndarray
    gap: np.ndarray
    provenance: dict

    def select(self, names):
        names = tuple(names)
        if len(set(names)) != len(names):
            raise ValueError('duplicate selected cells')
        index = [self.names.index(n) for n in names]
        return Connectome(names, self.chemical[np.ix_(index, index)].copy(),
                          self.gap[np.ix_(index, index)].copy(), self.provenance.copy())

    def reachable(self, sources, targets):
        adjacency = (self.chemical + self.gap) > 0
        seen = {self.names.index(n) for n in sources}
        frontier = set(seen)
        while frontier:
            nxt = {int(j) for i in frontier for j in np.flatnonzero(adjacency[i])} - seen
            seen.update(nxt)
            frontier = nxt
        return {n: self.names.index(n) in seen for n in targets}

    def summary(self):
        return dict(cells=len(self.names), chemical_edges=int(np.count_nonzero(self.chemical)),
                    gap_directed_entries=int(np.count_nonzero(self.gap)),
                    gap_pairs=int(np.count_nonzero(np.triu(self.gap, 1))),
                    gap_self_entries=int(np.count_nonzero(np.diag(self.gap))),
                    chemical_weight=float(self.chemical.sum()), gap_weight=float(self.gap.sum()),
                    gap_symmetric=bool(np.array_equal(self.gap, self.gap.T)))


def load_connectome(path=DATA_ROOT / 'cook2019_herm.json'):
    path = Path(path)
    raw = path.read_bytes()
    d = json.loads(raw)
    names = tuple(d['nodes'])
    if len(set(names)) != len(names):
        raise ValueError('duplicate cell names')
    matrices = {k: np.zeros((len(names), len(names)), dtype=float)
                for k in ('chemical', 'gap_junction')}
    seen = set()
    for i, j, kind, weight in d['edges']:
        if not (0 <= i < len(names) and 0 <= j < len(names)):
            raise ValueError('invalid cell index')
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('anatomical weights must be finite and positive')
        if (i, j, kind) in seen:
            raise ValueError('duplicate edge')
        seen.add((i, j, kind))
        matrices[kind][i, j] = weight
    if not np.array_equal(matrices['gap_junction'], matrices['gap_junction'].T):
        raise ValueError('gap adjacency must be symmetric')
    provenance = dict(d['provenance'], derived_data_sha256=hashlib.sha256(raw).hexdigest())
    return Connectome(names, matrices['chemical'], matrices['gap_junction'], provenance)


def rewire_directed(mask, seed, swaps_per_edge=20):
    """Directed double-edge swaps preserve each labelled node's in/out degree.

    Self loops are held fixed; new self loops and duplicate edges are forbidden.
    This is a finite rewiring procedure, not a claim of uniform graph sampling.
    """
    mask = np.asarray(mask, dtype=bool)
    edges = set(zip(*np.nonzero(mask)))
    movable = sorted((i, j) for i, j in edges if i != j)
    rng = np.random.default_rng(seed)
    accepted = 0
    for _ in range(swaps_per_edge * len(movable)):
        if len(movable) < 2:
            break
        a, b = rng.choice(len(movable), 2, replace=False)
        i, j = movable[a]
        k, l = movable[b]
        if i == k or j == l or i == l or k == j or (i, l) in edges or (k, j) in edges:
            continue
        edges.remove((i, j)); edges.remove((k, l))
        edges.update(((i, l), (k, j)))
        movable[a], movable[b] = (i, l), (k, j)
        accepted += 1
    result = np.zeros_like(mask)
    for i, j in edges:
        result[i, j] = True
    return result, dict(accepted_swaps=accepted, changed_entries=int(np.count_nonzero(result != mask)))


def candidate_audit():
    full = load_connectome()
    neurons = json.loads((DATA_ROOT / 'herm_neurons.json').read_text())['names']
    candidates = {'afd_aiy4': AFD_AIY, 'classic8': CORE, 'sensory16': SENSORY_EXTENSION, 'all302': neurons}
    report = dict(provenance=full.provenance, full=full.summary(), candidates={})
    for label, names in candidates.items():
        graph = full.select(names)
        outputs = ('RIAL', 'RIAR') if 'RIAL' in names else ('AIYL', 'AIYR')
        report['candidates'][label] = dict(graph.summary(), names=list(names),
            afd_to_readout=graph.reachable(('AFDL', 'AFDR'), outputs))
    return report
