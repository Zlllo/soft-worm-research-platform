import json
from pathlib import Path
import re

import numpy as np
import pytest
import torch

from thermotaxis.connectome import CORE, DATA_ROOT, candidate_audit, load_connectome, rewire_directed
from thermotaxis.connectome_policy import CircuitPolicy, GenericGRUPolicy


def test_import_matches_upstream_independent_spot_checks_and_totals():
    graph = load_connectome()
    raw = (DATA_ROOT / 'upstream_validation.yaml').read_text()
    sections = raw.split('  synapse: ')
    checks = 0
    for index, kind in enumerate(('Generic_CS', 'Generic_GJ')):
        matrix = graph.chemical if kind == 'Generic_CS' else graph.gap
        for pre, post, weight in re.findall(r'pre: (\S+)\n\s+post: (\S+)\n\s+weight: (\d+)', sections[index]):
            assert matrix[graph.names.index(pre), graph.names.index(post)] == int(weight)
            checks += 1
        totals = sections[index + 1]
        assert np.count_nonzero(matrix) == int(re.search(r'total_nonzero_conns: (\d+)', totals)[1])
        assert matrix.sum() == int(re.search(r'total_weight: (\d+)', totals)[1])
    assert checks == 19
    assert len(graph.names) == 473
    assert np.array_equal(graph.gap, graph.gap.T)
    assert graph.chemical[graph.names.index('AFDL'), graph.names.index('AIYL')] == 25
    assert graph.chemical[graph.names.index('AIYL'), graph.names.index('AFDL')] == 2


def test_neuronal_filter_excludes_muscles_and_retains_all_302():
    report = candidate_audit()
    assert report['candidates']['all302']['cells'] == 302
    assert 'MDL13' not in report['candidates']['all302']['names']
    assert all(report['candidates']['classic8']['afd_to_readout'].values())


def test_rewire_preserves_labelled_degrees_and_loops_and_changes_layout():
    mask = load_connectome().select(CORE).chemical > 0
    for seed in range(10):
        rewired, metadata = rewire_directed(mask, seed)
        assert np.array_equal(mask.sum(0), rewired.sum(0))
        assert np.array_equal(mask.sum(1), rewired.sum(1))
        assert np.array_equal(np.diag(mask), np.diag(rewired))
        assert metadata['changed_entries'] > 0
        assert np.array_equal(rewired, rewire_directed(mask, seed)[0])


def test_gap_is_symmetric_dissipative_and_missing_chemical_edges_have_no_parameters():
    graph = load_connectome().select(CORE)
    model = CircuitPolicy(graph)
    chemical, gap = model.effective_matrices()
    assert len(model.chemical_weights) == np.count_nonzero(graph.chemical)
    assert torch.all(chemical[torch.tensor(graph.chemical == 0)] == 0)
    assert torch.equal(gap, gap.T)
    assert torch.all(gap >= 0)
    h = torch.randn(20, len(CORE))
    diffusion = h @ gap - h * gap.sum(0)
    assert torch.allclose(diffusion.sum(1), torch.zeros(20), atol=1e-6)
    assert torch.all((diffusion * h).sum(1) <= 1e-6)


def test_temperature_only_enters_afd_and_rate_respects_physical_time():
    graph = load_connectome().select(CORE)
    torch.manual_seed(3)
    model = CircuitPolicy(graph, dt_s=.2)
    p, v, hidden = model(torch.tensor([1.]), torch.zeros(1, len(CORE)))
    assert set(torch.nonzero(hidden[0]).flatten().tolist()) == {0, 1}
    assert -np.expm1(-.2 * .01) <= p.item() <= -np.expm1(-.2 * .5)
    (p.sum() + v.sum() + hidden.sum()).backward()
    assert model.input_gain.grad is not None


def test_local_observation_does_not_disclose_heading_or_gradient():
    # A current reading cannot distinguish opposite headings at the same point.
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'analysis' / 'connectome_pilot.py'
    spec = importlib.util.spec_from_file_location('connectome_pilot', path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    task = module.Task(batch=4, noise_K=0)
    env = module.PointBatch(task, 3)
    before = env.observation()
    env.heading *= -1
    assert torch.equal(before, env.observation())
    env.x *= -1; env.gradient *= -1
    assert torch.equal(before, env.observation())


def test_generic_gru_capacity_and_temporal_memory():
    torch.manual_seed(3)
    model = GenericGRUPolicy()
    assert model.parameter_count() == 36
    initial = torch.zeros(1, 2)
    _, _, hot = model(torch.tensor([1.]), initial)
    _, _, cold = model(torch.tensor([-1.]), initial)
    _, _, first = model(torch.tensor([0.]), hot)
    _, _, second = model(torch.tensor([0.]), cold)
    assert not torch.equal(first, second)
