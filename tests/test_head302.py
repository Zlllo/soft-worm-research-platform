import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from thermotaxis.head302 import HeadSteering, masks, neuron_graph

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "head302"))
from compare import CASES, STEPS, rollout, scan


def test_whole_graph_control_preserves_labelled_degrees_by_edge_type():
    graph = neuron_graph()
    raw = masks(graph, "real")
    shuffled = masks(graph, "rewired", 101)
    for real, null in zip(raw, shuffled):
        assert np.array_equal(real.sum(0), null.sum(0))
        assert np.array_equal(real.sum(1), null.sum(1))
        assert np.array_equal(real.diagonal(), null.diagonal())
        assert np.count_nonzero(real != null) > 1000
    assert np.array_equal(shuffled[1], shuffled[1].T)
    designs = {"real": raw, "rewired": shuffled, "dense": masks(graph, "dense")}
    for topology, (chemical, gap) in designs.items():
        model = HeadSteering(graph.names, chemical, gap)
        assert model.size == 302
        assert [graph.names[i] for i in model.inputs] == ["AFDL", "AFDR"]
        assert [graph.names[i] for i in model.outputs] == ["RIAL", "RIAR"]
        assert model.weight.numel() == (91204 if topology == "dense" else 3709)
        assert torch.allclose(model.electrical.sum(0), torch.zeros(302), atol=1e-6)


def test_local_scans_reverse_sides_under_mirror_and_ignore_uniform_offset():
    y, x = np.indices((40, 40))
    field = .4*x + .7*y
    px, py, heading = np.array([11.2]), np.array([14.3]), np.array([.8])
    noise = np.zeros((1, 3))
    observed = scan(field, px, py, heading, noise)
    reflected = scan(field[::-1] + 25., px, 39-py, -heading, noise)
    torch.testing.assert_close(observed, reflected.flip([1]))


def test_zero_turn_network_matches_blind_kinematics():
    graph = neuron_graph()
    model = HeadSteering(graph.names, *masks(graph, "real"))
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.zero_()
    field = np.hypot(*np.indices((40, 40)))
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        score, residence, trace = rollout(field, model)
    finally:
        torch.set_num_threads(threads)
    for actual, expected in zip((score, residence, trace), rollout(field, None)):
        assert actual == pytest.approx(expected, abs=1e-7)
    assert score.shape == residence.shape == (CASES,)
    assert trace.shape == (STEPS+1, 2)
    assert (trace >= 0).all() and (trace <= 39).all()
