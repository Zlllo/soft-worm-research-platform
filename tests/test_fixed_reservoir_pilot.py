"""Core causal and graph checks for the fixed reservoir pilot."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from fixed_reservoir_pilot import simulate, graph_and_weights
from thermotaxis.connectome import rewire_directed


def test_no_future_leak_and_no_mutation():
    graph = np.array([[0,1],[1,0]], dtype=bool)
    W = graph_and_weights(graph, 7, 2)
    original = W.copy()
    gap = np.zeros((2,2)); inputs = np.array([1.,0.])
    a = np.zeros((1,20)); b = a.copy(); b[0,10:] = 1
    out_a = simulate(a,W,gap,inputs,.1)
    out_b = simulate(b,W,gap,inputs,.1)
    np.testing.assert_array_equal(out_a[:,:10],out_b[:,:10])
    assert np.any(out_a[:,10:] != out_b[:,10:])
    np.testing.assert_array_equal(W,original)


def test_single_node_update_and_rewiring_degrees():
    u = np.ones((1,1)); W = np.zeros((1,1)); gap = np.zeros((1,1))
    out = simulate(u,W,gap,np.ones(1),.1)
    np.testing.assert_allclose(out[0,0,0], np.tanh(1.) / 3)
    graph = np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]],dtype=bool)
    shuffled,_ = rewire_directed(graph, 12)
    np.testing.assert_array_equal(graph.sum(0),shuffled.sum(0))
    np.testing.assert_array_equal(graph.sum(1),shuffled.sum(1))
