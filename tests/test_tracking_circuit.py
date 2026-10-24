import numpy as np
import pytest
import torch

from thermotaxis.connectome import CORE, load_connectome
from thermotaxis.tracking_learning import TrackingNetworkConfig, TrackingNetwork
from thermotaxis.tracking_circuit import CircuitTrackingNetwork


def window(length=41):
    x = torch.zeros(3, 41, 4)
    generator = torch.Generator().manual_seed(19)
    x[:, -length:, :3] = torch.randn(3, length, 3, generator=generator)
    x[:, -length:, 3] = 1
    return x


@pytest.mark.parametrize('interface', ['restricted', 'full'])
def test_padding_is_ignored_and_short_history_has_correct_dynamics(interface):
    graph = load_connectome().select(CORE)
    model = CircuitTrackingNetwork(graph, interface=interface)
    x = window(7)
    altered = x.clone()
    altered[:, :-7, :3] = 123456
    assert torch.equal(model(x), model(altered))
    short = CircuitTrackingNetwork(graph, config=TrackingNetworkConfig(window_samples=7), interface=interface)
    short.load_state_dict(model.state_dict())
    assert torch.allclose(model(x), short(x[:, -7:]), atol=1e-8)


def test_real_and_rewired_are_parameter_matched_with_shared_gap_and_initial_values():
    graph = load_connectome().select(CORE)
    torch.manual_seed(7)
    real = CircuitTrackingNetwork(graph)
    torch.manual_seed(7)
    rewired = CircuitTrackingNetwork(graph, topology='rewired', topology_seed=11)
    real_meta, rewired_meta = real.structural_metadata(), rewired.structural_metadata()
    a, b = np.array(real_meta['chemical_mask']), np.array(rewired_meta['chemical_mask'])
    assert np.array_equal(a.sum(0), b.sum(0))
    assert np.array_equal(a.sum(1), b.sum(1))
    assert np.array_equal(a.diagonal(), b.diagonal())
    assert not np.array_equal(a, b)
    assert real_meta['parameters'] == rewired_meta['parameters']
    assert torch.equal(real.effective_matrices()[1], rewired.effective_matrices()[1])
    for (_, p), (_, q) in zip(real.named_parameters(), rewired.named_parameters()):
        assert torch.equal(p, q)


def test_exact_shared_preprocessing_and_first_update_representation_gradients():
    graph = load_connectome().select(CORE)
    config = TrackingNetworkConfig(encoding='demeaned')
    model = CircuitTrackingNetwork(graph, config=config)
    generic = TrackingNetwork(config)
    x = window()
    assert torch.equal(model.features(x), generic.features(x))
    y = model(x)
    assert torch.all(y.abs() <= config.max_turn_rate_rad_s)
    (y - .1).square().mean().backward()
    assert model.chemical_weights.grad.abs().sum() > 0
    assert model.input_weights.grad.abs().sum() > 0
    assert model.bias.grad.abs().sum() > 0


def test_restricted_interface_sends_temperature_to_afd_and_phase_to_aiy():
    graph = load_connectome().select(CORE)
    model = CircuitTrackingNetwork(graph)
    baseline = torch.zeros(1, 1, 4)
    baseline[:, :, 3] = 1
    base = model.encode(baseline)
    temperature = baseline.clone(); temperature[:, :, 0] = 1
    phase = baseline.clone(); phase[:, :, 1] = 1
    temp_effect = model.encode(temperature) - base
    phase_effect = model.encode(phase) - base
    assert set(torch.nonzero(temp_effect[0]).flatten().tolist()) == {0, 1}
    assert set(torch.nonzero(phase_effect[0]).flatten().tolist()) == {2, 3}
    with pytest.raises(ValueError, match='empty'):
        model(torch.zeros(3, 41, 4))
