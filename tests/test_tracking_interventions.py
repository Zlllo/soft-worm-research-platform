import math

import pytest
import torch

from thermotaxis.connectome import CORE, load_connectome
from thermotaxis.tracking_circuit import CircuitTrackingNetwork
from thermotaxis.tracking_interventions import with_time_constant
from thermotaxis.tracking_learning import TrackingNetworkConfig


def circuit():
    torch.manual_seed(7)
    return CircuitTrackingNetwork(load_connectome().select(CORE),
        TrackingNetworkConfig(encoding='demeaned'), interface='full')


def test_intervention_preserves_all_learned_values_and_is_independent():
    original = circuit()
    altered = with_time_constant(original, .9)
    assert original.tau_s == .3
    assert altered.alpha == pytest.approx(1 - math.exp(-.1 / .9))
    for key, value in original.state_dict().items():
        assert torch.equal(value, altered.state_dict()[key])
    with torch.no_grad():
        altered.bias.add_(1)
    assert not torch.equal(original.bias, altered.bias)


def test_one_sample_obeys_relaxation_equation_and_padding_cannot_change_it():
    model = with_time_constant(circuit(), .1)
    x = torch.zeros(2, 41, 4)
    x[:, -1, :] = torch.tensor([.5, .7, -.2, 1.])
    injection = model.features(x)[:, -1, :4] @ model.input_weights
    expected = model.alpha * torch.tanh(model.bias + injection)
    assert torch.allclose(model.encode(x), expected, atol=1e-7)
    x[:, :-1, :3] = 1e6
    assert torch.allclose(model.encode(x), expected, atol=1e-7)


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf'), True])
def test_invalid_time_constant_is_rejected(value):
    with pytest.raises(ValueError):
        with_time_constant(circuit(), value)


def test_identity_intervention_reproduces_baseline_outputs_exactly():
    model = circuit().eval()
    identity = with_time_constant(model, .3)
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(4, 41, 4, generator=generator)
    x[:, :, 3] = 1
    with torch.no_grad():
        assert torch.equal(model(x), identity(x))
