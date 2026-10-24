import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from thermotaxis.learning_probes import ErrorFilterProbePolicy, MemoryProbePolicy


def load_analysis(name):
    analysis = Path(__file__).resolve().parents[1] / 'analysis'
    # These scripts reuse each other's explicitly declared task, not private data.
    if str(analysis) not in sys.path:
        sys.path.insert(0, str(analysis))
    spec = importlib.util.spec_from_file_location(name, analysis / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_error_filter_class_can_exactly_express_teacher_without_teacher_states():
    script = load_analysis('connectome_memory_probe')
    from connectome_pilot import Task
    x, target = script.teacher_data(Task(batch=8, steps=180), 82004)
    model = ErrorFilterProbePolicy()
    with torch.no_grad():
        model.bias.fill_(.1)
        model.gain.fill_(-.2)
        predicted = script.predict(model, x)
    assert torch.allclose(predicted, target, atol=2e-6, rtol=2e-5)


def test_absolute_error_model_is_invariant_to_temperature_polarity_and_resets():
    model = ErrorFilterProbePolicy()
    with torch.no_grad():
        model.bias.fill_(.1)
        model.gain.fill_(-.2)
    series = torch.tensor([.8, .6, .4, .5, .7])
    states = [torch.zeros(1, model.size), torch.zeros(1, model.size)]
    for value in series:
        p, _, states[0] = model(value[None], states[0])
        q, _, states[1] = model(-value[None], states[1])
        assert torch.equal(p, q)
    _, _, reset = model(torch.tensor([.2]), torch.zeros(1, model.size))
    assert reset[0, 0] == pytest.approx(.3)
    assert reset[0, 1] == pytest.approx(.3)


def test_learned_filter_timescales_positive_and_receive_history_gradients():
    model = ErrorFilterProbePolicy(learn_taus=True)
    assert model.taus_s.tolist() == pytest.approx([.8, 3.])
    with torch.no_grad():
        model.gain.fill_(-.05)
    h = torch.zeros(1, model.size)
    for value in [.8, .7, .6, .62]:
        p, _, h = model(torch.tensor([value]), h)
    p.sum().backward()
    assert torch.all(torch.isfinite(model.raw_taus.grad))
    assert torch.any(model.raw_taus.grad != 0)


def test_capacity_and_input_ablation_keep_parameter_counts_explicit():
    small = MemoryProbePolicy(2)
    signed = MemoryProbePolicy(16)
    absolute = MemoryProbePolicy(16, absolute_input=True)
    count = lambda model: sum(p.numel() for p in model.parameters())
    assert count(small) == 33
    assert count(signed) == count(absolute) == 929
    assert count(ErrorFilterProbePolicy()) == 2
    assert count(ErrorFilterProbePolicy(learn_taus=True)) == 4
    absolute.load_state_dict(signed.state_dict())
    h = torch.zeros(1, 16)
    assert torch.equal(signed(torch.tensor([.4]), h)[2], absolute(torch.tensor([.4]), h)[2])
    assert torch.equal(absolute(torch.tensor([.4]), h)[2], absolute(torch.tensor([-.4]), h)[2])
