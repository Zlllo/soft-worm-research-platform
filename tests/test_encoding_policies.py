import numpy as np
import pytest
import torch

from thermotaxis.connectome import CORE, load_connectome
from thermotaxis.encoding_policies import make_encoded_policy


def make(architecture='anatomical', encoding='adaptive', seed=7, topology_seed=10001):
    torch.manual_seed(seed)
    return make_encoded_policy(load_connectome().select(CORE),architecture,encoding,topology_seed)


def test_paired_encodings_have_identical_initial_parameters_and_action_range():
    for arch in ('anatomical','rewired','dense','gru2','gru16'):
        raw,adapt=make(arch,'raw'),make(arch,'adaptive')
        assert raw.size==adapt.size
        assert sum(p.numel() for p in raw.parameters())==sum(p.numel() for p in adapt.parameters())
        assert all(torch.equal(value,adapt.state_dict()[key]) for key,value in raw.state_dict().items())
        hidden=torch.zeros(4,raw.size)
        p,_,_=raw(torch.randn(4),hidden)
        q,_,_=adapt(torch.randn(4),hidden)
        assert torch.allclose(p,q)
        assert p[0].item()==pytest.approx(-np.expm1(-.255))


def test_signed_temporal_signal_odd_and_invariant_to_constant_temperature_offset():
    model=make()
    states=[torch.zeros(1,3) for _ in range(3)]
    for x in (.2,.3,.45,.4,.35):
        pos,states[0]=model.encode(torch.tensor([x]),states[0])
        neg,states[1]=model.encode(torch.tensor([-x]),states[1])
        offset,states[2]=model.encode(torch.tensor([x+2]),states[2])
        assert torch.allclose(pos,-neg)
        assert torch.allclose(pos[:,1],offset[:,1],atol=2e-5)
        assert offset[0,0]-pos[0,0]==pytest.approx(2.)


def test_raw_filter_states_cannot_leak_to_controller_output():
    for arch in ('anatomical','gru16'):
        model=make(arch,'raw')
        with torch.no_grad():
            model.core.actor.weight.fill_(.3)
        h=torch.zeros(4,model.size)
        changed=h.clone()
        changed[:,-3:]=torch.tensor([999.,-999.,1.])
        p,_,first=model(torch.ones(4)*.2,h)
        q,_,second=model(torch.ones(4)*.2,changed)
        assert torch.equal(p,q)
        assert torch.equal(first[:,:model.core.size],second[:,:model.core.size])


def test_filter_has_no_initial_temperature_transient_and_relaxes_at_constant_input():
    model=make()
    features,state=model.encode(torch.tensor([.7]),torch.zeros(1,3))
    assert features[0,1].item()==0.
    features,state=model.encode(torch.tensor([.8]),state)
    assert features[0,1].item()>0
    for _ in range(100):
        features,state=model.encode(torch.tensor([.8]),state)
    assert features[0,1].abs().item()<1e-5


def test_core_input_is_sensory_only_and_rewiring_independent_of_training_seed():
    model=make('anatomical')
    _,_,h=model(torch.tensor([.7]),torch.zeros(1,model.size))
    assert set(torch.nonzero(h[0,:model.core.size]).flatten().tolist())=={0,1}
    a=make('rewired',seed=7,topology_seed=10001)
    b=make('rewired',seed=17,topology_seed=10001)
    assert torch.equal(a.core.pre,b.core.pre)
    assert torch.equal(a.core.post,b.core.post)
    assert a.rewiring['changed_entries']>0
