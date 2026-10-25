import pytest
import torch
from thermotaxis.navigation_policy import NavigationPolicy


def packet():
    x=torch.zeros(2,4,5);x[:,:,0]=293.15;x[:,:,1]=.3;x[:,:,2]=.9;x[:,:,3:]=1
    return x


@pytest.mark.parametrize('name',['real8','rewired8-44','isolated8','gru2','gru8'])
def test_navigation_policy_shapes_gradients_and_invalid_samples(name):
    torch.manual_seed(7);m=NavigationPolicy(name,interface='restricted',encoding='adaptive')
    x=packet();a,v,h=m(x,m.initial(4));assert a.shape==(4,1) and v.shape==(4,)
    (a-.1).square().mean().backward()
    assert m.actor.weight.grad.abs().sum()>0
    empty=x.clone();empty[:,:,4]=0
    _,_,new=m(empty,h)
    assert all(torch.equal(p,q) for p,q in zip(h,new))


def test_restricted_input_requires_network_transmission_while_full_isolated_has_path():
    x=packet();y=x.clone();y[:,:,0]+=.04
    a=NavigationPolicy('isolated8','restricted')
    assert torch.equal(a(x,a.initial(4))[0],a(y,a.initial(4))[0])
    b=NavigationPolicy('isolated8','full')
    assert not torch.equal(b(x,b.initial(4))[0],b(y,b.initial(4))[0])
