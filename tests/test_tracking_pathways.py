import pytest
import torch

from thermotaxis.connectome import CORE, load_connectome
from thermotaxis.tracking_circuit import CircuitTrackingNetwork
from thermotaxis.tracking_learning import TrackingNetworkConfig
from thermotaxis.tracking_pathways import with_pathways


def network(interface='full'):
    torch.manual_seed(7)
    return CircuitTrackingNetwork(load_connectome().select(CORE),
        TrackingNetworkConfig(encoding='demeaned'), interface=interface).eval()


def history():
    g=torch.Generator().manual_seed(41)
    x=torch.randn(4,41,4,generator=g);x[:,:,3]=1
    return x


def test_intact_pathway_clone_reproduces_original_without_mutation():
    original=network();altered=with_pathways(original,'intact');x=history()
    assert torch.equal(original(x),altered(x))
    isolated=with_pathways(original,'isolated')
    assert all(torch.equal(v,isolated.state_dict()[k]) for k,v in original.state_dict().items())
    a,b=isolated.effective_matrices();c,g=original.effective_matrices()
    assert torch.equal(a.diag(),c.diag())
    assert torch.count_nonzero(a-torch.diag(a.diag()))==0
    assert torch.count_nonzero(b)==0
    assert torch.count_nonzero(g)>0


def test_isolated_restricted_readout_cannot_see_observations_but_full_can():
    x=history();y=x.clone();y[:,:,:3]+=1
    restricted=with_pathways(network('restricted'),'isolated')
    assert torch.equal(restricted(x),restricted(y))
    full=with_pathways(network(),'isolated')
    assert not torch.allclose(full(x),full(y),atol=1e-9,rtol=0)
    full(x).sum().backward()
    offdiag=full.pre!=full.post
    assert torch.count_nonzero(full.chemical_weights.grad[offdiag])==0
    assert full.input_weights.grad.abs().sum()>0


@pytest.mark.parametrize('mode',['no_chemical','no_gap'])
def test_selective_pathways_leave_the_other_matrix_unchanged(mode):
    original=network();altered=with_pathways(original,mode)
    a,b=original.effective_matrices();c,d=altered.effective_matrices()
    if mode=='no_chemical':
        assert torch.count_nonzero(c)==0 and torch.equal(b,d)
    else:
        assert torch.equal(a,c) and torch.count_nonzero(d)==0
