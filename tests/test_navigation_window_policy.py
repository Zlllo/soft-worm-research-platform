import numpy as np
import torch
from thermotaxis.navigation_window_policy import NavigationFeatures,WindowActorCritic


def test_causal_features_retain_level_and_do_not_leak_adaptation_into_raw():
    p=np.zeros((2,3,5),dtype=np.float32);p[-1,:,0]=293.15;p[-1,:,2:]=1
    raw=NavigationFeatures(3,'raw');adaptive=NavigationFeatures(3,'adaptive')
    a=raw.observe(p);b=adaptive.observe(p)
    assert a.shape==(3,41,7) and np.all(a[:,:,2:4]==0)
    p[:,:,0]=293.17;p[:,:,4]=1
    a=raw.observe(p);b=adaptive.observe(p)
    assert np.array_equal(a[:,:,:2],b[:,:,:2]) and np.array_equal(a[:,:,4:],b[:,:,4:])
    assert np.any(b[:,:,2:4]!=0)
    old=a.copy();invalid=p.copy();invalid[:,:,4]=0;invalid[:,:,0]=1e6
    assert np.array_equal(raw.observe(invalid),old)


def test_graph_actor_initialization_and_degrees_are_paired_and_isolated_path_is_absent():
    torch.manual_seed(7);real=WindowActorCritic('real8','restricted')
    torch.manual_seed(7);rewired=WindowActorCritic('rewired8-44','restricted')
    assert real.metadata()['actor_parameters']==rewired.metadata()['actor_parameters']
    assert all(torch.equal(p,q) for p,q in zip(real.parameters(),rewired.parameters()))
    torch.manual_seed(7);isolated=WindowActorCritic('isolated8','restricted')
    x=torch.zeros(2,41,7);x[:,:,6]=1;y=x.clone();y[:,:,:6]=1
    assert torch.equal(isolated.actor_mean(x),isolated.actor_mean(y))


def test_padding_is_ignored_and_actor_parameters_have_gradients():
    for name in ['real8','gru16']:
        model=WindowActorCritic(name,'full');x=torch.zeros(2,41,7);x[:,-8:,6]=1;x[:,-8:,0]=.2
        y=x.clone();y[:,:-8,:6]=1e6
        assert torch.equal(model.actor_mean(x),model.actor_mean(y))
        model.actor_mean(x).sum().backward();assert model.actor.weight.grad.abs().sum()>0


def test_critic_is_identical_across_actors_and_seeds():
    torch.manual_seed(7);a=WindowActorCritic('real8','restricted')
    torch.manual_seed(27);b=WindowActorCritic('gru16','full')
    assert all(torch.equal(p,q) for p,q in zip(a.critic.parameters(),b.critic.parameters()))
