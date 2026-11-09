"""Check parameter isolation, actual graph surgery and causal probe features."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from thermotaxis.navigation import NavigationConfig
from thermotaxis.navigation_window_policy import WindowActorCritic

ANALYSIS=Path(__file__).resolve().parents[1]/'analysis'
sys.path.insert(0,str(ANALYSIS))
spec=importlib.util.spec_from_file_location('pathway_stage1',ANALYSIS/'run_pathway_stage1.py')
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)


@pytest.mark.parametrize('arm',stage.ARMS)
def test_only_declared_parameters_can_change(arm):
    torch.manual_seed(9)
    model=WindowActorCritic('real8','restricted')
    before={k:v.clone() for k,v in model.state_dict().items()}
    selected=stage.configure_trainable(model,arm)
    x,_=stage.stimuli();x=x[:8]
    if selected:
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.003)
        loss=(torch.tanh(model.actor_mean(x))-.2).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    changed={k for k,v in before.items() if not torch.equal(v,model.state_dict()[k])}
    assert changed<=set(selected)
    assert all(not n.startswith('critic.') and n!='log_std' for n in selected)
    if arm=='frozen':assert not changed
    if arm=='R':assert set(selected)=={'actor.weight','actor.bias'}
    if arm=='IR':assert set(selected)=={'actor.weight','actor.bias','input_weights'}
    if arm=='All':assert {'chemical','bias','gap_logits'}<=set(selected)


def test_ablation_changes_only_existing_afd_to_aiy_and_preserves_original():
    model=WindowActorCritic('real8','restricted')
    before={k:v.clone() for k,v in model.state_dict().items()}
    altered,edges=stage.ablate_afd_aiy(model)
    assert edges
    mask=(model.pre<2)&(model.post>=2)&(model.post<4)
    assert torch.all(altered.chemical[mask]==0)
    assert torch.equal(altered.chemical[~mask],model.chemical[~mask])
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all(torch.equal(v,altered.state_dict()[k]) for k,v in before.items() if k!='chemical')


def test_balanced_cases_have_eight_per_field_and_temperature_side():
    cases=stage.balanced_cases(NavigationConfig())
    assert len(set(cases['seeds']))==32
    counts={}
    for c in cases['cases']:
        key=c['field'],c['initial_error_K']>0
        counts[key]=counts.get(key,0)+1
    assert len(counts)==4 and set(counts.values())=={8}


def test_online_probe_no_future_stimulus_or_padding_changes():
    t,base,changed=stage.online_windows(.002,'step',.5)
    np.testing.assert_array_equal(base[t<4],changed[t<4])
    np.testing.assert_array_equal(base[:,:,:,2:],changed[:,:,:,2:])
    assert np.any(base[t>=4,:,:,0]!=changed[t>=4,:,:,0])
    # At onset only the newest observation changes, not preceding temperatures.
    index=np.flatnonzero(t==4)[0]
    absolute0=(base[index,:,:,0]+base[index,:,:,1])*.05
    absolute1=(changed[index,:,:,0]+changed[index,:,:,1])*.05
    np.testing.assert_allclose(absolute1[:,:-1],absolute0[:,:-1],atol=2e-8)
