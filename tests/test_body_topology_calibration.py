import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from thermotaxis.navigation_window_policy import WindowActorCritic

ANALYSIS=Path(__file__).resolve().parents[1]/'analysis';sys.path.insert(0,str(ANALYSIS))
spec=importlib.util.spec_from_file_location('body_topology_calibration',ANALYSIS/'body_topology_calibration.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('name',['real8','isolated8','gru16'])
def test_calibration_updates_memory_but_keeps_critic_and_graph_buffers(name):
    model=WindowActorCritic(name,'restricted' if not name.startswith('gru') else 'full')
    selected=module.select_parameters(model)
    assert 'actor.weight' in selected
    assert not any(n.startswith('critic.') or n=='log_std' for n in selected)
    if name=='gru16':assert 'memory.weight_hh_l0' in selected
    else:assert 'chemical' in selected and 'gap_logits' in selected
    if name=='isolated8':
        assert module.reachable(model,[0,1],[6,7])=={'6':False,'7':False}
