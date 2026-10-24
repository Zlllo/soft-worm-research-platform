from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from thermotaxis.isothermal_scene import (TrackingConfig,SweepReading,RegressionControlConfig,
    PhaseRegressionController,run_tracking)
from thermotaxis.tracking_learning import (TrackingNetworkConfig,TrackingNetwork,
    ReadingWindow,NeuralTrackingController,ImitationRecorder)


def reading(t):
    return SweepReading(t,293.15+.01*math.sin(t),math.pi*t+.2)


def test_window_is_causal_right_aligned_and_has_no_geometry_truth():
    config=TrackingNetworkConfig()
    history=ReadingWindow(config)
    history.observe(reading(0.))
    first=history.array()
    assert first.shape==(41,4)
    assert first[:-1].sum()==0
    assert first[-1,-1]==1
    for i in range(1,45):
        history.observe(reading(i*.1))
    assert first[-1,0]==0
    assert history.array()[:,-1].sum()==41
    with pytest.raises(ValueError,match='interval'):
        history.observe(reading(4.51))


@pytest.mark.parametrize('encoding',['raw','demeaned'])
def test_packed_gru_matches_valid_sequence_and_ignores_padding(encoding):
    torch.manual_seed(7)
    model=TrackingNetwork(TrackingNetworkConfig(encoding=encoding))
    histories=[]
    for length in (1,17,41):
        window=ReadingWindow(model.config)
        for i in range(length):window.observe(reading(i*.1))
        histories.append(window.array())
    x=torch.tensor(np.array(histories))
    features=model.features(x)
    lengths=[1,17,41]
    with torch.no_grad():
        actual=model(x)
        for i,length in enumerate(lengths):
            _,state=model.memory(features[i:i+1,-length:])
            expected=.3*torch.tanh(model.head(state[-1]).squeeze())
            assert actual[i].item()==pytest.approx(expected.item(),abs=1e-8)
        altered=x.clone();altered[0,:-1,:3]=12345.
        assert model(altered)[0].item()==pytest.approx(actual[0].item(),abs=1e-8)


def test_demeaning_preserves_temperature_information_and_initial_memory_gradient():
    torch.manual_seed(7)
    model=TrackingNetwork(TrackingNetworkConfig(encoding='demeaned'))
    history=ReadingWindow(model.config)
    for i in range(41):history.observe(reading(i*.1))
    x=torch.tensor(history.array()[None])
    features=model.features(x)
    assert torch.allclose(features[:,:,0]+features[:,:,1],x[:,:,0])
    prediction=model(x)
    ((prediction-.1)**2).sum().backward()
    assert sum(p.grad.abs().sum().item() for p in model.memory.parameters())>0
    assert abs(prediction.item())<=.3


def test_teacher_targets_match_same_tick_commands_and_window_availability():
    config=TrackingConfig(duration_s=8.)
    expert=PhaseRegressionController()
    recorder=ImitationRecorder(expert,TrackingNetworkConfig())
    result=run_tracking(config,recorder,101,record_trajectory=True)
    x,y,t=recorder.dataset()
    assert np.allclose(t,result['control_commands'][:,0])
    assert np.allclose(y,result['control_commands'][:,1])
    assert x[0,:,-1].sum()==1
    assert x[20,:,-1].sum()==41
    replay=PhaseRegressionController()
    commands=[]
    for i,sample in enumerate(result['sensor_samples']):
        replay.observe(SweepReading(*sample[:3]))
        if i%2==0:commands.append(replay.command(sample[0]))
    assert y==pytest.approx(commands)


def test_neural_controller_online_command_equals_offline_window_query():
    torch.manual_seed(7)
    model=TrackingNetwork()
    controller=NeuralTrackingController(model)
    for i in range(50):
        controller.observe(reading(i*.1))
    with torch.no_grad():expected=float(model(torch.tensor(controller.window.array()[None]))[0])
    assert controller.command(4.9)==expected
    with pytest.raises(ValueError,match='future'):
        controller.command(4.8)


def test_learner_only_aggregation_labels_do_not_become_input_or_actions():
    torch.manual_seed(7)
    model=TrackingNetwork()
    config=TrackingConfig(duration_s=4.)
    recorder=ImitationRecorder(PhaseRegressionController(),model.config,model,beta=0.)
    mixed=run_tracking(config,recorder,102,record_trajectory=True)
    standalone=run_tracking(config,NeuralTrackingController(model),102,record_trajectory=True)
    assert np.array_equal(mixed['trajectory'],standalone['trajectory'])
    assert recorder.expert_actions==0
    assert len(recorder.targets)==mixed['clocks']['control_updates']
