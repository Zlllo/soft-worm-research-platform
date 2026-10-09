"""Contracts for genuine frozen legacy policies and explicit coordinate mapping."""
import numpy as np
import pytest
from thermotaxis.legacy_policy import FrozenQTable, LegacyFrameStack, LegacyGridMap, legacy_action_heading


def test_frozen_table_requires_valid_full_state_and_cannot_mutate_training_table():
    original=np.arange(3*5*4).reshape(3,5,4).astype(float)
    policy=FrozenQTable(original,tie_seed=7)
    original[1,2,:]=100
    assert policy.select_action(2,1) == 3
    exposed=policy.q_table; exposed.setflags(write=True); exposed[:]=0
    assert policy.select_action(2,1) == 3
    for invalid in ((-1,0),(5,0),(0,3),(1.2,0),(True,0)):
        with pytest.raises(ValueError): policy.select_action(*invalid)
    for invalid in (np.zeros((2,2,8)),np.full((2,2,4),np.nan)):
        with pytest.raises(ValueError): FrozenQTable(invalid,tie_seed=0)


def test_seeded_ties_preserve_greedy_choices_and_explicit_y_flip():
    q=np.zeros((3,5,4));q[:,:,2:]=1
    a=FrozenQTable(q,tie_seed=8); b=FrozenQTable(q,tie_seed=8)
    actions=[a.select_action(0,0) for _ in range(80)]
    assert actions == [b.select_action(0,0) for _ in range(80)]
    assert set(actions) == {2,3}
    mapping=LegacyGridMap(5,3,.004,.002,y_axis_up=True)
    assert mapping.indices((0,0))==(0,2)
    assert mapping.indices((.004,.002))==(4,0)
    assert mapping.indices((.002,.001))==(2,1)
    assert legacy_action_heading(0,y_axis_up=True)==pytest.approx(np.pi/2)
    for p in ((-.001,0),(0,.003),(float('nan'),0)):
        with pytest.raises(ValueError): mapping.indices(p)
    with pytest.raises(ValueError): mapping.callback(FrozenQTable(np.zeros((2,2,4)),tie_seed=0))


def test_old_text_export_requires_every_cell_and_declared_action_header(tmp_path):
    text='Q表数据 - test\n位置(x,y)\t温度\t上Q\t下Q\t左Q\t右Q\t偏好动作\n---\n'
    rows=['(%d,%d)\t20.0\t1.1\t2.2\t3.3\t4.4\t右' % (x,y) for y in range(2) for x in range(2)]
    p=tmp_path/'q_table.npy';p.write_text(text+'\n'.join(rows))
    policy=FrozenQTable.from_legacy_text(p,width=2,height=2,tie_seed=7)
    assert policy.select_action(1,1)==3
    p.write_text(text+'\n'.join(rows[:-1]))
    with pytest.raises(ValueError): FrozenQTable.from_legacy_text(p,width=2,height=2,tie_seed=7)


def test_real_frames_are_stacked_without_truncation_or_zero_substitution():
    stack=LegacyFrameStack();first=np.arange(8,dtype=np.float32)
    np.testing.assert_array_equal(stack.push(first),np.tile(first,4))
    np.testing.assert_array_equal(stack.push(first+10),np.r_[np.tile(first,3),first+10])
    for bad in (np.zeros(7),np.zeros(9),np.full(8,np.inf)):
        with pytest.raises(ValueError): stack.push(bad)


@pytest.mark.parametrize('architecture',['simple','dueling'])
def test_frozen_actual_legacy_architecture_matches_network_without_learning(tmp_path,architecture):
    torch=pytest.importorskip('torch')
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'程序'))
    from core.neural_networks import SimpleNeuralNetwork,DuelingDQN
    from thermotaxis.legacy_policy import FrozenLegacyDQN
    torch.manual_seed(81021)
    network=(SimpleNeuralNetwork if architecture=='simple' else DuelingDQN)(input_size=32,hidden_size=12,output_size=4)
    network.eval();state={k:v.detach().clone() for k,v in network.state_dict().items()}
    checkpoint=tmp_path/'synthetic-contract-checkpoint.pt';torch.save(state,checkpoint)
    frozen=FrozenLegacyDQN(checkpoint,architecture=architecture)
    stack=LegacyFrameStack()
    for value in (1.,2.,-.5):
        frame=np.arange(8,dtype=np.float32)*value
        with torch.no_grad(): expected=int(network(torch.from_numpy(stack.push(frame)).unsqueeze(0)).argmax())
        assert frozen.select_action(frame)==expected
    for k,v in frozen._network.state_dict().items(): assert torch.equal(v,state[k])
    assert all(not p.requires_grad and p.grad is None for p in frozen._network.parameters())
    assert not frozen._network.training
    with pytest.raises(ValueError): frozen.select_action(np.zeros(10))
