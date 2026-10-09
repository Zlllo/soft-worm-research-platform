"""Evaluation termination never projects pose or requests a terminal action."""
from dataclasses import replace
import pytest
from thermotaxis.phase2_control import ControlConfig, run_four_action


def test_terminal_frame_retains_outside_position_and_no_terminal_callback():
    config=replace(ControlConfig(),duration_s=1.,physics_dt_s=.02,control_dt_s=.02,initially_active=True)
    called=[]
    def policy(obs):
        called.append(obs.time_s)
        return 2
    # The body advances naturally left; terminal criterion is a declared domain.
    result=run_four_action(config,policy,terminal_condition=lambda head,time:time>0 and head[0]<-.000500001)
    assert result['termination'] is not None
    terminal=result['trajectory'][-1]
    assert terminal['head_x_m']<-.000500001
    assert terminal['x_m']!=0
    assert terminal['time_s']==result['termination']['time_s']
    assert terminal['time_s']/config.physics_dt_s == pytest.approx(round(terminal['time_s']/config.physics_dt_s))
    assert called[-1]<terminal['time_s']
    assert len(called)==len(result['controls'])
    assert len(result['trajectory'])==round(terminal['time_s']/config.physics_dt_s)+1


def test_terminal_at_initial_state_does_not_read_privileged_observer():
    config=replace(ControlConfig(),duration_s=1.,observation_mode='legacy_privileged')
    def never(*args):
        raise AssertionError('terminal frame must not ask policy or observer')
    result=run_four_action(config,never,legacy_observer=never,terminal_condition=lambda head,time:True)
    assert len(result['trajectory'])==1
    assert result['controls']==[]
    assert result['termination']['time_s']==0
