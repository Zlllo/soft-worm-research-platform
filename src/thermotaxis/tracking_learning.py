"""Finite-history neural tracking controllers and causal imitation recorders.

Only SweepReading enters the policy. Expert commands are supervised targets,
never input features. GRU state resets for EACH complete history-window query.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True)
class TrackingNetworkConfig:
    architecture: str = 'gru'
    encoding: str = 'raw'
    window_samples: int = 41
    sensor_dt_s: float = .1
    preferred_K: float = 293.15
    temperature_scale_K: float = .05
    max_turn_rate_rad_s: float = .3
    hidden_size: int = 16

    def __post_init__(self):
        if self.architecture not in ('gru','mlp') or self.encoding not in ('raw','demeaned'):
            raise ValueError('invalid network architecture/encoding')
        for key in ('window_samples','hidden_size'):
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,int) or v<1:
                raise ValueError(f'{key} must be positive integer')
        for key in ('sensor_dt_s','temperature_scale_K','max_turn_rate_rad_s'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:
                raise ValueError(f'{key} must be finite positive')
        if not math.isfinite(self.preferred_K):
            raise ValueError('preferred temperature must be finite')


class ReadingWindow:
    def __init__(self,config):
        self.config=config
        self.readings=deque(maxlen=config.window_samples)

    def observe(self,reading):
        if not all(math.isfinite(v) for v in (reading.sample_time_s,reading.temperature_K,reading.motor_phase_rad)):
            raise ValueError('nonfinite reading')
        if self.readings:
            interval=reading.sample_time_s-self.readings[-1].sample_time_s
            if not math.isclose(interval,self.config.sensor_dt_s,abs_tol=1e-9,rel_tol=0):
                raise ValueError('unexpected sensor interval')
        self.readings.append(reading)

    def array(self):
        if not self.readings:
            raise ValueError('a reading is required before querying a command')
        c=self.config
        result=np.zeros((c.window_samples,4),dtype=np.float32)
        start=c.window_samples-len(self.readings)
        for i,r in enumerate(self.readings,start):
            result[i]=((r.temperature_K-c.preferred_K)/c.temperature_scale_K,
                       math.sin(r.motor_phase_rad),math.cos(r.motor_phase_rad),1.)
        return result


class TrackingNetwork(nn.Module):
    def __init__(self,config=TrackingNetworkConfig()):
        super().__init__()
        self.config=config
        if config.architecture=='gru':
            self.memory=nn.GRU(5,config.hidden_size,batch_first=True)
            self.head=nn.Linear(config.hidden_size,1)
        else:
            self.memory=nn.Sequential(nn.Flatten(),nn.Linear(config.window_samples*5,64),
                nn.Tanh(),nn.Linear(64,32),nn.Tanh())
            self.head=nn.Linear(32,1)
        # Nonzero readout permits representation gradients at the first update.
        nn.init.normal_(self.head.weight,std=.02)
        nn.init.zeros_(self.head.bias)

    def features(self,windows):
        mask=windows[:,:,3:4]
        level=windows[:,:,:1]
        if self.config.encoding=='demeaned':
            mean=(level*mask).sum(1,keepdim=True)/mask.sum(1,keepdim=True).clamp_min(1)
            local=(level-mean)*mask
            background=mean.expand_as(level)*mask
        else:
            local=level*mask
            background=torch.zeros_like(local)
        return torch.cat((local,background,windows[:,:,1:3]*mask,mask),dim=2)

    def forward(self,windows):
        features=self.features(windows)
        if self.config.architecture=='gru':
            lengths=windows[:,:,3].sum(1).to(torch.long)
            if torch.any(lengths<1):
                raise ValueError('empty history window')
            if torch.all(lengths==self.config.window_samples):
                _,hidden=self.memory(features)
            else:
                # Move valid RIGHT-aligned histories to the left for packing.
                t=torch.arange(self.config.window_samples,device=windows.device)[None,:]
                index=(self.config.window_samples-lengths[:,None]+t).clamp_max(self.config.window_samples-1)
                shifted=features.gather(1,index[:,:,None].expand(-1,-1,5))
                packed=pack_padded_sequence(shifted,lengths.cpu(),batch_first=True,enforce_sorted=False)
                _,hidden=self.memory(packed)
            representation=hidden[-1]
        else:
            representation=self.memory(features)
        return self.config.max_turn_rate_rad_s*torch.tanh(self.head(representation).squeeze(-1))


class NeuralTrackingController:
    def __init__(self,model):
        self.model=model
        self.window=ReadingWindow(model.config)
        self.queries=0

    def observe(self,reading):
        self.window.observe(reading)

    def command(self,time_s):
        if self.window.readings[-1].sample_time_s>time_s+1e-10:
            raise ValueError('future observation')
        with torch.no_grad():
            command=float(self.model(torch.from_numpy(self.window.array()[None]))[0])
        self.queries+=1
        return command

    def diagnostics(self):
        return dict(controller='finite_window_neural',queries=self.queries,
                    observation='local noisy temperature, motor phase, causal window availability')


class ImitationRecorder:
    """Expert-only rollout, or seeded expert/learner mixture for aggregation."""
    def __init__(self,expert,network_config,learner=None,beta=1.,seed=0):
        if not 0<=beta<=1:
            raise ValueError('mixture beta must be in [0,1]')
        self.expert=expert
        self.learner=NeuralTrackingController(learner) if learner is not None else None
        self.window=ReadingWindow(network_config)
        self.beta=beta
        self.rng=np.random.default_rng(seed)
        self.windows=[]
        self.targets=[]
        self.times=[]
        self.expert_actions=0

    def observe(self,reading):
        self.window.observe(reading)
        self.expert.observe(reading)
        if self.learner is not None:
            self.learner.observe(reading)

    def command(self,time_s):
        target=float(self.expert.command(time_s))
        self.windows.append(self.window.array())
        self.targets.append(target)
        self.times.append(time_s)
        if self.learner is None or self.rng.random()<self.beta:
            self.expert_actions+=1
            return target
        return self.learner.command(time_s)

    def dataset(self):
        return (np.asarray(self.windows,dtype=np.float32),np.asarray(self.targets,dtype=np.float32),
                np.asarray(self.times,dtype=np.float32))

    def diagnostics(self):
        return dict(controller='supervised_rollout_recorder',labels=len(self.targets),
                    expert_actions=self.expert_actions,beta=self.beta)
