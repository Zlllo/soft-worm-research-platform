"""Controlled window-policy factorial: topology x I/O x sensory adaptation.

All conditions retain the same old temperature-level and phase features.
Additional adaptive channels are causal threshold response/state. These use
paper-anchored times, not a full physiological reconstruction. Critic is a
separate identical MLP for every actor; it never changes actor observations.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .connectome import CORE,load_connectome,rewire_directed
from .sensory_adaptation import ThresholdAdaptation


class NavigationFeatures:
    def __init__(self,batch,encoding='raw'):
        if encoding not in ('raw','adaptive'):raise ValueError('invalid sensory factor')
        self.encoding=encoding;self.values=np.zeros((batch,41,6),dtype=np.float32)
        self.threshold=np.zeros(batch);self.initialized=np.zeros(batch,dtype=bool)
        self.sensor=ThresholdAdaptation()

    def observe(self,packet):
        for sample in packet:
            valid=sample[:,4]>0
            if not valid.any():continue
            level=sample[:,0]-293.15;before=np.where(self.initialized,self.threshold,level)
            response=self.sensor.response(level,before)
            updated=self.sensor.update(level,before,.1)
            shifted=np.roll(self.values,-1,axis=1)
            shifted[:,-1]=np.stack([level/.05,sample[:,1],sample[:,2],response,before/.05,sample[:,4]],axis=1)
            self.values=np.where(valid[:,None,None],shifted,self.values)
            self.threshold=np.where(valid,updated,self.threshold);self.initialized|=valid
        mask=self.values[:,:,5:6];level=self.values[:,:,:1]
        mean=(level*mask).sum(1,keepdims=True)/np.maximum(1,mask.sum(1,keepdims=True))
        adaptive=self.values[:,:,3:5]*mask if self.encoding=='adaptive' else np.zeros_like(self.values[:,:,3:5])
        return np.concatenate([(level-mean)*mask,np.broadcast_to(mean,level.shape)*mask,adaptive,self.values[:,:,1:3]*mask,mask],axis=2).astype(np.float32)


class WindowActorCritic(nn.Module):
    def __init__(self,name='real8',interface='full'):
        super().__init__()
        if interface not in ('full','restricted'):raise ValueError('invalid interface')
        self.name=name;self.interface=interface;self.is_gru=name.startswith('gru')
        self.size=int(name[3:]) if self.is_gru else 8
        if self.is_gru:
            self.memory=nn.GRU(7,self.size,batch_first=True);outputs=list(range(self.size))
        else:
            graph=load_connectome().select(CORE);mask=graph.chemical>0
            if name.startswith('rewired8-'):mask,_=rewire_directed(mask,int(name.split('-')[1]))
            elif name not in ('real8','isolated8'):raise ValueError('unknown actor')
            pre,post=np.nonzero(mask);self.register_buffer('pre',torch.tensor(pre));self.register_buffer('post',torch.tensor(post))
            self.chemical=nn.Parameter(torch.randn(len(pre))/math.sqrt(8))
            i,j=np.nonzero(np.triu(graph.gap>0,1));self.register_buffer('gap_i',torch.tensor(i));self.register_buffer('gap_j',torch.tensor(j))
            self.gap_logits=nn.Parameter(torch.zeros(len(i)));self.gap_scale=.2/max(1,int(np.sum(graph.gap>0,axis=0).max()))
            self.bias=nn.Parameter(.02*torch.randn(8))
            self.input_weights=nn.Parameter(.5*torch.randn(6,8 if interface=='full' else 2))
            outputs=list(range(8)) if interface=='full' else [6,7]
        self.register_buffer('outputs',torch.tensor(outputs,dtype=torch.long))
        self.actor=nn.Linear(len(outputs),1);nn.init.normal_(self.actor.weight,std=.02);nn.init.zeros_(self.actor.bias)
        self.log_std=nn.Parameter(torch.tensor(-.5))
        # Same separate critic architecture; no actor representation privilege.
        self.critic=nn.Sequential(nn.Flatten(),nn.Linear(41*7,32),nn.Tanh(),nn.Linear(32,1))
        critic_rng=np.random.default_rng(12345)
        with torch.no_grad():
            for layer in self.critic:
                if isinstance(layer,nn.Linear):
                    layer.weight.copy_(torch.from_numpy(critic_rng.normal(0,1/math.sqrt(layer.in_features),layer.weight.shape)).float())
                    layer.bias.zero_()

    def actor_mean(self,features):
        if torch.any(features[:,:,6].sum(1)<1):raise ValueError('empty observation history')
        if self.is_gru:
            length=features[:,:,6].sum(1).to(torch.long)
            if torch.all(length==41):_,h=self.memory(features)
            else:
                t=torch.arange(41,device=features.device)[None,:]
                index=(41-length[:,None]+t).clamp_max(40)
                shifted=features.gather(1,index[:,:,None].expand(-1,-1,7))
                packed=pack_padded_sequence(shifted,length.cpu(),batch_first=True,enforce_sorted=False)
                _,h=self.memory(packed)
            representation=h[-1]
        else:
            chemical=self.bias.new_zeros(8,8).index_put((self.pre,self.post),self.chemical)
            conductance=self.gap_scale*torch.sigmoid(self.gap_logits)
            gap=self.bias.new_zeros(8,8).index_put((self.gap_i,self.gap_j),conductance)
            gap=gap.index_put((self.gap_j,self.gap_i),conductance)
            if self.name=='isolated8':chemical=chemical*torch.eye(8);gap=gap*0
            if self.interface=='full':injection=features[:,:,:6]@self.input_weights
            else:
                sensory=features[:,:,:4]@self.input_weights[:4]
                motor=features[:,:,4:6]@self.input_weights[4:]
                injection=torch.cat([sensory,motor,features.new_zeros(*features.shape[:2],4)],dim=2)
            hidden=features.new_zeros(len(features),8)
            alpha=1-math.exp(-.1/.3)
            for t in range(41):
                drive=hidden@chemical+self.bias+injection[:,t]
                diffusion=hidden@gap-hidden*gap.sum(0)
                updated=hidden+alpha*(torch.tanh(drive)-hidden+diffusion)
                hidden=torch.where(features[:,t,6:7]>0,updated,hidden)
            representation=hidden[:,self.outputs]
        return self.actor(representation).squeeze(-1)

    def forward(self,features):return self.actor_mean(features),self.critic(features).squeeze(-1)

    def metadata(self):
        actor_parameters=sum(p.numel() for n,p in self.named_parameters() if not n.startswith('critic.'))
        return dict(name=self.name,interface=self.interface,neural_states=self.size,actor_parameters=actor_parameters,
            critic_parameters=sum(p.numel() for p in self.critic.parameters()),
            history_s=4.,sensory_features=4,motor_features=2,
            raw_auxiliary_channels_masked=True,critic='identical separate 287->32->1 MLP',
            restricted_scope='thermal channels at AFD, phase at AIY as engineering projection, RIA output',
            isolated_scope='chemical self loops retained; no cross-node/gap communication')
