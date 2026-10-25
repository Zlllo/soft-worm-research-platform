"""Persistent-state navigation policies with explicit observation/I-O factors.

All policies see the same local sensor packet. Restricted chemical models route
thermal features to AFD and proprioception to AIY, reading RIA. Phase routing is
an engineering control, not a validated AIY head-phase pathway. AFD-inspired
threshold dynamics are shared across architectures when enabled.
"""
import math

import numpy as np
import torch
from torch import nn

from .connectome import CORE,load_connectome,rewire_directed


class NavigationPolicy(nn.Module):
    def __init__(self,name='real8',interface='restricted',encoding='raw',tau_s=.3,action_dims=1):
        super().__init__()
        if interface not in ('restricted','full') or encoding not in ('raw','adaptive'):raise ValueError('invalid interface/encoding')
        if not tau_s>0 or action_dims not in (1,2):raise ValueError('invalid dynamics/action dimensions')
        self.name=name;self.interface=interface;self.encoding=encoding;self.tau_s=tau_s;self.action_dims=action_dims
        self.is_gru=name.startswith('gru');self.size=int(name[3:]) if self.is_gru else 8
        if self.is_gru:
            self.memory=nn.GRUCell(6,self.size);outputs=list(range(self.size))
        else:
            graph=load_connectome().select(CORE);mask=graph.chemical>0
            if name.startswith('rewired8-'):mask,_=rewire_directed(mask,int(name.split('-')[1]))
            elif name not in ('real8','isolated8'):raise ValueError('unknown navigation model')
            pre,post=np.nonzero(mask);self.register_buffer('pre',torch.tensor(pre));self.register_buffer('post',torch.tensor(post))
            self.chemical=nn.Parameter(torch.randn(len(pre))/math.sqrt(8))
            i,j=np.nonzero(np.triu(graph.gap>0,1));self.register_buffer('gap_i',torch.tensor(i));self.register_buffer('gap_j',torch.tensor(j))
            self.gap_logits=nn.Parameter(torch.zeros(len(i)));self.gap_scale=.2/max(1,int(np.sum(graph.gap>0,axis=0).max()))
            self.bias=nn.Parameter(.02*torch.randn(8))
            self.input_weights=nn.Parameter(.5*torch.randn(6,8 if interface=='full' else 2))
            outputs=list(range(8)) if interface=='full' else [6,7]
        self.register_buffer('outputs',torch.tensor(outputs,dtype=torch.long))
        self.actor=nn.Linear(len(outputs),action_dims);self.critic=nn.Linear(len(outputs),1)
        nn.init.normal_(self.actor.weight,std=.02);nn.init.zeros_(self.actor.bias)
        self.log_std=nn.Parameter(torch.full((action_dims,),-.5))

    def initial(self,batch):
        p=next(self.parameters())
        return (p.new_zeros(batch,self.size),p.new_zeros(batch),p.new_zeros(batch,dtype=torch.bool))

    def matrices(self):
        chemical=self.bias.new_zeros(8,8).index_put((self.pre,self.post),self.chemical)
        conductance=self.gap_scale*torch.sigmoid(self.gap_logits)
        gap=self.bias.new_zeros(8,8).index_put((self.gap_i,self.gap_j),conductance)
        gap=gap.index_put((self.gap_j,self.gap_i),conductance)
        if self.name=='isolated8':
            chemical=chemical*torch.eye(8,device=chemical.device);gap=gap*0
        return chemical,gap

    def forward(self,packet,state):
        hidden,threshold,initialized=state
        matrices=None if self.is_gru else self.matrices()
        for sample in packet:
            valid=sample[:,4]>0
            level=sample[:,0]-293.15
            before=torch.where(initialized,threshold,level)
            tau=torch.where(level>=before,45.,840.)
            updated=before+(-torch.expm1(-.1/tau))*(level-before)
            response=((level-before)/.05).clamp(0,1)
            aux=response if self.encoding=='adaptive' else torch.zeros_like(level)
            background=before/.05 if self.encoding=='adaptive' else torch.zeros_like(level)
            features=torch.stack([(level/.05).clamp(-8,8),aux,background.clamp(-8,8),sample[:,1],sample[:,2],sample[:,3]],dim=1)
            if self.is_gru:
                hnew=self.memory(features,hidden)
            else:
                chemical,gap=matrices
                if self.interface=='full':injection=features@self.input_weights
                else:
                    sensory=features[:,:3]@self.input_weights[:3]
                    motor=features[:,3:]@self.input_weights[3:]
                    injection=torch.cat([sensory,motor,features.new_zeros(len(features),4)],dim=1)
                drive=hidden@chemical+self.bias+injection
                diffusion=hidden@gap-hidden*gap.sum(0)
                alpha=1-math.exp(-.1/self.tau_s)
                hnew=hidden+alpha*(torch.tanh(drive)-hidden+diffusion)
            hidden=torch.where(valid[:,None],hnew,hidden)
            threshold=torch.where(valid,updated,threshold);initialized=initialized|valid
        readout=hidden[:,self.outputs]
        return self.actor(readout),self.critic(readout).squeeze(-1),(hidden,threshold,initialized)

    def metadata(self):
        result=dict(name=self.name,interface=self.interface,encoding=self.encoding,persistent_state=True,
            parameters=sum(p.numel() for p in self.parameters()),neural_states=self.size,
            preprocessing_states=2,actor_dimensions=self.action_dims,
            observation='noisy local temperature, common drive phase and signed speed only',
            thermosensor_scope='45/840s threshold relaxation; working rectification, not calcium-trace calibration')
        if not self.is_gru:
            a,b=self.matrices()
            result.update(chemical_mask=((a.detach()!=0).int().tolist()),gap_mask=((b.detach()!=0).int().tolist()),
                chemical_input='AFD' if self.interface=='restricted' else 'all',
                motor_input='AIY engineering projection' if self.interface=='restricted' else 'all',readout=self.outputs.tolist())
        return result
