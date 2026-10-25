"""Replay the already-recorded initialization and typed-graph control audits."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from thermotaxis.connectome import CORE,load_connectome,rewire_directed
from thermotaxis.navigation_window_policy import WindowActorCritic


def main():
    checks=[]
    for interface in ['full','restricted']:
        for seed in [7,17,27]:
            torch.manual_seed(seed);base=WindowActorCritic('real8',interface)
            mask=torch.zeros(8,8).index_put((base.pre,base.post),torch.ones(len(base.pre)))
            diagonal=torch.zeros(8,8).index_put((base.pre,base.post),base.chemical).detach().diag()
            for graph in [11,22,33,44,55,66,77,88,99,110]:
                torch.manual_seed(seed);other=WindowActorCritic('rewired8-'+str(graph),interface)
                other_mask=torch.zeros(8,8).index_put((other.pre,other.post),torch.ones(len(other.pre)))
                other_diag=torch.zeros(8,8).index_put((other.pre,other.post),other.chemical).detach().diag()
                facts=dict(all_parameters_equal=all(torch.equal(p,q) for p,q in zip(base.parameters(),other.parameters())),
                    in_degrees=torch.equal(mask.sum(0),other_mask.sum(0)),out_degrees=torch.equal(mask.sum(1),other_mask.sum(1)),
                    self_mask=torch.equal(mask.diag(),other_mask.diag()),self_values=torch.equal(diagonal,other_diag),layout_changed=not torch.equal(mask,other_mask))
                assert all(facts.values());checks.append(dict(interface=interface,seed=seed,graph=graph,checks=facts))
    g=load_connectome().select(CORE);a=(g.chemical>0).astype(int);b=(g.gap>0).astype(int);np.fill_diagonal(b,0)
    degrees={}
    for i in range(8):degrees.setdefault((int(a[i].sum()),int(a[:,i].sum()),int(a[i,i]),int(b[i].sum())),[]).append(i)
    permutations=[]
    for choices in itertools.product(*[list(itertools.permutations(v)) for v in degrees.values()]):
        p=list(range(8))
        for original,new in zip(degrees.values(),choices):
            for i,j in zip(original,new):p[i]=j
        if np.array_equal(b[np.ix_(p,p)],b):permutations.append(p)
    iso=[]
    for seed in [11,22,33,44,55,66,77,88,99,110]:
        c,_=rewire_directed(a,seed);hits=[p for p in permutations if np.array_equal(c[np.ix_(p,p)],a)]
        iso.append(dict(graph_seed=seed,isomorphic_to_real_with_shared_gap=bool(hits)))
    print('initialization checks',len(checks),'isomorphism',iso)


if __name__=='__main__':main()
