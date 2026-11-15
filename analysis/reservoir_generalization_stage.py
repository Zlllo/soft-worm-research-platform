"""Prospective paired noise and sample-budget engineering comparison."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from thermotaxis.connectome import load_connectome,rewire_directed
from fixed_reservoir_pilot import graph_and_weights
from reservoir_stability_stage import simulate
P=Path('docs/reports/assets/reservoir-generalization-2026-11-15')

def fit(train,y,val,vy):
 mean=train.mean(0); ym=y.mean(); X=train-mean; V=val-mean
 # Explicit covariance eigenanalysis; penalty scales with number of training samples.
 eig,Q=np.linalg.eigh((X.T@X)/len(X)); eig=np.maximum(eig,0); target=Q.T@(X.T@(y-ym)/len(X))
 best=None
 for lam in (1e-8,1e-6,1e-4,1e-2):
  c=Q@(target/(eig+lam)); err=float(np.mean((V@c+ym-vy)**2))
  if best is None or err<best[0]: best=(err,lam,c)
 return mean,ym,best[2],best[1]

def assess(clean,noisy,target,budget):
 X=clean[:64,10:].reshape(64,-1,clean.shape[-1]); V=clean[64:80,10:].reshape(-1,clean.shape[-1]); Y=target[:64,10:].reshape(64,-1); vy=target[64:80,10:].reshape(-1)
 start=time.perf_counter(); m,b,c,l=fit(X[:budget].reshape(-1,clean.shape[-1]),Y[:budget].reshape(-1),V,vy); fit_time=time.perf_counter()-start
 truth=target[80:,10:].reshape(-1); zero=float(np.mean(truth**2)); scores=[]
 for data in noisy:
  d=data[:,10:].reshape(-1,clean.shape[-1]); scores.append(float(1-np.mean(((d-m)@c+b-truth)**2)/zero))
 return dict(scores=scores,penalty=l,fit_seconds=fit_time,zero_mse=zero)

def main():
 names=json.loads(Path('data/connectome/herm_neurons.json').read_text())['names']; g=load_connectome().select(names); mask=g.chemical>0
 gap=(g.gap>0).astype(float); np.fill_diagonal(gap,0); gap*=.05/max(1,gap.sum(0).max())
 inp=np.zeros(len(names)); inp[[names.index('AFDL'),names.index('AFDR')]]=1; ria=[names.index('RIAL'),names.index('RIAR')]
 rng=np.random.default_rng(91821); u=np.empty((112,120))
 for row in u:
  t=0
  while t<120:
   length=int(rng.integers(3,10)); row[t:t+length]=rng.normal(0,.35); t+=length
 y=u-np.roll(u,5,axis=1); y[:,:5]=0
 noise=np.random.default_rng(91822).normal(size=(32,120)); noisy=[u[80:],u[80:]+.05*noise,u[80:]+.15*noise]
 results=[]; base=[]
 with threadpool_limits(limits=1):
  for budget in (4,16,64): base.append(dict(budget=budget,**assess(u[:,:,None],[x[:,:,None] for x in noisy],y,budget)))
  for label,graph in [('real',mask)]+[(str(i),rewire_directed(mask,i)[0]) for i in range(101,106)]:
   for seed in (7,17,27):
    W=graph_and_weights(graph,seed,len(names)); start=time.perf_counter(); clean=simulate(u,W,gap,inp,.025,'heun'); simsec=time.perf_counter()-start
    ns=[clean[80:]]+[simulate(x,W,gap,inp,.025,'heun') for x in noisy[1:]]
    for labelout,idx in [('ria',ria),('all',list(range(len(names))))]:
     for budget in (4,16,64): results.append(dict(graph=label,seed=seed,readout=labelout,budget=budget,simulation_seconds=simsec,**assess(clean[:,:,idx],[x[:,:,idx] for x in ns],y,budget)))
    print(label,seed,flush=True)
 (P/'results.json').write_text(json.dumps(dict(protocol=json.loads((P/'protocol.json').read_text()),baseline=base,rows=results,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__':main()
