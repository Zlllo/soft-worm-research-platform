"""Compare Euler and Heun on fixed models, identical input and frozen readout."""
import json, time, hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from thermotaxis.connectome import load_connectome,rewire_directed
from fixed_reservoir_pilot import graph_and_weights, ridge

OUT=Path('docs/reports/assets/reservoir-stability-stage-2026-11-15')
def simulate(u,W,gap,inputs,step,method):
 x=np.zeros((len(u),len(W))); out=np.empty((len(u),u.shape[1],len(W)))
 def f(z,signal): return (-z+np.tanh(z@W+z@gap-z*gap.sum(0)+signal[:,None]*inputs))/.3
 for t in range(u.shape[1]):
  for _ in range(round(.1/step)):
   k=f(x,u[:,t]); x=x+step*k if method=='euler' else x+step*(k+f(x+step*k,u[:,t]))/2
  out[:,t]=x
 return out

def main():
 names=json.loads(Path('data/connectome/herm_neurons.json').read_text())['names']; g=load_connectome().select(names); base=g.chemical>0
 gap=(g.gap>0).astype(float); np.fill_diagonal(gap,0); gap*=.05/max(1,gap.sum(0).max())
 inputs=np.zeros(len(names)); inputs[[names.index('AFDL'),names.index('AFDR')]]=1
 rng=np.random.default_rng(4815); u=np.repeat(rng.normal(0,.35,(96,20)),5,axis=1); y=u-np.roll(u,5,axis=1); y[:,:5]=0; Y=y[:,10:].reshape(-1)
 rows=[]
 with threadpool_limits(limits=1):
  for label,graph in [('real',base),('rewired-101',rewire_directed(base,101)[0]),('rewired-104',rewire_directed(base,104)[0])]:
   for seed in (7,17,27):
    W=graph_and_weights(graph,seed,len(names)); start=time.perf_counter(); ref=simulate(u,W,gap,inputs,.00625,'heun'); refseconds=time.perf_counter()-start
    trajectories={}
    for method in ('euler','heun'):
     for dt in (.1,.05,.025,.0125):
      start=time.perf_counter(); trajectories[method,dt]=(simulate(u,W,gap,inputs,dt,method),time.perf_counter()-start)
    for name,idx in [('ria',[names.index('RIAL'),names.index('RIAR')]),('all',list(range(len(names))))]:
     X=ref[:,10:,idx].reshape(-1,len(idx)); m,ym,c,p=ridge(X[:5760],Y[:5760],X[5760:7200],Y[5760:7200]); truth=Y[7200:]; z=np.mean(truth**2)
     for method in ('euler','heun'):
      for dt in (.1,.05,.025,.0125):
       a,elapsed=trajectories[method,dt]; A=a[:,10:,idx].reshape(-1,len(idx)); score=1-np.mean(((A[7200:]-m)@c+ym-truth)**2)/z
       rows.append(dict(graph=label,seed=seed,readout=name,method=method,dt=dt,relative_rms=float(np.linalg.norm(A-X)/np.linalg.norm(X)),score=float(score),seconds=elapsed,reference_seconds=refseconds,penalty=p))
    print(label,seed,flush=True)
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'results.json').write_text(json.dumps(dict(rows=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__': main()
