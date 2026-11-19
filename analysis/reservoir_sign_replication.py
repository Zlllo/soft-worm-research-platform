"""Engineering sensitivity to an anatomical-count-derived weight assumption."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from thermotaxis.connectome import load_connectome,rewire_directed
from reservoir_stability_stage import simulate
from reservoir_generalization_stage import assess
P=Path('docs/reports/assets/reservoir-sign-replication-2026-11-15')
names=json.loads(Path('data/connectome/herm_neurons.json').read_text())['names'];g=load_connectome().select(names);mask=g.chemical>0;n=len(names)
weights=[('real',np.log1p(g.chemical))]
for seed in range(201,211):
 graph=rewire_directed(mask,seed)[0]; rng=np.random.default_rng(92000+seed); w=np.zeros((n,n))
 for i in range(n):
  vals=np.log1p(g.chemical[i,mask[i]]); idx=np.flatnonzero(graph[i]); assert len(vals)==len(idx)
  w[i,idx]=rng.permutation(vals)
 weights.append((str(seed),w))
norm=max(w.sum(0).max() for _,w in weights)
gap=(g.gap>0).astype(float); np.fill_diagonal(gap,0);gap*=.05/gap.sum(0).max();inp=np.zeros(n);inp[[names.index('AFDL'),names.index('AFDR')]]=1;ria=[names.index('RIAL'),names.index('RIAR')]
rng=np.random.default_rng(72821);u=np.empty((112,120))
for row in u:
 t=0
 while t<120:
  length=int(rng.integers(3,10));row[t:t+length]=rng.normal(0,.35);t+=length
y=u-np.roll(u,5,axis=1);y[:,:5]=0;rows=[]
with threadpool_limits(limits=1):
 baseline=assess(u[:,:,None],[u[80:,:,None]],y,64)
 for label,w in weights:
  for seed in (37,47,57):
   signs=np.random.default_rng(seed).choice([-1.,1.],size=n); W=w*signs[:,None]/norm
   s=simulate(u,W,gap,inp,.025,'heun')
   for out,idx in [('ria',ria),('all',list(range(n)))]:
    metric=assess(s[:,:,idx],[s[80:,:,idx]],y,64);rows.append(dict(graph=label,seed=seed,readout=out,**metric))
   print(label,seed,flush=True)
(P/'results.json').write_text(json.dumps(dict(protocol=json.loads((P/'protocol.json').read_text()),normalization=float(norm),baseline=baseline,rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
