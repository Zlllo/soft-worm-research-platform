"""AFD-to-AIY deletion and source-matched controls in an uncalibrated model."""
import hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from thermotaxis.connectome import load_connectome
from fixed_reservoir_pilot import graph_and_weights
from reservoir_stability_stage import simulate
from reservoir_generalization_stage import fit
P=Path('docs/reports/assets/reservoir-pathway-controls-2026-11-15')
names=json.loads(Path('data/connectome/herm_neurons.json').read_text())['names'];g=load_connectome().select(names); mask=g.chemical>0; n=len(names)
afd=[names.index('AFDL'),names.index('AFDR')]; aiy=[names.index('AIYL'),names.index('AIYR')]; ria=[names.index('RIAL'),names.index('RIAR')]
target=list(zip(afd,aiy)); assert all(mask[i,j] for i,j in target)
opts=[np.flatnonzero(mask[i] & ~np.isin(np.arange(n),afd+[j])) for i,j in target]; rng=np.random.default_rng(91824)
controls=[]
while len(controls)<4:
 candidate=[(i,int(rng.choice(o))) for i,o in zip(afd,opts)]
 if candidate not in controls: controls.append(candidate)
gap=(g.gap>0).astype(float);np.fill_diagonal(gap,0);gap*=.05/gap.sum(0).max();inp=np.zeros(n);inp[afd]=1
rng=np.random.default_rng(91821);u=np.empty((112,120))
for row in u:
 t=0
 while t<120:
  length=int(rng.integers(3,10));row[t:t+length]=rng.normal(0,.35);t+=length
y=u-np.roll(u,5,axis=1);y[:,:5]=0;Y=y[:,10:].reshape(-1);truth=Y[80*110:];zero=np.mean(truth**2)
def getfit(s):
 X=s[:,10:,ria].reshape(-1,2);m,b,c,l=fit(X[:64*110],Y[:64*110],X[64*110:80*110],Y[64*110:80*110]);return m,b,c,l

def score(s,model):
 m,b,c,l=model;X=s[80:,10:,ria].reshape(-1,2);return float(1-np.mean(((X-m)@c+b-truth)**2)/zero)
rows=[]
with threadpool_limits(limits=1):
 for seed in (7,17,27):
  W=graph_and_weights(mask,seed,n); base=simulate(u,W,gap,inp,.025,'heun'); model=getfit(base)
  for label,edges in [('target',target)]+[(f'control-{i+1}',e) for i,e in enumerate(controls)]:
   w=W.copy()
   for i,j in edges: w[i,j]=0
   s=simulate(u,w,gap,inp,.025,'heun'); norm=float(np.sqrt(np.mean((s[80:,:,ria]-base[80:,:,ria])**2)))
   rows.append(dict(seed=seed,condition=label,edges=[[names[i],names[j]] for i,j in edges],baseline=score(base,model),frozen=score(s,model),refit=score(s,getfit(s)),ria_rms_change=norm))
   print(seed,label,flush=True)
(P/'results.json').write_text(json.dumps(dict(protocol=json.loads((P/'protocol.json').read_text()),rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
