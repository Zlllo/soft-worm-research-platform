"""Count-weighted layout comparison under raw and engineering AFD inputs."""
import json,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from thermotaxis.connectome import load_connectome,rewire_directed
from thermotaxis.sensory_adaptation import ThresholdAdaptation
from reservoir_stability_stage import simulate
from reservoir_generalization_stage import assess
P=Path('docs/reports/assets/reservoir-structural-afd-2026-11-15');names=json.loads(Path('data/connectome/herm_neurons.json').read_text())['names'];g=load_connectome().select(names);mask=g.chemical>0;n=len(names)
weights=[('real',np.log1p(g.chemical))]
for seed in range(201,211):
 graph=rewire_directed(mask,seed)[0];rng=np.random.default_rng(92000+seed);w=np.zeros((n,n))
 for i in range(n):w[i,np.flatnonzero(graph[i])]=rng.permutation(np.log1p(g.chemical[i,mask[i]]))
 weights.append((str(seed),w))
norm=max(w.sum(0).max() for _,w in weights)
rng=np.random.default_rng(53119);temp=np.empty((112,120))
for row in temp:
 t=0
 while t<120:
  k=int(rng.integers(3,10));row[t:t+k]=rng.normal(0,.007);t+=k
sensor=ThresholdAdaptation();th=np.zeros(112);adapt=np.empty_like(temp)
for t in range(120):adapt[:,t]=sensor.response(temp[:,t],th);th=sensor.update(temp[:,t],th,.1)
y=(temp-np.roll(temp,5,axis=1))/.05;y[:,:5]=0
inp=np.zeros(n);inp[[names.index('AFDL'),names.index('AFDR')]]=1;ria=[names.index('RIAL'),names.index('RIAR')];gap=(g.gap>0).astype(float);np.fill_diagonal(gap,0);gap*=.05/gap.sum(0).max();rows=[];baseline=[]
with threadpool_limits(limits=1):
 for interface,u in [('raw',temp/.05),('adapted',adapt)]:
  baseline.append(dict(interface=interface,**assess(u[:,:,None],[u[80:,:,None]],y,64)))
  for label,w in weights:
   for seed in (7,17,27):
    W=w*np.random.default_rng(seed).choice([-1.,1.],size=n)[:,None]/norm;s=simulate(u,W,gap,inp,.025,'heun')
    for read,idx in [('ria',ria),('all',list(range(n)))]:rows.append(dict(interface=interface,graph=label,seed=seed,readout=read,**assess(s[:,:,idx],[s[80:,:,idx]],y,64)))
   print(interface,label,flush=True)
(P/'results.json').write_text(json.dumps(dict(protocol=json.loads((P/'protocol.json').read_text()),baseline=baseline,normalization=float(norm),rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
