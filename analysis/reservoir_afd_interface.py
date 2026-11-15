"""Compare raw and simplified AFD-adaptation inputs in a toy prediction task."""
import json,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from thermotaxis.connectome import load_connectome,rewire_directed
from thermotaxis.sensory_adaptation import ThresholdAdaptation
from fixed_reservoir_pilot import graph_and_weights
from reservoir_stability_stage import simulate
from reservoir_generalization_stage import assess
P=Path('docs/reports/assets/reservoir-afd-interface-2026-11-15');names=json.loads(Path('data/connectome/herm_neurons.json').read_text())['names'];g=load_connectome().select(names);mask=g.chemical>0;n=len(names)
rng=np.random.default_rng(53119);temp=np.empty((112,120))
for row in temp:
 t=0
 while t<120:
  k=int(rng.integers(3,10));row[t:t+k]=rng.normal(0,.007);t+=k
sensor=ThresholdAdaptation();th=np.zeros(112);adapted=np.empty_like(temp)
for t in range(120):
 adapted[:,t]=sensor.response(temp[:,t],th);th=sensor.update(temp[:,t],th,.1)
raw=temp/.05;y=(temp-np.roll(temp,5,axis=1))/.05;y[:,:5]=0
inp=np.zeros(n);inp[[names.index('AFDL'),names.index('AFDR')]]=1;ria=[names.index('RIAL'),names.index('RIAR')];gap=(g.gap>0).astype(float);np.fill_diagonal(gap,0);gap*=.05/gap.sum(0).max();rows=[];baseline=[]
with threadpool_limits(limits=1):
 for interface,u in [('raw',raw),('adapted',adapted)]:
  baseline.append(dict(interface=interface,**assess(u[:,:,None],[u[80:,:,None]],y,64)))
  for name,graph in [('real',mask)]+[(str(i),rewire_directed(mask,i)[0]) for i in (101,104)]:
   for seed in (7,17,27):
    W=graph_and_weights(graph,seed,n);s=simulate(u,W,gap,inp,.025,'heun')
    for read,idx in [('ria',ria),('all',list(range(n)))]:rows.append(dict(interface=interface,graph=name,seed=seed,readout=read,**assess(s[:,:,idx],[s[80:,:,idx]],y,64)))
    print(interface,name,seed,flush=True)
(P/'results.json').write_text(json.dumps(dict(protocol=json.loads((P/'protocol.json').read_text()),baseline=baseline,rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
