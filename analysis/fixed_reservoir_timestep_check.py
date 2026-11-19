"""Post-hoc finer-time-step check, same data and frozen readout between steps."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from thermotaxis.connectome import load_connectome
from fixed_reservoir_pilot import graph_and_weights, simulate, ridge

names = json.loads(Path('data/connectome/herm_neurons.json').read_text())['names']
g = load_connectome().select(names); chem = g.chemical > 0
gap = (g.gap > 0).astype(float); np.fill_diagonal(gap,0); gap *= .05 / max(1,gap.sum(0).max())
inputs = np.zeros(len(names)); inputs[[names.index('AFDL'),names.index('AFDR')]] = 1
rng = np.random.default_rng(4815); u = np.repeat(rng.normal(0,.35,(96,20)),5,axis=1)
y = u-np.roll(u,5,axis=1); y[:,:5]=0
rows=[]
with threadpool_limits(limits=1):
 for seed in (7,17,27):
  W=graph_and_weights(chem,seed,len(names))
  a=simulate(u,W,gap,inputs,.05); b=simulate(u,W,gap,inputs,.025)
  for label, indices in [('ria',[names.index('RIAL'),names.index('RIAR')]),('all',list(range(len(names))))]:
   A=a[:,10:,indices].reshape(-1,len(indices)); B=b[:,10:,indices].reshape(-1,len(indices)); Y=y[:,10:].reshape(-1)
   m,ym,c,p=ridge(A[:5760],Y[:5760],A[5760:7200],Y[5760:7200]); truth=Y[7200:]; zero=np.mean(truth**2)
   scores=[float(1-np.mean(((x[7200:]-m)@c+ym-truth)**2)/zero) for x in (A,B)]
   rows.append(dict(seed=seed,readout=label,score_005=scores[0],score_0025_same_readout=scores[1],penalty=p,
                    state_relative_rms=float(np.sqrt(np.mean((A-B)**2))/max(1e-15,np.sqrt(np.mean(B**2))))))
Path('docs/reports/assets/fixed-reservoir-pilot-2026-11-15/timestep-check.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
