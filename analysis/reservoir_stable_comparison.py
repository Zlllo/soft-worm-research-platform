"""Post-hoc re-evaluation with more accurate integration; old test data reused."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from thermotaxis.connectome import load_connectome,rewire_directed
from fixed_reservoir_pilot import graph_and_weights,evaluate
from reservoir_stability_stage import simulate
p=Path('docs/reports/assets/reservoir-stable-comparison-2026-11-15')
names=json.loads(Path('data/connectome/herm_neurons.json').read_text())['names']; graph=load_connectome().select(names); base=graph.chemical>0
gap=(graph.gap>0).astype(float); np.fill_diagonal(gap,0); gap*=.05/max(1,gap.sum(0).max())
inputs=np.zeros(len(names)); inputs[[names.index('AFDL'),names.index('AFDR')]]=1
rng=np.random.default_rng(4815); u=np.repeat(rng.normal(0,.35,(96,20)),5,axis=1); y=u-np.roll(u,5,axis=1); y[:,:5]=0
rows=[]
with threadpool_limits(limits=1):
 for label,g in [('real',base)]+[(str(i),rewire_directed(base,i)[0]) for i in range(101,106)]:
  for seed in (7,17,27):
   W=graph_and_weights(g,seed,len(names)); start=time.perf_counter(); s=simulate(u,W,gap,inputs,.025,'heun'); elapsed=time.perf_counter()-start
   rows.append(dict(graph=label,seed=seed,seconds=elapsed,ria=evaluate(s[:,:,[names.index('RIAL'),names.index('RIAR')]],y),all_nodes=evaluate(s,y)))
   print(label,seed,flush=True)
(p/'results.json').write_text(json.dumps(dict(rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
