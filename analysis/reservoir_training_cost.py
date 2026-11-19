"""Matched minibatch update wall times for fixed versus trainable connections."""
import json,time,hashlib
from pathlib import Path
import numpy as np,torch
from thermotaxis.connectome import load_connectome
from fixed_reservoir_pilot import graph_and_weights
P=Path('docs/reports/assets/reservoir-training-cost-2026-11-15');torch.set_num_threads(1)
names=json.loads(Path('data/connectome/herm_neurons.json').read_text())['names'];g=load_connectome().select(names);mask=g.chemical>0;n=len(names)
gap=(g.gap>0).astype(float);np.fill_diagonal(gap,0);gap*=.05/gap.sum(0).max();G=torch.tensor(gap,dtype=torch.float32);GD=G.sum(0);inp=torch.zeros(n);inp[[names.index('AFDL'),names.index('AFDR')]]=1;ria=[names.index('RIAL'),names.index('RIAR')];M=torch.tensor(mask,dtype=torch.float32)
rng=np.random.default_rng(91821); u=np.empty((112,120))
for row in u:
 t=0
 while t<120:
  length=int(rng.integers(3,10));row[t:t+length]=rng.normal(0,.35);t+=length
U=torch.tensor(u,dtype=torch.float32);Y=U-torch.roll(U,5,1);Y[:,:5]=0

def forward(u,w):
 x=torch.zeros((len(u),n));states=[]
 for t in range(120):
  for _ in range(2):
   def f(z):return (-z+torch.tanh(z@w+z@G-z*GD+u[:,t,None]*inp))/.3
   k=f(x);x=x+.025*(k+f(x+.05*k))
  states.append(x[:,ria])
 return torch.stack(states,dim=1)

rows=[]
for seed in (7,17,27):
 initial=torch.tensor(graph_and_weights(mask,seed,n),dtype=torch.float32); batches=np.random.default_rng(seed+37000).integers(0,64,size=(32,8));
 start=time.perf_counter()
 with torch.no_grad(): cache=forward(U,initial)
 cache_time=time.perf_counter()-start
 for mode in ('fixed','trainable'):
  torch.manual_seed(seed);out=torch.nn.Linear(2,1); torch.nn.init.zeros_(out.weight);torch.nn.init.zeros_(out.bias)
  w=torch.nn.Parameter(initial.clone(),requires_grad=mode=='trainable'); optim=torch.optim.Adam(list(out.parameters())+([w] if mode=='trainable' else []),lr=.003)
  start=time.perf_counter()
  for ids in batches:
   optim.zero_grad();states=cache[ids] if mode=='fixed' else forward(U[ids],w*M); loss=((out(states[:,10:]).squeeze(-1)-Y[ids,10:])**2).mean();loss.backward();optim.step()
  elapsed=time.perf_counter()-start
  with torch.no_grad():
   teststates=cache[80:] if mode=='fixed' else forward(U[80:],w*M);pred=out(teststates[:,10:]).squeeze(-1);truth=Y[80:,10:];mse=torch.mean((pred-truth)**2).item();zero=torch.mean(truth**2).item()
  rows.append(dict(seed=seed,mode=mode,training_seconds=elapsed,cache_seconds=cache_time,heldout_mse=mse,r2_vs_zero=1-mse/zero,last_batch_loss=float(loss.detach()),internal_max_change=float(torch.max(torch.abs(w.detach()-initial)))))
  print(seed,mode,rows[-1]['r2_vs_zero'],elapsed,flush=True)
(P/'results.json').write_text(json.dumps(dict(protocol=json.loads((P/'protocol.json').read_text()),rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
