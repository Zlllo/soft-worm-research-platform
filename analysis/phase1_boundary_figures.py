"""Regenerate legible figures from completed boundary summary; no simulation rerun."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/phase1-boundary-stationarity-2026-10-04'
summaries=json.loads((OUT/'summary.json').read_text())
for name,mode in [('late-occupation','blocks'),('final-initial-side-occupation','initial')]:
 fig,axes=plt.subplots(2,3,figsize=(13,6),sharex=True)
 for col,s in enumerate(summaries):
  L=s['length_m'];edges=np.linspace(-L/2*1000,L/2*1000,61)
  for row,g in enumerate(['response','no_response']):
   ax=axes[row,col];gg=s['groups'][g]
   series=[(f'{1800+j*600}–{2400+j*600} s',h)for j,h in enumerate(gg['mean_occupation_by_block'])] if mode=='blocks' else [(f'{side} start',gg[f'{side}_mean_occupation_by_block'][-1])for side in ['cold','hot']]
   for label,h in series:ax.stairs(np.array(h)/(L/60*1000),edges,label=label,alpha=.85)
   if g=='no_response':ax.plot([-L/2*1000,L/2*1000],[1/(L*1000)]*2,'k--',linewidth=1,label='uniform baseline')
   ax.axvspan(-2.5,2.5,color='green',alpha=.1);ax.set_title(f'{g.replace("_"," ")} | L={L*1000:.0f} mm');ax.set_ylabel('Occupation density (1/mm)');ax.grid(alpha=.2);ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
   if row:ax.set_xlabel('Position relative to comfort center (mm)')
 axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8);fig.tight_layout()
 for suffix in ['png','svg']:fig.savefig(OUT/f'{name}.{suffix}',dpi=180)
 plt.close(fig)
p=Path(__file__)
(OUT/'figure-manifest.json').write_text(json.dumps({'script_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'input_summary_sha256':hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest(),'command':'MPLCONFIGDIR=/private/tmp/softworm-boundary-mpl python3 analysis/phase1_boundary_figures.py'},indent=2)+'\n')
