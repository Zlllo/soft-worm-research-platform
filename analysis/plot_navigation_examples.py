"""Render predeclared example cases without choosing a winning graph/seed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path('results/navigation-examples-2026-10-25');OUT=Path('docs/reports/assets/navigation-study-2026-10-25')


def main():
    index=json.loads((ROOT/'index.json').read_text());rows=[r for r in index['runs'] if r['status']=='complete']
    fig,axes=plt.subplots(2,len(rows),figsize=(4*len(rows),7),layout='constrained',squeeze=False)
    data={r['name']:dict(np.load(ROOT/r['path'])) for r in rows}
    for field in range(2):
        allx=np.concatenate([v['history'][:,field,3]*1000 for v in data.values()]);ally=np.concatenate([v['history'][:,field,4]*1000 for v in data.values()])
        lo=np.array([min(allx.min(),-6),min(ally.min(),-6)]);hi=np.array([max(allx.max(),6),max(ally.max(),6)]);span=max(hi-lo)*1.12
        middle=(lo+hi)/2
        for col,row in enumerate(rows):
            values=data[row['name']];h=values['history'][:,field];ax=axes[field,col];event=values['event'][field];pre=h[:,0]<event
            angle=values['angle'][field];sign=values['sign'][field]
            if field==0:
                for n,offset,style in [(angle,0.,'--'),(angle+sign*np.pi/3,sign*2.,':')]:
                    normal=np.array([np.cos(n),np.sin(n)]);tangent=np.array([-np.sin(n),np.cos(n)])
                    points=offset*normal+np.array([-span*3,span*3])[:,None]*tangent
                    ax.plot(points[:,0],points[:,1],color='#777777',linestyle=style,linewidth=1.)
            else:
                theta=np.linspace(0,2*np.pi,200)
                for center,style in [(np.zeros(2),'--'),(2*np.array([np.cos(angle),np.sin(angle)]),':')]:
                    ax.plot(center[0]+5*np.cos(theta),center[1]+5*np.sin(theta),color='#777777',linestyle=style,linewidth=1.)
            ax.plot(h[pre,3]*1000,h[pre,4]*1000,color='#2c5c9b',label='Before change')
            ax.plot(h[~pre,3]*1000,h[~pre,4]*1000,color='#bf8b22',label='After change')
            ax.scatter(h[0,3]*1000,h[0,4]*1000,facecolors='none',edgecolors='#222222',s=32,zorder=4)
            at=np.flatnonzero(~pre)[0];ax.scatter(h[at,3]*1000,h[at,4]*1000,marker='x',color='#222222',s=35,zorder=4)
            ax.scatter(h[-1,3]*1000,h[-1,4]*1000,marker='^',color='#222222',s=25,zorder=4)
            ax.set_xlim(middle[0]-span/2,middle[0]+span/2);ax.set_ylim(middle[1]-span/2,middle[1]+span/2);ax.set_aspect('equal')
            ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)');ax.set_title(row['name']+' / '+('linear' if field==0 else 'radial'))
            ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Fixed examples: training seed7, cases61001/61002; final raw/full policies',fontsize=12)
    OUT.mkdir(parents=True,exist_ok=True);fig.savefig(OUT/'example-trajectories.png',dpi=170);fig.savefig(OUT/'example-trajectories.svg')
    p=OUT/'example-trajectories.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');plt.close(fig)


if __name__=='__main__':main()
