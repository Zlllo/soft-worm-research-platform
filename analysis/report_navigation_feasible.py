"""Review and report the physics-motivated replication without selecting graphs."""
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path('results/navigation-feasible-2026-10-25');OUT=Path('docs/reports/assets/navigation-study-2026-10-25')


def main():
    d=json.loads((ROOT/'results.json').read_text());assert d['complete'] and len(d['runs'])==42
    assert all(d['source_hashes_unchanged'].values())
    reference=json.loads((ROOT/'reference-controls.json').read_text());groups=d['groups']
    for group in groups.values():
        for b in ['0','65536','262144']:assert group[b]['ruled_out_good_event_cases']==0
    names=['real8']+['rewired8-'+str(x) for x in [11,22,33,44,55,66,77,88,99,110]]+['isolated8','gru2','gru16']
    real=groups['real8']['262144']['complete_success'];null=[groups[n]['262144']['complete_success'] for n in names[1:11]]
    reached={}
    cross={}
    for name in names:
        rows=[r for r in d['runs'] if r['spec']['name']==name];counts=[];paired=[]
        for r in rows:
            first=None
            for endpoint in r['primary']:
                m=endpoint['summary']
                if m['complete_success']>=.75 and m['post_tracking']>=.5:
                    first=38400+endpoint['learner_interactions'];break
            counts.append(first)
            cases=next(x['cases'] for x in r['primary'] if x['learner_interactions']==262144)
            sub=[x for x in cases if x['seed'] in d['design']['cross_seeds']]
            paired.append(float(np.mean([x['complete_success'] for x in sub])))
        reached[name]=counts
        cross[name]={**groups[name]['cross'],'trained15_test15':dict(complete_success=float(np.mean(paired)),ruled_out_good_event_cases=0)}
    reviewed=dict(complete=True,real_final=real,null_graph_mean=float(np.mean(null)),null_graphs_above_real=sum(x>real for x in null),
        threshold_total_interactions=reached,cross=cross,reference=reference,
        all_primary_good_event_states_not_ruled_out_by_speed_bound=True,
        group_results=groups,rollouts=sum(len(p['cases']) for r in d['runs'] for p in r['primary'])+
            sum(len(cases) for r in d['runs'] for cases in r['cross'].values()))
    (ROOT/'reviewed-summary.json').write_text(json.dumps(reviewed,indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained');x=np.array([38400,103936,300544])/1000
    for ax,metric,title in zip(axes,['complete_success','tracking'],['Complete navigation','Tracking time']):
        null_curves=np.array([[groups[n][str(b)][metric]*100 for b in [0,65536,262144]] for n in names[1:11]])
        for y in null_curves:ax.plot(x,y,color='#737b83',alpha=.25,linewidth=.8)
        ax.fill_between(x,null_curves.min(0),null_curves.max(0),color='#737b83',alpha=.08)
        ax.plot(x,null_curves.mean(0),color='#737b83',label='10 rewired: mean / range',marker='.')
        for n,color,label in [('real8','#2c5c9b','Real'),('gru16','#bf8b22','GRU16'),('isolated8','#6a7740','Isolated')]:
            values=[groups[n][str(b)][metric]*100 for b in [0,65536,262144]]
            ax.plot(x,values,color=color,label=label,marker='o',linewidth=1.8)
            for xi,b in zip(x,[0,65536,262144]):
                m='complete_success' if metric=='complete_success' else 'full_tracking'
                vals=[v[m]*100 for v in groups[n][str(b)]['by_seed'].values()]
                ax.vlines(xi,min(vals),max(vals),color=color,alpha=.4)
        reference_metric='complete_success' if metric=='complete_success' else 'full_tracking'
        ax.axhline(reference['reference15']['summary'][reference_metric]*100,color='#222222',linestyle=':',label='Local reference')
        ax.set_xscale('log');ax.minorticks_off();ax.set_xticks(x,['38.4','103.9','300.5']);ax.set_ylim(0,103)
        ax.set_xlabel('Teacher + learner interactions (thousands)');ax.set_ylabel('Percent');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(.01,.89));fig.suptitle('Geometry-controlled replication: 15° field turn, new cases',fontsize=14)
    OUT.mkdir(parents=True,exist_ok=True);fig.savefig(OUT/'feasible-learning.png',dpi=170);fig.savefig(OUT/'feasible-learning.svg')
    p=OUT/'feasible-learning.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n');plt.close(fig)
    p=[];p.append('# 可达性控制后的完整导航复查\n\n日期：2026-10-25。此为发现原任务几何问题后开展的后验物理复查，使用全新病例，不伪装成原始预登记结果。')
    p.append('## 为什么追加\n\n原场绕固定原点转向60°时，沿旧等温线走得更远的个体，可能离新目标太远。即使朝目标直走，也可能没有时间进入并保持10秒追踪。因此较好的前段表现会让后段面对更难的几何条件，原完整成功率不宜单独解释为恢复能力。')
    p.append('最乐观的时间下界只考虑距离和速度，不计转向或感觉延迟：\n\n$$t_{\\min}=\\frac{\\max(0,(|T_c-T^*|-\\Delta T)/G)}{v}+t_{\\mathrm{stable}}.\\tag{1}$$\n\n超过剩余时间即可判定无法完成；低于剩余时间并不证明一定可控。')
    p.append('预定例图中，真实图种子7在病例61001的下界加稳定时长为118.5秒，剩余113.2秒。该例未按成绩挑选。另有两个独立几何测试，分别复现原60°反例和15°范围下的速度下界。')
    p.append('## 复查规则\n\n保留原始105组全部结果。追加全部full／raw模型：真实图、10张重排图、隔离节点、GRU2／16，各3个训练种子，共42组RL训练。直接复用原32条示范、2048更新后的同一检查点，不重复计算示范拟合次数；PPO算法、预算和训练随机种子不变，只把场转角改为15°。径向条件不变。')
    p.append('新病例71001—71032在42组训练全完成后才评估。另用前16个同源病例交叉检查训练60°／15°与测试60°／15°，避免把不同病例平均值误认为几何因果效应。学习数据仍包括共同38,400次教师交互，追加65,536／262,144次学习者交互。')
    p.append(f"同一局部参考控制器，在这32个病例的60°条件完整成功{reference['reference60']['summary']['complete_success']*100:.2f}%，有{reference['reference60']['ruled_out_good_cases']}个原先处于舒适带的病例被速度下界排除；15°条件完整成功{reference['reference15']['summary']['complete_success']*100:.2f}%，排除数为{reference['reference15']['ruled_out_good_cases']}。追加实验所有原先处于舒适带的病例均未被该速度下界排除，但这仍不是所有连续控制可达性的数学证明。")
    p.append('## 全部模型的固定端点\n\n| 模型 | 共同示范后成功 | +65,536次交互成功 | +262,144次交互成功 | 最终追踪比例 | 达标种子数 |\n| --- | --- | --- | --- | --- | --- |')
    for n in names:
        a=groups[n];p.append('| '+n+' | '+' | '.join(f"{a[str(b)]['complete_success']*100:.2f}%" for b in [0,65536,262144])+f" | {a['262144']['tracking']*100:.2f}% | {sum(v is not None for v in reached[n])}/3 |")
    p.append(f"真实图最终成功均值{real*100:.2f}%，10张重排图图均值的平均为{np.mean(null)*100:.2f}%，其中{sum(x>real for x in null)}/10张高于真实图。结果限定在该任务、共同示范和固定优化预算，不是完整生物回路比较。")
    initial_null=float(np.mean([groups[n]['0']['complete_success'] for n in names[1:11]]))
    initial_real=groups['real8']['0']['complete_success']
    seed_scores={seed:round(value['complete_success']*100,2) for seed,value in groups['real8']['262144']['by_seed'].items()}
    p.append(f"真实图的三个训练种子最终分别为{seed_scores}，均值领先存在明显种子波动。进入PPO前真实图已经为{initial_real*100:.2f}%，重排平均{initial_null*100:.2f}%；PPO阶段真实图增加{(real-initial_real)*100:.2f}个百分点，重排平均增加{(np.mean(null)-initial_null)*100:.2f}个百分点。因此当前差异主要在共同示范初始化后已存在，不能解释成已验证的额外RL样本效率优势。")
    p.append('![几何控制后的学习曲线](assets/navigation-study-2026-10-25/feasible-learning.png)\n\n竖线为3个训练种子范围，灰带为10张图均值的范围，不是置信区间。各轨迹仍只对应32个共同初态种子。')
    p.append('## 配对的训练／测试几何\n\n下表仅用同16个新病例；不要与上面32病例均值直接相减。\n\n| 模型 | 训练60°测试60° | 训练60°测试15° | 训练15°测试60° | 训练15°测试15° |\n| --- | --- | --- | --- | --- |')
    for n in ['real8','gru16','isolated8']:
        p.append('| '+n+' | '+' | '.join(f"{cross[n][k]['complete_success']*100:.2f}%" for k in ['trained60_test60','trained60_test15','trained15_test60','trained15_test15'])+' |')
    p.append('## 已知边界与复现\n\n此处只复查full／raw，原接口与适应矩阵仍按60°压力条件解释。没有重新选择最佳图、最佳种子或最佳检查点；原始压力数据和失败均保留。生物功能校准、从零RL以及RFT／弹性身体仍未由本轮验证。完整回归225项通过，14条既有依赖弃用警告。')
    p.append('```bash\nPYTHONPATH=src python3 analysis/navigation_feasibility_replication.py prepare\nPYTHONPATH=src python3 analysis/navigation_feasibility_replication.py fit\nPYTHONPATH=src python3 analysis/navigation_feasibility_replication.py evaluate\nPYTHONPATH=src python3 analysis/navigation_feasibility_replication.py collect\nPYTHONPATH=src python3 analysis/navigation_feasible_reference.py\nMPLCONFIGDIR=/tmp/worm-matplotlib PYTHONPATH=src python3 analysis/report_navigation_feasible.py\n```\n\n实际运行的3个分片互不重叠，全部训练结束后才评估。归档含原始模型、新模型、源代码、设计和审计结果。')
    p.append('[完整材料](assets/navigation-study-2026-10-25/README.md)、[原始压力比较](navigation-study-2026-10-25.md)、[协议](../navigation-protocol-2026-10-25.md)。')
    text='\n\n'.join(p)+'\n';text=re.sub(r'(\|[^\n]*\|)\n\n(?=\|)',r'\1\n',text)
    Path('docs/reports/navigation-feasible-2026-10-25.md').write_text(text)
    print('reviewed feasible real',real,'null mean',float(np.mean(null)),'above',sum(x>real for x in null),'gru16',groups['gru16']['262144']['complete_success'],flush=True)


if __name__=='__main__':main()
