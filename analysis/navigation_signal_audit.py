"""Posthoc fixed-window signal assay; no retraining or score-based selection.

All raw actors, BC and final checkpoints, common near-band teacher windows.
Add a1mK common temperature shift or phase-correlated fluctuation while keeping
features algebraically consistent. Measures signal use, not optimality or biology.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from navigation_window_ppo import feature_data
from thermotaxis.navigation_window_policy import WindowActorCritic


ROOT=Path('results/navigation-factorial-2026-10-25');OUT=Path('results/navigation-signal-audit-2026-10-25')


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1)
    original=feature_data('raw');values=original['x'];eligible=np.flatnonzero((values[:,:,6].sum(1)==41)&(np.abs(values[:,-1,1])<1.))
    rng=np.random.default_rng(26801);chosen=rng.choice(eligible,min(512,len(eligible)),replace=False)
    x=torch.from_numpy(values[chosen]);shift=x.clone();shift[:,:,1]+=.001/.05
    phase=x.clone();delta=(.001/.05)*x[:,:,4];average=delta.mean(1,keepdim=True)
    phase[:,:,0]+=delta-average;phase[:,:,1]+=average
    plan=dict(status='posthoc_signal_diagnostic',windows=len(chosen),selection_seed=26801,temperature_delta_K=.001,
        data='shared training teacher windows, not unseen behavior cases',eligible='full history and mean temperature within comfort band',
        selected_indices=chosen.tolist(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'design.json').write_text(json.dumps(plan,indent=2)+'\n')
    specs=json.loads((ROOT/'design.json').read_text())['specs'];rows=[]
    for s in specs:
        if s['encoding']!='raw':continue
        record=json.loads((ROOT/f'fit-{s["id"]:03d}'/'record.json').read_text())
        model=WindowActorCritic(s['name'],s['interface'])
        for budget in ['0','262144']:
            model.load_state_dict(torch.load(ROOT/f'fit-{s["id"]:03d}'/record['endpoints'][budget]['checkpoint'],map_location='cpu',weights_only=True));model.eval()
            with torch.no_grad():
                base=.3*torch.tanh(model.actor_mean(x));level=.3*torch.tanh(model.actor_mean(shift));sweep=.3*torch.tanh(model.actor_mean(phase))
            rows.append(dict(spec=s,learner_interactions=int(budget),command_std_rad_s=float(base.std(unbiased=False)),
                one_mK_level_rms_change_rad_s=float((level-base).square().mean().sqrt()),
                one_mK_phase_rms_change_rad_s=float((sweep-base).square().mean().sqrt()),
                teacher_mse_normalized=float(((base/.3-torch.from_numpy(original['y'][chosen]))**2).mean())))
    result=dict(design=plan,runs=rows,complete=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print('signal assays',len(rows),flush=True)


if __name__=='__main__':main()
