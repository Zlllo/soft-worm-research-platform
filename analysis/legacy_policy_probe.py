"""True old Worm2D training ONLY to demonstrate a frozen policy interface.

The generated table is newly trained, never described as a historical result.
All old program files are read/imported but never edited. No training/replay is
performed during frozen evaluation. Run PYTHONPATH=src python3 this_file.
"""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import random
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'程序'))
from core.environment import Environment2D
from core.worm_body import Worm2D, _action_vectors
from thermotaxis.legacy_policy import FrozenQTable, LEGACY_ACTION_VECTORS


def run(output_dir, *, seed=71031, rounds=60, steps_per_round=120):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    random.seed(seed); np.random.seed(seed)
    width = height = 17
    y,x=np.mgrid[:height,:width]
    field = 20.+100.*np.exp(-((x-8)**2+(y-8)**2)/32.)
    env=Environment2D(field,best_point=(8,8))
    cumulative=None; training_rows=[]
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture):
        for episode in range(rounds):
            # Every cell gets an opportunity to be sampled across varied starts.
            start=(int(np.random.randint(width)),int(np.random.randint(height)))
            worm=Worm2D(start,width,height,body_params={'num_segments':3,'segment_length':3.,'action_size':4},noise_params={'position_noise':0.,'angle_noise':0.,'thermal_noise':0.,'action_noise':0.,'noise_correlation':0.})
            worm.use_neural=False
            if cumulative is not None: worm.q_table=cumulative.tolist()
            for step in range(steps_per_round):
                if not worm.decide_move(env,epsilon=.35,alpha=.15,gamma=.9):
                    raise RuntimeError(f'old Worm2D failed on {episode}/{step}')
            cumulative=np.array(worm.q_table,dtype=float)
            training_rows.append({'episode':episode,'start':start,'final':[worm.x,worm.y],'total_reward':worm.total_reward})
    log=capture.getvalue()
    # Old code catches failures internally; reject visible fallback warnings.
    if any(token in log for token in ('⚠️','❌','超时')):
        raise RuntimeError('legacy training emitted a warning/fallback: inspect imported old environment')
    assert tuple(_action_vectors(4)) == LEGACY_ACTION_VECTORS
    checkpoint={
        'schema':'legacy-four-action-qtable-v1',
        'provenance':'newly_trained_interface_sample_not_historical_checkpoint',
        'training':{'seed':seed,'rounds':rounds,'steps_per_round':steps_per_round,'epsilon':.35,'alpha':.15,'gamma':.9,'state':'absolute_grid_xy','reward':'old_original_temperature_ladder_with_trend','termination':'fixed_steps_no_done'},
        'action_vectors':[list(v) for v in LEGACY_ACTION_VECTORS],
        'temperature_field':field.tolist(),'best_point':[8,8],
        'coordinate_units':'legacy_dimensionless_grid_no_historical_SI_calibration',
        'q_table':cumulative.tolist(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'程序/core/worm_body.py',ROOT/'程序/core/environment.py',ROOT/'程序/core/reward_functions.py',Path(__file__)]}}
    tablepath=output_dir/'new-training-interface-qtable.json'
    tablepath.write_text(json.dumps(checkpoint,ensure_ascii=False,indent=2))
    frozen=FrozenQTable.from_json(tablepath,tie_seed=seed+1)
    saved=cumulative.copy()
    # Verify exact greedy equivalence for every cell, counting possible ties.
    agreement=0; ties=0
    for yi in range(height):
        for xi in range(width):
            a=frozen.select_action(xi,yi); row=cumulative[yi,xi]
            agreement += int(row[a] == row.max())
            ties += int(np.count_nonzero(row == row.max()) > 1)
    evalrows=[]
    for start in [(1,1),(15,1),(1,15),(15,15),(4,8),(12,8)]:
        xi,yi=start; path=[list(start)]; reached=None
        for step in range(120):
            a=frozen.select_action(xi,yi); dx,dy=LEGACY_ACTION_VECTORS[a]
            nx,ny=xi+dx,yi+dy
            if not (0 <= nx < width and 0 <= ny < height):
                nx=max(0,min(width-1,xi-2*dx));ny=max(0,min(height-1,yi-2*dy))
            xi,yi=nx,ny;path.append([xi,yi])
            if reached is None and field[yi,xi] >= 110.: reached=step+1
        evalrows.append({'start':start,'path':path,'first_110_temperature_step':reached,'final_temperature':float(field[yi,xi])})
    assert np.array_equal(frozen.q_table,saved)
    summary={'historical_checkpoint_found':False,'provenance':checkpoint['provenance'],'training_steps':rounds*steps_per_round,'greedy_equivalence_cells':agreement,'total_cells':width*height,'tied_cells':ties,'nonzero_Q_values':int(np.count_nonzero(cumulative)),'frozen_no_weight_mutation':True,'evaluation':evalrows,'note':'Six held-out starts only an interface smoke test; training seed N=1, no performance or biological claim.'}
    (output_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (output_dir/'training-records.json').write_text(json.dumps(training_rows,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='evaluation'},ensure_ascii=False))
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'docs/reports/assets/legacy-policy-bridge-2026-10-04'))
    args=parser.parse_args(); run(args.output)
