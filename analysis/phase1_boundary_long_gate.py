"""Supplementary full-horizon acceleration gate, separate from fixed experiment."""
from pathlib import Path
from dataclasses import replace
import hashlib,json
from analysis.phase1_boundary_stationarity import config,OUT
from thermotaxis.phase1 import run_phase1_experiment
rows=[]
for gain in [0,4]:
    cfg=config(.06,.007,99888)
    cfg=replace(cfg,thermotaxis_controller=replace(cfg.thermotaxis_controller,response_gain_per_K_s=gain))
    outputs=[run_phase1_experiment(replace(cfg,time=replace(cfg.time,physics_dt_s=h)),record_trajectory=False)['metrics'] for h in [.01,.1]]
    differences={k:abs(outputs[0][k]-outputs[1][k]) for k in ['first_entry_time_s','comfort_residence_fraction','rms_temperature_error_K','turn_count']}
    assert max(differences.values())<1e-8
    rows.append({'gain':gain,'config_at_accelerated_step':cfg.to_dict(),'physics_dt_s':[.01,.1],'metrics':outputs,'absolute_differences':differences,'passed':True})
path=Path(__file__)
payload={'command':'MPLCONFIGDIR=/private/tmp/softworm-boundary-mpl PYTHONPATH=src:. python3 analysis/phase1_boundary_long_gate.py','script_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'fresh_seed_index':99888,'rows':rows}
(OUT/'long-horizon-gate.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps([{ 'gain':r['gain'], 'absolute_differences':r['absolute_differences']} for r in rows]))
