#!/usr/bin/env python3
"""Independent saved-metric reconstruction and one full core replay."""
import hashlib,json,math
from pathlib import Path
from thermotaxis.phase2_control import ControlConfig,run_four_action
from thermotaxis.legacy_policy import FrozenQTable, LegacyGridMap

root=Path(__file__).resolve().parents[1]
archive=root/'docs/reports/assets/phase2-four-action-2026-10-04'
out=root/'results/phase2-four-action-2026-10-04'
summary=json.loads((out/'summary.json').read_text())
errors=[]
for entry in summary['cardinal']:
    result=json.loads((out/(entry['name']+'.json')).read_text())
    rows=result['trajectory'];cfg=result['config'];dt=cfg['physics_dt_s']
    expected=(math.pi/2,-math.pi/2,math.pi,0)[entry['action']]
    wrap=lambda a:math.atan2(math.sin(a),math.cos(a))
    error=[abs(wrap(expected-r['axis_heading_rad'])) for r in rows]
    window=round(1/dt)+1
    settling=next((rows[i]['time_s'] for i in range(len(rows)-window+1) if max(error[i:i+window])<=.25),None)
    tail=[e for r,e in zip(rows,error) if r['time_s']>=rows[-1]['time_s']-4]
    actual={'final_x_mm':rows[-1]['x_m']*1e3,'final_y_mm':rows[-1]['y_m']*1e3,'dissipation_pJ':rows[-1]['dissipation_J']*1e12,'tail_mean_abs_axis_error_rad':sum(tail)/len(tail),'first_1s_within_025rad_s':settling}
    assert all(math.isclose(v,entry[k],rel_tol=1e-12,abs_tol=1e-14) for k,v in actual.items())
    assert all(b['dissipation_J']>=a['dissipation_J'] for a,b in zip(rows,rows[1:]))
    assert all(c['action']==entry['action'] for c in result['controls'])
    assert len(result['samples'])==round(cfg['duration_s']/cfg['sensor_dt_s'])+1
    errors.append({'name':entry['name'],'independently_recomputed':actual})
left=json.loads((out/'left.json').read_text())
replay=run_four_action(ControlConfig.from_dict(left['config']),lambda obs:2)
assert replay==left
rest=json.loads((out/'never_started.json').read_text())
assert all(r['power_W']==0 and r['dissipation_J']==0 and r['x_m']==0 and r['y_m']==0 for r in rest['trajectory'])
stop=json.loads((out/'relax_after_stop.json').read_text())
assert stop['controls'][20]['time_s']==4 and stop['controls'][20]['action'] is None
assert stop['trajectory'][200]['amplitude_rad']>0
assert stop['trajectory'][-1]['amplitude_rad']<1e-10
frozen_saved=json.loads((out/'frozen_qtable.json').read_text())
fs=json.loads((out/'frozen-policy-summary.json').read_text())
asset=root/fs['policy_asset']
assert hashlib.sha256(asset.read_bytes()).hexdigest()==fs['asset_sha256']
policy=FrozenQTable.from_json(asset,tie_seed=44001)
grid=LegacyGridMap(policy.width,policy.height,.004,.004,y_axis_up=True)
before=hashlib.sha256(policy.q_table.tobytes()).hexdigest()
frozen_replay=run_four_action(ControlConfig.from_dict(frozen_saved['config']),lambda indices:policy.select_action(*indices),legacy_observer=lambda obs,head:grid.indices(head),terminal_condition=lambda head,t:any(x<0 or x>.004 for x in head))
assert frozen_replay==frozen_saved
assert before==hashlib.sha256(policy.q_table.tobytes()).hexdigest()==fs['q_values_sha256_after']==fs['q_values_sha256_before']
assert fs['controls_executed']==len(frozen_saved['controls'])==44
assert fs['terminal']==frozen_saved['termination']
assert fs['final']==frozen_saved['trajectory'][-1]
assert frozen_saved['trajectory'][-2]['head_x_m']>=0 and frozen_saved['trajectory'][-1]['head_x_m']<0
manifest=json.loads((out/'manifest.json').read_text())
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in manifest['sources_sha256'].items())
archive.mkdir(parents=True,exist_ok=True)
result={'four_cardinal_metrics':errors,'full_24s_left_replay_exact':True,'never_started_rest_exact':True,'finite_stop_verified':True,'frozen_policy_full_replay_exact':True,'frozen_policy_weight_hash_unchanged':True,'first_sampled_domain_exit_s':fs['terminal']['time_s'],'manifest_all_sources_current':True,'replay_source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['src/thermotaxis/phase2_control.py','src/thermotaxis/phase2_body.py','analysis/validate_phase2_bridge.py']}}
(archive/'independent-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'cardinal_recomputed':4,'full_replay_s':24,'rest_and_stop':True}))
