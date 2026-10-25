"""Check whether full-I/O tracking needs inter-node transmission.

Separate frozen lesions from scratch-trained isolated-node models. A retained
self-loop is local memory, not communication between different nodes.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import torch

from compare_tracking_topology import make_model, fit, subset
from tracking_learning_surface import atomic_json, cache_data
from train_tracking_network import dataset_digest
from thermotaxis.tracking_pathways import with_pathways
from thermotaxis.tracking_learning import NeuralTrackingController
from thermotaxis.isothermal_scene import TrackingConfig, run_tracking
from thermotaxis.phased_field import PhasedTemperatureField, phased_tracking_metrics


OUT=Path('results/tracking-pathways-2026-10-25')
PARENT=Path('results/tracking-expanded-graphs-2026-10-24')
SOURCES=['analysis/probe_tracking_pathways.py','src/thermotaxis/tracking_pathways.py',
    'analysis/compare_tracking_topology.py','analysis/tracking_learning_surface.py','analysis/train_tracking_network.py',
    'src/thermotaxis/tracking_circuit.py','src/thermotaxis/tracking_learning.py',
    'src/thermotaxis/isothermal_scene.py','src/thermotaxis/phased_field.py',
    'src/thermotaxis/moving_isotherm.py','src/thermotaxis/connectome.py']


def cases(model,plan):
    cells=[]
    for scene in ['static','moving']:
        for phase in ([0.] if scene=='static' else plan['phases_rad']):
            for field in ['linear','radial']:
                c=replace(TrackingConfig(),field=field,uniform_wave_amplitude_K=0. if scene=='static' else .05,
                    uniform_wave_period_s=60.)
                if field=='radial':
                    import math
                    c=replace(c,target_radius_m=c.target_radius_m-c.uniform_wave_amplitude_K*math.sin(phase)/c.gradient_K_m)
                for seed in plan['case_seeds']:
                    row=run_tracking(c,PhasedTemperatureField(NeuralTrackingController(model),c,phase),seed,record_trajectory=True)
                    late=dict(row,trajectory=row['trajectory'][round(20/c.physics_dt_s):])
                    cells.append(dict(scene=scene,phase_rad=phase,field=field,case_seed=seed,initial_state=row['initial_state'],
                        metrics=phased_tracking_metrics(c,row,phase),late_metrics=phased_tracking_metrics(c,late,phase)))
    return cells


def main():
    torch.set_num_threads(1)
    if OUT.exists():raise RuntimeError('Preserve existing pathway batch.')
    OUT.mkdir(parents=True)
    train,val,fingerprint=cache_data()
    parent=json.loads(Path('results/tracking-topology-fullio-budgets-2026-10-24/results.json').read_text())
    assert fingerprint==parent['data_fingerprint'] and dataset_digest(train)==parent['full_training_data']
    assert dataset_digest(val)==parent['validation_data']
    train=subset(train,16)
    baseline_path=Path('results/tracking-timescale-2026-10-25/results.json')
    baseline=json.loads(baseline_path.read_text());assert baseline['complete']
    fit_records=json.loads((PARENT/'fitting.json').read_text())['runs']
    plan=dict(status='exploratory_full_interface_pathway_dependency',training_seeds=[7,17,27],
        frozen_modes=['no_chemical','no_gap','isolated'],refit_mode='isolated',tau_s=.3,
        episodes_per_field=16,updates=2048,case_seeds=[10501,10502],phases_rad=[0.,1.5707963267948966],
        baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        training_digest=dataset_digest(train),validation_digest=dataset_digest(val),
        semantics=dict(no_chemical='All chemical edges, including self loops, removed; gap retained.',
            no_gap='Gap removed; chemical unchanged.',isolated='Off-diagonal chemical and all gap removed; self loops and intrinsic relaxation retained.'),
        limitations=['Real graph only for frozen lesions; no topology advantage claim.',
            'Scratch-trained isolated nodes test existence of a full-I/O alternative, not parameter-matched competition.',
            'Nominal parameter tensors retained but masked entries have zero influence and gradient.',
            'All three isolated fits finish before behavior evaluation.',
            'Reused diagnostic cases are not independent confirmatory tests.'],
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in SOURCES})
    atomic_json(OUT/'design.json',plan)
    for n in SOURCES:
        dest=OUT/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,dest)
    report=dict(design=plan,fits=[],runs=[])
    for seed in plan['training_seeds']:
        torch.manual_seed(seed);model=with_pathways(make_model('real8','demeaned','full'),'isolated')
        curves=fit(model,train,val,2048,seed)
        checkpoint=f'isolated-{seed}-U2048.pt';torch.save(model.state_dict(),OUT/checkpoint)
        chem,gap=model.effective_matrices()
        report['fits'].append(dict(training_seed=seed,checkpoint=checkpoint,curves=curves,
            nominal_parameters=sum(p.numel() for p in model.parameters()),
            effective_chemical_edges=int(torch.count_nonzero(chem)),effective_gap_entries=int(torch.count_nonzero(gap)),
            effective_input_parameters=model.input_weights.numel(),
            masked_chemical_entries=int((model.pre!=model.post).sum()),masked_gap_parameters=model.gap_logits.numel()))
        atomic_json(OUT/'results.json',report);print('PATHWAY FIT',seed,flush=True)
    report['fitting_complete']=True;atomic_json(OUT/'results.json',report)
    for seed in plan['training_seeds']:
        intact=make_model('real8','demeaned','full')
        parent_row=next(r for r in fit_records if r['model']=='real8' and r['seed']==seed)
        source=PARENT/parent_row['endpoints']['2048']['checkpoint']
        dest=OUT/f'parent-real8-{seed}.pt';shutil.copyfile(source,dest)
        intact.load_state_dict(torch.load(dest,map_location='cpu',weights_only=True));intact.eval()
        base=next(r for r in baseline['runs'] if r['model']=='real8' and r['training_seed']==seed and r['tau_s']==.3)
        report['runs'].append(dict(training_seed=seed,mode='intact',treatment='reused_baseline',conditions=base['conditions']))
        for mode in plan['frozen_modes']:
            model=with_pathways(intact,mode)
            report['runs'].append(dict(training_seed=seed,mode=mode,treatment='frozen',conditions=cases(model,plan)))
            atomic_json(OUT/'results.json',report);print('PATHWAY FROZEN',seed,mode,flush=True)
        model=with_pathways(make_model('real8','demeaned','full'),'isolated')
        fitted=next(r for r in report['fits'] if r['training_seed']==seed)
        model.load_state_dict(torch.load(OUT/fitted['checkpoint'],map_location='cpu',weights_only=True));model.eval()
        report['runs'].append(dict(training_seed=seed,mode='isolated',treatment='scratch_refit',conditions=cases(model,plan)))
        atomic_json(OUT/'results.json',report);print('PATHWAY REFIT CASES',seed,flush=True)
    assert len(report['runs'])==15
    report['complete']=True
    report['source_hashes_unchanged']={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in plan['source_sha256'].items()}
    assert all(report['source_hashes_unchanged'].values())
    atomic_json(OUT/'results.json',report)


if __name__=='__main__':main()
