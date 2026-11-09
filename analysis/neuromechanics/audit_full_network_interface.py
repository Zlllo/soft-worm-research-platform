"""Full anatomical wiring export and bounded native muscle-interface probes.

This runs no neural fitting and simulates no full-neuron dynamics. The named
AS2 impulse is an explicit algebraic wiring test with invented unit efficacy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from thermotaxis.connectome import CORE
from thermotaxis.neuromechanics.anatomical_body import load_anatomical_body_wiring, PlanarMuscleProjection
from thermotaxis.neuromechanics.boyle_body import BoyleBody
from thermotaxis.neuromechanics.boyle_muscle_body import BoyleMuscleBody
from thermotaxis.neuromechanics.boyle_output import curvature, side_lengths


REPO = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main(root, reference):
    if (root / 'design.json').exists():
        raise FileExistsError('preserve the registered interface audit; use a new directory')
    root.mkdir(parents=True, exist_ok=True)
    w = load_anatomical_body_wiring()
    projection = PlanarMuscleProjection.rank_bins(w.muscle_names)
    paths = ['src/thermotaxis/neuromechanics/anatomical_body.py',
        'src/thermotaxis/neuromechanics/boyle_muscle_body.py',
        'analysis/neuromechanics/build_boyle_muscle_worker.py',
        'analysis/neuromechanics/audit_full_network_interface.py',
        'src/thermotaxis/neuromechanics/boyle_body.py',
        'src/thermotaxis/neuromechanics/boyle_output.py',
        'src/thermotaxis/connectome.py', 'data/connectome/cook2019_herm.json',
        'data/connectome/herm_neurons.json', 'tests/test_anatomical_body.py', 'tests/test_boyle_muscle_body.py']
    design = dict(status='declared_before_physical_interface_probes', full_neurons=302,
        target_branch='all302 -> explicitly parameterized anatomical NMJ -> 95 named muscles -> declared planar projection -> physical body',
        current_scope='full anatomical arrays and muscle interface, not full-neuron dynamics or thermotaxis',
        tests=dict(legacy_drive_replay_s=3., tick_s=.001, probe_duration_s=2., sample_dt_s=.02,
                   cases=['zero', 'head_dorsal', 'head_ventral', 'bilateral', 'AS2_algebraic_impulse']),
        projection='provisional symmetric rank bins, not reconstructed muscle coordinates; rank48; fixed missing-MVL24 contribution zero',
        sources_sha256={p: sha(REPO / p) for p in paths},
        native_sha256={'external_build': sha(root / 'build/build.json'),
                       'external_worker': sha(root / 'build/worker'),
                       'reference_build': sha(reference.with_name('build.json')),
                       'reference_worker': sha(reference)})
    write(root / 'design.json', design)
    np.savez_compressed(root / 'anatomical-wiring.npz', neuron_chemical=w.neuron_chemical,
        neuron_gap=w.neuron_gap, nmj_chemical=w.nmj_chemical, neuron_muscle_gap=w.neuron_muscle_gap,
        muscle_chemical=w.muscle_chemical, muscle_gap=w.muscle_gap, planar_projection=projection.matrix)
    names = dict(neurons=list(w.neuron_names), muscles=list(w.muscle_names),
        effective_drive_order='segment0 dorsal,segment0 ventral,...,segment47 ventral',
        absent_anatomical_muscle='MVL24', anatomical_provenance=w.provenance)
    write(root / 'names.json', names)
    source_ids = np.flatnonzero(w.nmj_chemical.any(axis=1))
    wiring = dict(neurons=302, body_wall_muscles=95, neuron_chemical_entries=int(np.count_nonzero(w.neuron_chemical)),
        neuron_gap_entries=int(np.count_nonzero(w.neuron_gap)), nmj_chemical_entries=int(np.count_nonzero(w.nmj_chemical)),
        neuron_muscle_gap_pairs=int(np.count_nonzero(w.neuron_muscle_gap)),
        muscle_chemical_entries=int(np.count_nonzero(w.muscle_chemical)),
        muscle_gap_entries=int(np.count_nonzero(w.muscle_gap)),
        direct_chemical_nmj_sources=len(source_ids), nmj_source_names=[w.neuron_names[i] for i in source_ids],
        muscle_targets_with_chemical_input=int(np.count_nonzero(w.nmj_chemical.any(axis=0))),
        old_core_direct_nmj_sources=[n for n in CORE if w.nmj_chemical[w.neuron_names.index(n)].any()],
        planar_projection_rank=int(np.linalg.matrix_rank(projection.matrix)),
        retained_gap_arrays='exported; not active in algebraic chemical test; require muscle membrane potentials')
    write(root / 'wiring-audit.json', wiring)

    started = time.monotonic()
    largest = dict(pose=0., muscle=0., stretch=0., side_length=0.)
    saved = []
    with BoyleBody(reference) as old, BoyleMuscleBody(root / 'build/worker') as new:
        old.reset(); new.reset()
        for i in range(3000):
            a = old.step(1); b = new.step(1, muscle_drive=a.neural)
            for key in largest:
                largest[key] = max(largest[key], float(np.max(abs(getattr(a, key) - getattr(b, key)))))
            if i % 20 == 19:
                saved.append((a.time_s, a.pose.copy(), b.pose.copy(), a.neural.copy()))
        assert max(largest.values()) == 0, largest
    np.savez_compressed(root / 'legacy-replay.npz', time_s=np.array([s[0] for s in saved]),
        reference_pose=np.stack([s[1] for s in saved]), external_pose=np.stack([s[2] for s in saved]),
        commands=np.stack([s[3] for s in saved]))
    replay = dict(duration_s=3., updates=3000, max_abs_difference=largest, wall_s=time.monotonic()-started,
                  interpretation='same input preserves original muscle/mechanical evolution, not a new anatomical controller')

    zero = np.zeros((48, 2)); dorsal = zero.copy(); dorsal[:8, 0] = .2
    ventral = zero.copy(); ventral[:8, 1] = .2
    bilateral = zero + .1
    activity = np.zeros(302); activity[w.neuron_names.index('AS2')] = 1
    # Unit mask / fixed gain are deliberately only a connectivity transmission probe.
    efficacy = .2 * (w.nmj_chemical > 0).astype(float)
    muscle_input = w.chemical_muscle_input(activity, efficacy=efficacy)
    as2 = projection.to_segments(muscle_input)
    probes = []
    for name, command in [('zero', zero), ('head_dorsal', dorsal), ('head_ventral', ventral),
                          ('bilateral', bilateral), ('AS2_algebraic_impulse', as2)]:
        start = time.monotonic()
        with BoyleMuscleBody(root / 'build/worker') as body:
            states = [body.reset()]
            for _ in range(100):
                states.append(body.step(20, muscle_drive=command))
        pose = np.stack([s.pose for s in states]); lengths = np.stack([s.side_length for s in states])
        shape_error = float(np.max(abs(lengths - side_lengths(pose))))
        ahead = max(s.internal_time_s - s.time_s for s in states)
        assert shape_error < 1e-17 and ahead < 1e-10
        np.savez_compressed(root / (name+'.npz'), time_s=np.array([s.time_s for s in states]), pose=pose,
            muscle=np.stack([s.muscle for s in states]), commanded_drive=command, side_length=lengths)
        probes.append(dict(name=name, max_abs_curvature_per_m=float(abs(curvature(pose)).max()),
            final_head_xy_m=pose[-1, 0, :2].tolist(), max_geometry_error_m=shape_error,
            max_solver_ahead_s=ahead, nonzero_drive_channels=int(np.count_nonzero(command)),
            wall_s=time.monotonic()-start, complete=True))
    for path, h in design['sources_sha256'].items():
        assert sha(REPO / path) == h, path
    write(root / 'summary.json', dict(complete=True, wiring=wiring, replay=replay, probes=probes,
        new_neural_training=False, full302_neuron_dynamics_run=False,
        limits=['NMJ signs/efficacies are not inferred from EM counts.',
                'AS2 pulse tests naming and propagation with explicitly synthetic efficacy, not physiological activity.',
                'Planar rank-bin projection loses left/right differences and lacks measured attachment locations.',
                'Electrical NMJ and muscle coupling are exported but not implemented dynamically.',
                'External worker disables legacy locomotor neurons; stretch remains an observation, not automatic feedback.']))
    print(json.dumps(dict(wiring=wiring, replay=replay, physical_probe_count=len(probes)), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('results/full-network-body-interface-2026-11-05'))
    p.add_argument('--reference', type=Path, default=Path('results/neuromechanics-stepper-2026-10-27/build-v2/worker'))
    a = p.parse_args()
    main(a.root, a.reference)
