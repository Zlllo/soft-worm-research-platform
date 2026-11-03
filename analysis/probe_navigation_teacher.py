"""Positive control and time-local sensitivity on existing teacher windows.

Reconstructs the actual local-regression teacher from policy-visible features;
uses no position, heading, true gradient, future data, or hidden teacher state.
This supplemental probe was specified after the first diagnostic batch.
"""
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_navigation_failures import ROOT, check, digest, load_model, temperature_perturbation, write


def teacher_command(features):
    x = np.asarray(features, dtype=np.float64)
    if not np.all(x[:, :, 6] == 1):
        raise ValueError('positive control requires full41-sample windows')
    n = len(x)
    t = np.broadcast_to(np.linspace(-1, 0, 41), (n, 41))
    matrix = np.stack([np.ones_like(t), t, x[:, :, 4], x[:, :, 5]], axis=2)
    # Undo float32 baseline subtraction in NavigationFeatures; the teacher
    # originally converted the temperature samples to Python floats first.
    temp = .05 * (x[:, :, 0] + x[:, :, 1]) + float(np.float32(293.15)) - 293.15
    weights = (np.linalg.pinv(matrix) @ temp[:, :, None]).squeeze(-1)
    longitudinal = weights[:, 1] / 4 / .0002
    lateral = weights[:, 2] / .00015
    error = (weights[:, 0] - .0005 * longitudinal) / .05
    return np.clip(-.2 * error * (lateral / 40) - .4 * (longitudinal / 40) * (lateral / 40), -.3, .3)


def main():
    torch.set_num_threads(1)
    plan = json.loads((ROOT / 'design.json').read_text()); check(plan)
    source = Path(__file__)
    out = ROOT / 'teacher-temporal-probe.json'
    if out.exists():
        raise RuntimeError('preserve existing temporal probe')
    extension = dict(status='supplemental_after_initial_diagnosis', amplitude_K=.002,
                     bins=[[0, 10], [10, 20], [20, 30], [30, 41]],
                     source_sha256=digest(source), purpose='teacher label alignment and available-history utilization',
                     limitations='Existing teacher histories; off-trajectory causal input perturbations, not independent behavioral success.')
    write(ROOT / 'teacher-temporal-design.json', extension)
    signal = json.loads((ROOT / 'signals.json').read_text())
    with np.load('results/navigation-feature-teacher-raw-2026-10-25.npz') as z:
        x = torch.from_numpy(z['x'][signal['teacher_indices']].copy())
        target = z['y'][signal['teacher_indices']].copy() * .3
    expected = teacher_command(x.numpy())
    control = dict(reconstruction_mse_normalized=float(np.mean(((expected-target)/.3)**2)),
                   reconstruction_max_abs_rad_s=float(np.max(np.abs(expected-target))),
                   zero_action_mse_normalized=float(np.mean((target/.3)**2)))
    if control['reconstruction_mse_normalized'] > 1e-6:
        raise RuntimeError('teacher labels not reproduced from observed features: '+str(control))
    variants = []
    for start, stop in extension['bins']:
        delta = torch.zeros_like(x[:, :, 0]); delta[:, start:stop] = x[:, start:stop, 4] * extension['amplitude_K']
        changed = temperature_perturbation(x, delta)
        reference = teacher_command(changed.numpy())
        variants.append((start, stop, changed, float(np.sqrt(np.mean((reference-expected)**2)))))
    rows = []
    with torch.no_grad():
        for s in plan['specs']:
            for checkpoint in ['BC.pt', 'model-RL262144.pt']:
                model = load_model(s, checkpoint)
                baseline = .3 * torch.tanh(model.actor_mean(x))
                for start, stop, changed, reference_rms in variants:
                    pred = .3 * torch.tanh(model.actor_mean(changed))
                    rows.append(dict(spec=s, checkpoint=checkpoint, sample_bin=[start, stop],
                                     teacher_rms_delta_rad_s=reference_rms,
                                     actor_rms_delta_rad_s=float((pred-baseline).square().mean().sqrt())))
    write(out, dict(design=extension, positive_control=control, rows=rows, complete=True))
    check(plan)
    print('Teacher reconstruction and temporal probes:', control, len(rows), flush=True)


if __name__ == '__main__':
    main()
