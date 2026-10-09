#!/usr/bin/env python3
"""Recorded, deterministic mechanical gates and prescribed-drive scan."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from thermotaxis.phase2_body import BodyConfig, DriveState, shape_geometry, solve_motion, simulate


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def metrics(c):
    rows = simulate(c)
    return {'displacement_x_m': float(rows[-1, 1]), 'displacement_y_m': float(rows[-1, 2]),
            'net_displacement_m': float(np.linalg.norm(rows[-1, 1:3])),
            'orientation_change_rad': float(rows[-1, 3] - rows[0, 3]), 'dissipation_J': float(rows[-1, 4])}, rows


def gates(c):
    mod = replace(c, amplitude_modulation_rad=.15, phase_modulation_rad=.2, bias_modulation_rad=.25)
    t, eps = .37, 1e-5
    g = shape_geometry(mod, t)
    before, after = shape_geometry(mod, t - eps), shape_geometry(mod, t + eps)
    derivative = (after.position_m - before.position_m) / (2 * eps)
    fd_error = float(np.max(np.abs(derivative - g.deformation_velocity_m_s)))
    centre_error = float(np.max(np.abs(g.weights_m @ g.position_m / c.length_m)))
    centre_rate_error = float(np.max(np.abs(g.weights_m @ g.deformation_velocity_m_s / c.length_m)))
    m = solve_motion(mod, t)
    rotated = solve_motion(mod, t, .73)
    R = np.array([[np.cos(.73), -np.sin(.73)], [np.sin(.73), np.cos(.73)]])
    rotation_error = float(np.max(np.abs(rotated.translation_m_s - R @ m.translation_m_s)))
    d = DriveState(.6, .1, .4, 5., .2, .3)
    mirrored = replace(d, amplitude_rad=-d.amplitude_rad, amplitude_rate_rad_s=-d.amplitude_rate_rad_s,
                       bias_rad=-d.bias_rad, bias_rate_rad_s=-d.bias_rate_rad_s)
    left, right = solve_motion(c, t, drive=d), solve_motion(c, t, drive=mirrored)
    reflection_error = float(max(abs(left.translation_m_s[0] - right.translation_m_s[0]),
                        abs(left.translation_m_s[1] + right.translation_m_s[1]),
                        c.length_m * abs(left.angular_velocity_rad_s + right.angular_velocity_rad_s)))
    doubled = solve_motion(replace(mod, drag_parallel_N_s_m2=2*mod.drag_parallel_N_s_m2), t)
    drag_scale_velocity_error = float(np.max(np.abs(doubled.translation_m_s - m.translation_m_s)))
    base_cycle, _ = metrics(mod)
    double_cycle, _ = metrics(replace(mod, drag_parallel_N_s_m2=2*mod.drag_parallel_N_s_m2))
    energy_scale = double_cycle['dissipation_J']/base_cycle['dissipation_J']
    iso, _ = metrics(replace(mod, drag_ratio=1.))
    # One reciprocal degree of freedom: a complete smooth amplitude cycle at fixed phase.
    dt = .005
    pose, energy = np.zeros(3), 0.
    def reciprocal(time):
        return DriveState(.45 + .15 * np.sin(2*np.pi*time), .15*2*np.pi*np.cos(2*np.pi*time), .3, 0., .1, 0.)
    for i in range(round(1 / dt)):
        t0 = i * dt
        a = solve_motion(c, t0, pose[2], reciprocal(t0))
        b = solve_motion(c, t0 + dt/2, pose[2] + dt*a.angular_velocity_rad_s/2, reciprocal(t0 + dt/2))
        pose[:2] += dt*b.translation_m_s
        pose[2] += dt*b.angular_velocity_rad_s
        energy += dt*b.power_W
    results = {'complete_geometry_derivative_max_error_m_s': fd_error,
               'material_centre_error_m': centre_error, 'material_centre_rate_error_m_s': centre_rate_error,
               'rotation_translation_error_m_s': rotation_error,
               'rotation_omega_error_rad_s': abs(m.angular_velocity_rad_s - rotated.angular_velocity_rad_s),
               'rotation_power_error_W': abs(m.power_W - rotated.power_W),
               'reflection_scaled_motion_error_m_s': reflection_error,
               'force_residual_N': float(np.linalg.norm(m.force_N)), 'torque_residual_N_m': abs(m.torque_N_m),
               'power_W': m.power_W, 'drag_scale_velocity_error_m_s': drag_scale_velocity_error,
               'drag_scale_power_ratio': doubled.power_W / m.power_W, 'drag_scale_energy_ratio': energy_scale, 'isotropic_net_translation_m': iso['net_displacement_m'],
               'reciprocal_cycle_displacement_m': float(np.linalg.norm(pose[:2])),
               'reciprocal_cycle_rotation_rad': float(pose[2]), 'reciprocal_cycle_dissipation_J': energy}
    checks = {'drag_scale_invariance': drag_scale_velocity_error < 1e-12 and abs(doubled.power_W/m.power_W - 2) < 1e-12 and abs(energy_scale - 2) < 1e-12, 'geometry_derivative': fd_error < 1e-10, 'centres': centre_error < 1e-15 and centre_rate_error < 1e-14,
              'rotation_covariance': rotation_error < 1e-12 and results['rotation_omega_error_rad_s'] < 1e-10,
              'reflection_symmetry': reflection_error < 1e-12, 'force_free': results['force_residual_N'] < 1e-20,
              'torque_free': results['torque_residual_N_m'] < 1e-23, 'positive_dissipation': m.power_W > 0,
              'isotropic_no_translation': iso['net_displacement_m'] < 1e-12,
              'reciprocal_no_net_pose': results['reciprocal_cycle_displacement_m'] < 1e-9 and abs(pose[2]) < 1e-6}
    return {'metrics': results, 'checks': {k: bool(v) for k,v in checks.items()}, 'passed': bool(all(checks.values()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results/phase2-body-2026-10-04')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / 'configs/phase2_body_reference.json'
    c = BodyConfig.from_dict(json.loads(config_path.read_text()))
    design = {'result_type': 'prescribed_body_mechanical_baseline', 'temperature_controller': False,
        'elasticity': False, 'biological_calibration': False, 'reference_config': asdict(c),
        'mechanical_gates': ['derivative','centre','rotation','reflection','force','torque','positive_power','isotropic_translation','reciprocal_pose','overall_drag_scaling_power_and_energy'],
        'scan_amplitude_rad': [.2, .4, .6, .8], 'scan_drag_ratio': [1., 1.5, 2., 3.],
        'dt_convergence_s': [.04, .02, .01, .005], 'outer_orders': [12, 24, 48, 72],
        'inner_orders': [8, 16, 32, 48], 'modulated_drive': {'amplitude_modulation_rad': .15,
           'phase_modulation_rad': .2, 'bias_modulation_rad': .25}}
    dump(output / 'design.json', design)
    sources = ['src/thermotaxis/phase2_body.py', 'analysis/phase2_body_baseline.py', 'configs/phase2_body_reference.json']
    dump(output / 'manifest.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
       'platform': platform.platform(), 'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'git_dirty': bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
       'source_sha256': {x: hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in sources},
       'command': ' '.join(sys.argv), 'randomness': 'none; deterministic prescribed drive'})
    gate = gates(c)
    dump(output / 'mechanical-gates.json', gate)
    mod = replace(c, **design['modulated_drive'])
    scan = []
    for ratio in design['scan_drag_ratio']:
        for amplitude in design['scan_amplitude_rad']:
            metric, _ = metrics(replace(c, drag_ratio=ratio, amplitude_rad=amplitude, dt_s=.02))
            scan.append({'drag_ratio': ratio, 'amplitude_rad': amplitude, **metric})
    dump(output / 'scan.json', scan)
    convergence = {'time': [], 'outer': [], 'inner': []}
    for dt in design['dt_convergence_s']:
        metric, _ = metrics(replace(mod, dt_s=dt, quadrature_order=72, integration_order=48))
        convergence['time'].append({'dt_s': dt, **metric})
    for n in design['outer_orders']:
        metric, _ = metrics(replace(mod, dt_s=.01, quadrature_order=n, integration_order=48))
        convergence['outer'].append({'quadrature_order': n, **metric})
    for n in design['inner_orders']:
        metric, _ = metrics(replace(mod, dt_s=.01, quadrature_order=72, integration_order=n))
        convergence['inner'].append({'integration_order': n, **metric})
    dump(output / 'convergence.json', convergence)
    trajectory_metrics = {}
    for name, config in [('reference', c), ('modulated', mod)]:
        metric, rows = metrics(config)
        trajectory_metrics[name] = metric
        np.savetxt(output / f'{name}-trajectory.csv', rows, delimiter=',', header='time_s,x_m,y_m,orientation_rad,dissipation_J,power_W', comments='')
    dump(output / 'summary.json', {'mechanical_gates_passed': gate['passed'], 'trajectories': trajectory_metrics,
                                  'scan_conditions': len(scan), 'convergence_conditions': 12,
                                  'reference_mean_net_speed_m_s': trajectory_metrics['reference']['net_displacement_m']/c.duration_s,
                                  'reference_max_material_speed_m_s': float(max(np.max(np.linalg.norm(solve_motion(c,float(t)).material_velocity_m_s,axis=1)) for t in np.linspace(0,c.duration_s,401))),
                                  'Reynolds_screen': {'Re_sampled_max':1000*c.length_m*float(max(np.max(np.linalg.norm(solve_motion(c,float(t)).material_velocity_m_s,axis=1)) for t in np.linspace(0,c.duration_s,401)))/.1, 'rho_kg_m3':1000., 'mu_Pa_s':.1, 'method':'rho L max material speed / mu; diagnostic values, drag not calibrated from mu'},
                                  'scientific_scope': 'mechanics of prescribed shape only; no thermotaxis or elasticity'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ratio in design['scan_drag_ratio']:
        records = [x for x in scan if x['drag_ratio'] == ratio]
        ax[0].plot([x['amplitude_rad'] for x in records], [x['displacement_x_m'] * 1e3 for x in records], 'o-', label=f'ratio={ratio:g}')
        ax[1].plot([x['amplitude_rad'] for x in records], [x['dissipation_J'] * 1e12 for x in records], 'o-', label=f'ratio={ratio:g}')
    ax[0].set_ylabel('COM x displacement (mm / 4 s)')
    ax[1].set_ylabel('Medium dissipation (pJ / 4 s)')
    for a in ax:
        a.set_xlabel('Tangent-angle amplitude (rad)'); a.legend(); a.grid(alpha=.2)
    fig.savefig(output / 'amplitude-drag-scan.png', dpi=180); fig.savefig(output / 'amplitude-drag-scan.svg'); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for a, name, key in zip(ax, ['time', 'outer', 'inner'], ['dt_s', 'quadrature_order', 'integration_order']):
        records = convergence[name]
        finest = records[-1]
        for metric, scale, label in [('displacement_x_m', c.length_m, 'x / L'), ('displacement_y_m', c.length_m, 'y / L'), ('dissipation_J', finest['dissipation_J'], 'energy / E')]:
            errors = [max(abs(r[metric] - finest[metric]) / scale, 1e-16) for r in records[:-1]]
            a.loglog([r[key] for r in records[:-1]], errors, 'o-', label=label)
        a.set_xlabel(key); a.set_ylabel('Difference from finest'); a.legend(); a.grid(alpha=.2)
    fig.savefig(output / 'numerical-convergence.png', dpi=180); fig.savefig(output / 'numerical-convergence.svg'); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name in ['reference', 'modulated']:
        rows = np.loadtxt(output / f'{name}-trajectory.csv', delimiter=',', skiprows=1)
        ax[0].plot(rows[:, 1] * 1e3, rows[:, 2] * 1e3, label=name)
        ax[1].plot(rows[:, 0], rows[:, 4] * 1e12, label=name)
    ax[0].set_xlabel('COM x (mm)'); ax[0].set_ylabel('COM y (mm)'); ax[0].axis('equal')
    ax[1].set_xlabel('Time (s)'); ax[1].set_ylabel('Cumulative dissipation (pJ)')
    for a in ax: a.legend(); a.grid(alpha=.2)
    fig.savefig(output / 'trajectory-energy.png', dpi=180); fig.savefig(output / 'trajectory-energy.svg'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, name, config in zip(axes, ['reference', 'modulated'], [c, mod]):
        rows = np.loadtxt(output / f'{name}-trajectory.csv', delimiter=',', skiprows=1)
        for time in [0., .25, .5, .75, 1.]:
            row = rows[np.argmin(abs(rows[:, 0] - time))]
            geom = shape_geometry(config, float(row[0]), s_m=np.linspace(0, config.length_m, 151))
            angle = row[3]
            rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            position = geom.position_m @ rotation.T + row[1:3]
            line, = ax.plot(position[:, 0]*1e3, position[:, 1]*1e3, label=f'{time:g} s')
            ax.plot(position[-1,0]*1e3,position[-1,1]*1e3,'o',ms=3,color=line.get_color())
        ax.set_title(name); ax.set_xlabel('World x (mm)'); ax.set_ylabel('World y (mm)')
        ax.axis('equal'); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output / 'body-shapes.png', dpi=180); fig.savefig(output / 'body-shapes.svg'); plt.close(fig)
    print(json.dumps({'output': str(output), 'gates_passed': gate['passed'], 'summary': trajectory_metrics}))
    if not gate['passed']: raise SystemExit('mechanical gate failed')

if __name__ == '__main__': main()
