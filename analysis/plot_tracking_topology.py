"""Summarize crossed topology experiments without treating graphs as seeds."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = {'real8': '#9868a6', 'rewired': '#73818b', 'dense8': '#cb8a44',
          'gru4': '#418d7f', 'gru16': '#377fa8'}


def group(name):
    return 'rewired' if name.startswith('rewired') else name


def stats(values):
    values = [float(x) for x in values]
    return dict(mean=float(np.mean(values)), minimum=min(values), maximum=max(values), n=len(values))


def summarize(report):
    result = {}
    for budget in report['design']['episodes_per_field']:
        result[str(budget)] = {}
        for label in ('real8', 'rewired', 'dense8', 'gru4', 'gru16'):
            rows = [r for r in report['runs'] if r['episodes_per_field'] == budget and group(r['model']) == label]
            if not rows:
                continue
            result[str(budget)][label] = dict(
                tracking=stats(r['mean_tracking'] for r in rows),
                per_graph={name: stats(r['mean_tracking'] for r in rows if r['model'] == name)
                           for name in sorted({r['model'] for r in rows})},
                validation_mse=stats(r['curves'][-1]['validation']['all']['normalized_mse'] for r in rows),
                fraction_runs_ge_80_percent=float(np.mean([r['mean_tracking'] >= .8 for r in rows])),
                parameters=sorted({r['metadata']['parameters'] for r in rows}))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restricted', type=Path, default=Path('results/tracking-topology-budgets-2026-10-24'))
    parser.add_argument('--full', type=Path, default=Path('results/tracking-topology-fullio-budgets-2026-10-24'))
    parser.add_argument('--diagnostics', type=Path, default=Path('results/tracking-topology-diagnostics-2026-10-24'))
    parser.add_argument('--optimization', type=Path, default=Path('results/tracking-topology-optimization-2026-10-24'))
    parser.add_argument('--output', type=Path, default=Path('docs/reports/assets/tracking-topology-2026-10-24'))
    args = parser.parse_args()
    restricted = json.loads((args.restricted / 'results.json').read_text())
    full = json.loads((args.full / 'results.json').read_text())
    diagnostic = json.loads((args.diagnostics / 'results.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    report_summary = dict(restricted=summarize(restricted), full=summarize(full), independent={},
                          note='Ranges are observed min/max; three random graphs are not nine independent topologies.')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    labels = {'real8': 'Real chemical graph', 'rewired': 'Degree-preserving rewires',
              'dense8': 'Dense rate8', 'gru4': 'GRU4', 'gru16': 'GRU16'}
    for ax, report, mode, title in ((axes[0, 0], restricted, 'restricted', 'Restricted I/O (real/rewired: 41 params)'),
                                   (axes[0, 1], full, 'full', 'Full learned I/O (real/rewired: 71 params)')):
        values = report_summary[mode]
        for label in ('real8', 'rewired', 'dense8', 'gru4', 'gru16'):
            source = report_summary['restricted'] if mode == 'full' and label.startswith('gru') else values
            if label not in source['4']:
                continue
            x = [8, 32, 128]
            y = np.array([source[str(n)][label]['tracking']['mean'] for n in (4, 16, 64)]) * 100
            lo = np.array([source[str(n)][label]['tracking']['minimum'] for n in (4, 16, 64)]) * 100
            hi = np.array([source[str(n)][label]['tracking']['maximum'] for n in (4, 16, 64)]) * 100
            ax.plot(x, y, '-o', color=COLORS[label], label=labels[label], lw=1.8, markersize=4)
            ax.fill_between(x, lo, hi, color=COLORS[label], alpha=.08)
        ax.axhline(80, ls=':', color='#aaaaaa', lw=1)
        ax.set(xscale='log', ylim=(0, 105), title=title, xlabel='Total teacher episodes (both fields)', ylabel='Tracking time (%)')
        ax.set_xticks([8, 32, 128], ['8', '32', '128'])
        ax.grid(alpha=.15)
        ax.legend(fontsize=7, frameon=False, loc='upper left')
    groups = [(args.restricted, restricted, 'real8', 'Real\n41'),
              (args.restricted, restricted, 'rewired', 'Rewire\n41'),
              (args.restricted, restricted, 'dense8', 'Dense\n87'),
              (args.restricted, restricted, 'gru4', 'GRU4\n137'),
              (args.restricted, restricted, 'gru16', 'GRU16\n1121'),
              (args.full, full, 'real8', 'Real\n71'),
              (args.full, full, 'rewired', 'Rewire\n71')]
    train, val, held, lows, highs = [], [], [], [], []
    for root, report, label, name in groups:
        rows = [r for r in report['runs'] if r['episodes_per_field'] == 64 and group(r['model']) == label]
        checks = [r for r in diagnostic['runs'] if r['root'] == str(root) and r['episodes_per_field'] == 64 and group(r['model']) == label]
        train.append(np.mean([r['training']['all']['normalized_mse'] for r in checks]))
        val.append(np.mean([r['curves'][-1]['validation']['all']['normalized_mse'] for r in rows]))
        values = [r['independent_mean_tracking'] * 100 for r in checks]
        held.append(np.mean(values)); lows.append(held[-1] - min(values)); highs.append(max(values) - held[-1])
        report_summary['independent'][name.replace('\n', '_')] = dict(tracking=stats(v / 100 for v in values),
            per_graph={n:stats(r['independent_mean_tracking'] for r in checks if r['model']==n) for n in sorted({r['model'] for r in checks})},
            training_mse=float(train[-1]), validation_mse=float(val[-1]))
    ax = axes[1, 0]
    positions = np.arange(len(groups))
    ax.bar(positions - .18, train, .35, color='#74a9bb', label='Fit data')
    ax.bar(positions + .18, val, .35, color='#596a91', label='Validation episodes')
    zero = restricted['runs'][0]['curves'][-1]['validation']['all']['zero_command_mse']
    ax.axhline(zero, ls='--', color='#888888', label='Zero-command validation baseline')
    ax.set_xticks(positions, [g[3] for g in groups], fontsize=8)
    ax.set(title='128 episodes: action error and parameter count', ylabel='Normalized action MSE')
    ax.set_ylim(0, max(max(train), max(val), zero) * 1.35)
    ax.legend(fontsize=8, frameon=False); ax.grid(axis='y', alpha=.15)
    ax = axes[1, 1]
    ax.bar(positions, held, color=[COLORS[g[2]] for g in groups], yerr=[lows, highs], capsize=3)
    ax.set_xticks(positions, [g[3] for g in groups], fontsize=8)
    expert = np.mean([r['metrics']['tracking_fraction'] for r in diagnostic['independent_baselines']['expert']]) * 100
    blind = np.mean([r['metrics']['tracking_fraction'] for r in diagnostic['independent_baselines']['blind']]) * 100
    ax.axhline(expert, ls='--', color='#6d7f79', lw=1, label='Expert')
    ax.axhline(blind, ls=':', color='#888888', lw=1, label='Heading hold')
    ax.set(ylim=(0, 106), title='Frozen full-budget models: 8 new cases', ylabel='Tracking time (%)')
    ax.grid(axis='y', alpha=.15); ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Chemical topology and interface: exploratory imitation learning, 512 updates per run', fontsize=13)
    fig.supxlabel('Bands / whiskers: observed run ranges; rewires = 3 graphs x 3 training seeds, not confidence intervals.', fontsize=9)
    fig.savefig(args.output / 'tracking-topology.png', dpi=180)
    plt.close(fig)
    optimization = json.loads((args.optimization / 'results.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    opt_summary = {}
    for i, name in enumerate(('real8', 'rewired8-11', 'gru16')):
        rows = [r for r in optimization['runs'] if r['model'] == name]
        parent = restricted if name == 'gru16' else full
        old = [r for r in parent['runs'] if r['model'] == name and r['episodes_per_field'] == 16]
        # The first 512 updates must repeat the original experiment exactly.
        for row in rows:
            previous = next(r for r in old if r['seed'] == row['seed'])
            point = next(c for c in row['curves'] if c['update'] == 512)
            assert point['validation'] == previous['curves'][-1]['validation']
        y = np.array([[c['validation']['all']['normalized_mse'] for c in r['curves']] for r in rows])
        x = [c['update'] for c in rows[0]['curves']]
        color = COLORS[group(name)]
        label = 'Rewire graph 11' if name.startswith('rewired') else labels[name]
        axes[0].plot(x, y.mean(0), color=color, label=label)
        axes[0].fill_between(x, y.min(0), y.max(0), color=color, alpha=.12)
        before = [r['mean_tracking'] * 100 for r in old]
        after = [r['mean_tracking'] * 100 for r in rows]
        axes[1].bar(i - .18, np.mean(before), .34, color='#91a7b9', label='512 updates' if i == 0 else None)
        axes[1].bar(i + .18, np.mean(after), .34, color='#318f85', label='2048 updates' if i == 0 else None)
        for row in rows:
            earlier = next(r for r in old if r['seed'] == row['seed'])
            axes[1].plot([i - .18, i + .18], [earlier['mean_tracking']*100, row['mean_tracking']*100],
                         color='#53616c', alpha=.5, lw=.8, marker='.', markersize=3)
        opt_summary[name] = dict(after_512=stats(v / 100 for v in before), after_2048=stats(v / 100 for v in after),
                                 prefix_512_validation_identical=True)
    axes[0].axhline(zero, ls='--', color='#888888', lw=1)
    axes[0].axvline(512, ls=':', color='#888888', lw=1)
    axes[0].set(xlabel='Optimizer updates; no new teacher trajectories', ylabel='Validation normalized action MSE', title='Same 32 episodes; shading = 3-seed range')
    axes[0].legend(fontsize=8, frameon=False); axes[0].grid(alpha=.15)
    axes[1].set_xticks(range(3), ['Real / full I/O', 'Rewire11 / full I/O', 'GRU16'])
    axes[1].set(ylim=(0, 106), ylabel='Tracking time (%)', title='Same 4 development cases; lines pair training seeds')
    axes[1].legend(fontsize=8, frameon=False); axes[1].grid(axis='y', alpha=.15)
    fig.suptitle('Post-hoc optimization control: four times the updates, identical teacher data', fontsize=12)
    fig.savefig(args.output / 'optimization-control.png', dpi=180); plt.close(fig)
    report_summary['optimization_control'] = opt_summary
    (args.output / 'summary.json').write_text(json.dumps(report_summary, indent=2) + '\n')
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
