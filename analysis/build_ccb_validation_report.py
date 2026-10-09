"""从已保存检验数据导出报告证据和静态图，不重算训练结果。"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '程序'))
from core.plot_fonts import setup_chinese_font
setup_chinese_font()
SOURCE = ROOT / 'results/ccb-continuous-2026-10-08'
OUT = ROOT / 'docs/reports/assets/ccb-continuous-2026-10-08'
OUT.mkdir(parents=True, exist_ok=True)


def compact(name, start_mode):
    path = SOURCE / name
    data = json.loads(path.read_text())
    config = data['configuration'].copy()
    config.pop('output', None)
    config.setdefault('start_mode', start_mode)
    config.setdefault('sampling', 'bilinear')
    runs = []
    for source in data['runs']:
        run = {key: source[key] for key in ['seed', 'reward', 'rounds', 'replay_transitions']}
        for label in ['initial_evaluation', 'final_evaluation']:
            run[label] = []
            for row in source[label]:
                item = {key: value for key, value in row.items() if key != 'trajectory'}
                item['head_trajectory'] = [step['position'] for step in row['trajectory']]
                run[label].append(item)
        runs.append(run)
    return {'source_file': name, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'configuration': config, 'geometry': data['geometry'], 'runs': runs}


studies = {key: compact(name, mode) for key, name, mode in [
    ('fixed_starts', 'default-noise.json', 'fixed'),
    ('mixed_starts_250', 'mixed-starts.json', 'mixed'),
    ('grid_ablation_250', 'grid-ablation.json', 'mixed'),
    ('ui_defaults_800', 'ui-defaults.json', 'mixed'),
]}
(OUT / 'validation-summary.json').write_text(json.dumps(studies, ensure_ascii=False, indent=2))
main = studies['ui_defaults_800']
fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), constrained_layout=True)
colors = ['#236b8e', '#d08a35', '#689d45']
for color, run in zip(colors, main['runs']):
    success = np.array([row['success'] for row in run['rounds']], dtype=float)
    rolling = np.convolve(success, np.ones(25)/25, mode='valid')
    axes[0].plot(np.arange(25, len(success)+1), rolling, label=f"种子 {run['seed']}", color=color)
axes[0].set(xlabel='训练轮次', ylabel='最近25轮到达比例', ylim=(-.03, 1.05),
            title='单热源混合起点训练（含0.1位置噪声）')
axes[0].grid(alpha=.2); axes[0].legend()
y, x = np.mgrid[:40, :40]
field = 20 + 100*np.exp(-np.hypot(x-20, y-20)/15)
axes[1].imshow(field, origin='lower', cmap='coolwarm', alpha=.65)
for row in main['runs'][0]['final_evaluation']:
    points = np.array(row['head_trajectory'])
    axes[1].plot(points[:,0], points[:,1], linewidth=1.4)
    axes[1].scatter(*row['actual_start'], marker='s', s=18, color='#333333')
axes[1].add_patch(plt.Circle((20, 20), .75, fill=False, color='black', linewidth=1.5))
axes[1].set(xlabel='X坐标', ylabel='Y坐标', title='种子7：关闭探索和位置噪声的8条评估轨迹')
fig.savefig(OUT / 'single-source-navigation.png', dpi=160)
plt.close(fig)
manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in OUT.iterdir() if path.name != 'manifest.json'}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(OUT)
