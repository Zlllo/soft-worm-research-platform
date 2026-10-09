"""Presentation helpers for the original platform; simulation behavior stays in the engine."""
import html
import json
import re
from pathlib import Path

import streamlit as st


def clean_label(value):
    """Keep internal selection values intact while presenting short, quiet labels."""
    text = re.sub(r'^[^\w\u4e00-\u9fff]+', '', str(value)).strip()
    return re.sub(r'\s*\([a-z_]+\)\s*$', '', text)


def render_header():
    st.markdown('<nav class="lab-nav" aria-label="平台标识"><span class="lab-wordmark">'
                '<span class="brand-mark" aria-hidden="true">∿</span>C. elegans <b>Lab</b></span>'
                '<span class="nav-caption">趋温行为实验平台</span></nav>', unsafe_allow_html=True)


def render_welcome(model, model_label, algorithm, field, mode):
    # Illustration only: it does not claim to show a temperature field or a learned trajectory.
    paths = {
        'worm2d': 'M 95 170 L 135 156 L 174 160 L 207 144 L 244 151 L 277 135',
        'continuous_centerline': 'M 83 173 C 126 173 131 119 174 132 S 224 176 272 143 S 312 114 333 120',
        'active_deformation': 'M 83 153 C 110 98 138 206 168 153 S 220 101 249 153 S 301 201 333 140',
    }
    path = paths.get(model, paths['worm2d'])
    head = (277, 135) if model == 'worm2d' else ((333, 140) if model == 'active_deformation' else (333, 120))
    illustration = f'<svg viewBox="0 0 416 270" role="img" aria-label="{html.escape(model_label)}身体示意">'
    illustration += '<ellipse cx="207" cy="223" rx="116" ry="7" fill="#1d1d1f" opacity=".045"/>'
    illustration += f'<path d="{path}" fill="none" stroke="#d4e4f4" stroke-width="22" stroke-linecap="round" stroke-linejoin="round"/>'
    illustration += f'<path d="{path}" fill="none" stroke="#395b7b" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
    illustration += f'<circle cx="{head[0]}" cy="{head[1]}" r="8" fill="#395b7b"/><circle cx="{head[0]+2}" cy="{head[1]-2}" r="2" fill="white"/></svg>'
    st.markdown(f'<section class="welcome-hero"><div class="hero-copy"><span class="eyebrow">THERMOTAXIS LAB</span>'
                '<h1>观察运动。<br>理解趋温。</h1><p>从一次简单的实验开始，<br>探索感知、决策与身体运动之间的联系。</p></div>'
                f'<div class="body-preview">{illustration}<span>身体示意 · 非训练轨迹</span></div></section>', unsafe_allow_html=True)
    render_experiment_preview(model_label, algorithm, field, mode)


def render_experiment_preview(model_label, algorithm, field, mode):
    labels = [('身体模型', model_label), ('学习算法', algorithm), ('温度环境', field)]
    items = ''.join(f'<div><span>{label}</span><strong>{html.escape(value)}</strong></div>' for label, value in labels)
    st.markdown(f'<section class="experiment-preview"><div class="preview-heading"><h2>当前实验</h2>'
                f'<span>{html.escape(clean_label(mode))}</span></div><div class="preview-values">{items}</div></section>', unsafe_allow_html=True)


def render_saved_results(result):
    """Read the completed bundle on reruns instead of re-running a training engine."""
    root = Path(result['output_dir'])
    st.divider()
    st.markdown('### 最近一次实验')
    st.caption(result['name'])
    st.markdown('<div class="result-status"><span></span>训练完成 · 结果已保存</div>', unsafe_allow_html=True)
    graph, animation, logs = st.tabs(['曲线与轨迹', '运动回放', '运行日志'])
    with graph:
        evaluation_path = root / 'policy_evaluation.json'
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
            st.info(f"冻结策略评估：{evaluation['success_count']}/{evaluation['episode_count']} 个起点到达目标。")
            st.caption(f"关闭探索与位置噪声，固定温度场；目标半径 {evaluation['goal_radius']} 格。观测包含已知目标距离。")
            st.dataframe([{'起点': str(row['actual_start']), '到达目标': row['success'],
                           '实际步数': row['steps'], '最终距离': round(row['final_distance'], 3)}
                          for row in evaluation['episodes']], use_container_width=True)
        image = next(iter(sorted(root.glob('*training_results*.png'))), None)
        if image:
            st.image(str(image), use_container_width=True)
        else:
            st.caption('此实验没有保存训练图表。')
    with animation:
        movie = next(iter(sorted(root.glob('*.mp4'))), None)
        gif = next(iter(sorted(root.glob('*.gif'))), None)
        if movie:
            st.video(str(movie))
        elif gif:
            st.image(str(gif), use_container_width=True)
        else:
            st.caption('此实验没有保存运动回放。')
    with logs:
        log = root / 'experiment.log'
        if log.exists():
            st.code(log.read_text(encoding='utf-8', errors='replace')[-12000:], language='text')
        st.caption(f'结果目录：{root}')
