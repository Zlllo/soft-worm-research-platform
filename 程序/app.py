"""
Streamlit Web 应用 - 秀丽隐杆线虫仿真系统
支持标准训练、迁移学习、课程学习三种模式
与 simulation_engine.py 和 core 文件夹完全兼容
采用极简浅色主题，支持身体参数和噪声调节
修复核心Bug：
1. UI参数正确传递到Worm2D类
2. 移除导致卡死的st.rerun()调用 ✅
3. 优化UI流畅度和稳定性 ✅
"""

import streamlit as st
import datetime
import os
import time
import sys
import io
import shutil
import traceback
from pathlib import Path
from collections import deque

# 将core文件夹添加到Python路径中
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

# 🔧 Windows GBK编码修复：强制stdout使用UTF-8，避免emoji打印崩溃
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    elif hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

# 从 simulation_engine.py 导入所有引擎
from simulation_engine import (
    run_standard_simulation_engine,
    run_transfer_simulation_engine,
    run_curriculum_simulation_engine
)

# 从 core 文件夹导入必要组件
from core.environment import ExperimentConfig
from core.neural_networks import PYTORCH_AVAILABLE

# --- 页面基础配置 ---
st.set_page_config(
    page_title="C. elegans 仿真系统",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 极简界面主题 ---
from core.ui import clean_label, render_header, render_welcome, render_experiment_preview, render_saved_results
st.markdown("<style>" + (Path(__file__).parent / "assets/ui.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)

# --- 初始化 Session State ---
if 'is_simulating' not in st.session_state:
    st.session_state.is_simulating = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'current_config' not in st.session_state:
    st.session_state.current_config = None
if 'selected_mode_index' not in st.session_state:
    st.session_state.selected_mode_index = 0


def request_experiment():
    st.session_state.run_requested = True
    st.session_state.is_simulating = True
    st.session_state.stop_requested = False


def request_stop():
    st.session_state.stop_requested = True
    st.session_state.stopped_feedback = True
    st.session_state.is_simulating = False
    st.session_state.run_requested = False


# --- 主标题区域 ---
render_header()

# 侧边栏保留全部实验参数；显示文字与内部选项值分别处理。
with st.sidebar:
    st.markdown('<div class="sidebar-brand">实验配置<span>CONFIGURATION</span></div>', unsafe_allow_html=True)
    
    # 实验名称
    if "experiment_name" not in st.session_state:
        st.session_state.experiment_name = f"exp_{datetime.datetime.now().strftime('%m%d_%H%M%S')}"
    experiment_name = st.text_input(
        "实验名称",
        key="experiment_name",
        disabled=st.session_state.is_simulating,
        help="实验标识符"
    )
    
    # 实验模式选择
    st.markdown("### 训练方式")
    mode_options = ["🎓 标准训练", "🔄 迁移学习", "📚 课程学习"]
    
    if not st.session_state.is_simulating:
        selected_mode = st.radio(
            "选择模式",
            mode_options,
            index=st.session_state.selected_mode_index,
            label_visibility="collapsed",
            key="mode_selector",
            horizontal=True,
            format_func=clean_label,
        )
        st.session_state.selected_mode_index = mode_options.index(selected_mode)
        
        if "标准训练" in selected_mode:
            mode_choice = "标准训练模式"
        elif "迁移学习" in selected_mode:
            mode_choice = "迁移学习实验"
        elif "课程学习" in selected_mode:
            mode_choice = "课程学习实验"
        else:
            mode_choice = "标准训练模式"
            
        st.caption({"标准训练模式": "在一个环境中学习运动策略。", "迁移学习实验": "比较策略在新环境中的表现。", "课程学习实验": "从简单环境逐步学习复杂任务。"}[mode_choice])
    else:
        current_mode = mode_options[st.session_state.selected_mode_index]
        if "标准训练" in current_mode:
            mode_choice = "标准训练模式"
        elif "迁移学习" in current_mode:
            mode_choice = "迁移学习实验"
        elif "课程学习" in current_mode:
            mode_choice = "课程学习实验"
        else:
            mode_choice = "标准训练模式"
        st.caption(f"{mode_choice} · 运行中")

    st.markdown("---")

    # ==========================================================================
    # 身体参数配置
    # ==========================================================================
    with st.expander("身体模型与形态", expanded=False):
        # 身体模型类型选择器
        body_model_map = {
            "🪱 多节段链条 (Worm2D)": "worm2d",
            "📏 连续中心线 (Continuous)": "continuous_centerline",
            "🌊 主动形变波 (Active Wave)": "active_deformation",
        }
        body_model_display = st.selectbox(
            "身体模型",
            list(body_model_map.keys()),
            index=0,
            disabled=st.session_state.is_simulating,
            format_func=clean_label,
            key="body_model_selector",
            help="选择线虫身体的数学模型"
        )
        body_model_type = body_model_map[body_model_display]

        # ---- Worm2D 参数 ----
        if body_model_type == "worm2d":
            st.markdown("##### 身体结构")
            col1, col2 = st.columns(2)
            with col1:
                num_segments = st.slider(
                    "节段数量", 3, 15, 6, 1,
                    disabled=st.session_state.is_simulating,
                    key="num_segments_slider",
                    help="线虫身体的节段数量"
                )
                segment_length = st.slider(
                    "节段长度", 3.0, 8.0, 5.0, 0.5,
                    disabled=st.session_state.is_simulating,
                    key="segment_length_slider",
                    help="每个节段的长度(像素)"
                )
            with col2:
                head_radius = st.slider(
                    "头部半径", 2.0, 6.0, 4.0, 0.5,
                    disabled=st.session_state.is_simulating,
                    key="head_radius_slider",
                    help="头部圆形半径(像素)"
                )
                body_width = st.slider(
                    "身体宽度", 1.0, 4.0, 2.5, 0.25,
                    disabled=st.session_state.is_simulating,
                    key="body_width_slider",
                    help="身体节段宽度(像素)"
                )
            st.markdown("##### 运动参数")
            col3, col4 = st.columns(2)
            with col3:
                max_turn_angle = st.slider(
                    "最大转向角", 10, 60, 30, 5,
                    disabled=st.session_state.is_simulating,
                    key="max_turn_angle_slider",
                    help="每步最大转向角度(度)"
                )
                forward_speed = st.slider(
                    "前进速度", 1.0, 5.0, 2.0, 0.5,
                    disabled=st.session_state.is_simulating,
                    key="forward_speed_slider",
                    help="前进步长(像素)"
                )
            with col4:
                backward_speed = st.slider(
                    "后退速度", 0.5, 3.0, 1.0, 0.25,
                    disabled=st.session_state.is_simulating,
                    key="backward_speed_slider",
                    help="后退步长(像素)"
                )
                turning_speed = st.slider(
                    "转向速度", 0.5, 3.0, 1.5, 0.25,
                    disabled=st.session_state.is_simulating,
                    key="turning_speed_slider",
                    help="转向时的移动速度"
                )
            # 为Q-Learning兼容设置默认值
            sample_count = num_segments
            body_length_ui = segment_length
            curvature_limit = max_turn_angle
            length_stiffness = 0.85
            curvature_stiffness = 0.35
            damping = 0.72
            wave_amplitude = 1.0
            wave_frequency = 0.25
            wave_speed = 1.0
            wave_length = 12.0
            # RFT 参数默认值 (仅 ADB 使用)
            sub_steps = 10
            drag_coeff = 0.35
            curriculum_freeze_steps = 0
            wave_envelope = True

        # ---- ContinuousCenterlineBody 参数 ----
        elif body_model_type == "continuous_centerline":
            st.markdown("##### 中心线")
            col1, col2 = st.columns(2)
            with col1:
                sample_count = st.slider(
                    "采样点数", 3, 20, 9, 1,
                    disabled=st.session_state.is_simulating,
                    key="cc_sample_count",
                    help="沿中心线的等距采样点数"
                )
                body_length_ui = st.slider(
                    "身体总弧长", 5.0, 30.0, 12.0, 0.5,
                    disabled=st.session_state.is_simulating,
                    key="cc_body_length",
                    help="中心线总弧长（场地坐标单位）"
                )
                head_radius = st.slider(
                    "头部半径", 2.0, 6.0, 3.0, 0.5,
                    disabled=st.session_state.is_simulating,
                    key="cc_head_radius",
                    help="头部圆形半径(像素)"
                )
            with col2:
                body_width = st.slider(
                    "身体宽度", 1.0, 4.0, 2.0, 0.25,
                    disabled=st.session_state.is_simulating,
                    key="cc_body_width",
                    help="身体节段宽度(像素)"
                )
                forward_speed = st.slider(
                    "前进速度", 0.5, 5.0, 1.2, 0.1,
                    disabled=st.session_state.is_simulating,
                    key="cc_forward_speed",
                    help="每步前进基础步长"
                )
                max_turn_angle = st.slider(
                    "最大转向角", 10, 60, 35, 5,
                    disabled=st.session_state.is_simulating,
                    key="cc_max_turn_angle",
                    help="每步最大转向角度(度)"
                )
            st.markdown("##### 身体约束")
            col3, col4 = st.columns(2)
            with col3:
                curvature_limit = st.slider(
                    "曲率限制角", 15, 90, 45, 5,
                    disabled=st.session_state.is_simulating,
                    key="cc_curvature_limit",
                    help="相邻段最大弯折角度(度)"
                )
                length_stiffness = 1.0
                st.caption("节段长度固定；相邻节段转角不超过上述限制。")
            with col4:
                curvature_stiffness = 1.0
                damping = st.slider(
                    "运动阻尼", 0.3, 0.95, 0.72, 0.05,
                    disabled=st.session_state.is_simulating,
                    key="cc_damping",
                    help="速度平滑阻尼系数"
                )
            # 为兼容性设置默认值
            num_segments = sample_count
            segment_length = body_length_ui
            backward_speed = 1.0
            turning_speed = 1.5
            wave_amplitude = 0.0
            wave_frequency = 0.0
            wave_speed = 0.0
            wave_length = body_length_ui
            # RFT 参数默认值 (仅 ADB 使用)
            sub_steps = 10
            drag_coeff = 0.35
            curriculum_freeze_steps = 0
            wave_envelope = True

        # ---- ActiveDeformationBody 参数 (RFT 力基波驱动) ----
        else:  # active_deformation
            st.info("🌊 ADB 波驱动模型 (RFT): 速度与转向由波参数经力/力矩平衡涌现, "
                    "没有头部步长/转向角参数; 动作 = 连续 (波幅, 频率, 曲率偏置)")
            st.markdown("##### 中心线")
            col1, col2 = st.columns(2)
            with col1:
                sample_count = st.slider(
                    "采样点数", 3, 20, 9, 1,
                    disabled=st.session_state.is_simulating,
                    key="ad_sample_count",
                    help="沿中心线的等距采样点数"
                )
                body_length_ui = st.slider(
                    "身体总弧长", 5.0, 30.0, 12.0, 0.5,
                    disabled=st.session_state.is_simulating,
                    key="ad_body_length",
                    help="中心线总弧长（场地坐标单位）"
                )
            with col2:
                head_radius = st.slider(
                    "头部半径", 2.0, 6.0, 3.0, 0.5,
                    disabled=st.session_state.is_simulating,
                    key="ad_head_radius",
                    help="头部圆形半径(像素)"
                )
                body_width = st.slider(
                    "身体宽度", 1.0, 4.0, 2.0, 0.25,
                    disabled=st.session_state.is_simulating,
                    key="ad_body_width",
                    help="身体节段宽度(像素)"
                )
            st.markdown("##### 驱动波参数")
            col5, col6 = st.columns(2)
            with col5:
                wave_amplitude = st.slider(
                    "波幅 (初始值)", 0.1, 2.0, 1.0, 0.1,
                    disabled=st.session_state.is_simulating,
                    key="ad_wave_amplitude",
                    help="正弦波侧向摆动幅度 (AC 动作边界 [0.05, 2.0])"
                )
                wave_frequency = st.slider(
                    "波频率 (初始值)", 0.02, 0.8, 0.25, 0.02,
                    disabled=st.session_state.is_simulating,
                    key="ad_wave_frequency",
                    help="肌肉波频率 (AC 动作边界 [0.02, 0.8])"
                )
            with col6:
                wave_speed = st.slider(
                    "波传播速度", 0.1, 3.0, 1.0, 0.1,
                    disabled=st.session_state.is_simulating,
                    key="ad_wave_speed",
                    help="波沿身体传播速度倍率"
                )
                wave_length = st.slider(
                    "波长", 3.0, 30.0, 12.0, 0.5,
                    disabled=st.session_state.is_simulating,
                    key="ad_wave_length",
                    help="正弦波的波长 (默认=体长, 身体上恰好一个完整波)"
                )
            st.markdown("##### RFT 力学参数")
            col7, col8 = st.columns(2)
            with col7:
                sub_steps = st.slider(
                    "子步积分步数", 2, 50, 10, 1,
                    disabled=st.session_state.is_simulating,
                    key="ad_sub_steps",
                    help="每决策步内的物理子步数: 越大波传播越平滑、积分越准"
                )
                drag_coeff = st.slider(
                    "阻力系数 (能耗尺度)", 0.1, 1.0, 0.35, 0.05,
                    disabled=st.session_state.is_simulating,
                    key="ad_drag_coeff",
                    help="阻力绝对量级: 只缩放能耗, 不影响速度 (速度由波参数决定, 与粘度无关)"
                )
                curriculum_freeze_steps = st.number_input(
                    "课程阶段一: 冻结转向的决策步数", 0, 100000, 0, 100,
                    disabled=st.session_state.is_simulating,
                    key="ad_curriculum_freeze",
                    help="前 N 步强制曲率偏置 b=0 (只学速度-能耗), 0=跳过阶段一"
                )
            with col8:
                wave_envelope = st.checkbox(
                    "头尾波幅包络", True,
                    disabled=st.session_state.is_simulating,
                    key="ad_wave_envelope",
                    help="波幅在头尾渐变为零 (真实线虫形态; 削弱端点伪影)"
                )
                st.caption("法向/切向阻力比固定 1.4 (C. elegans 实测值)")
                st.caption("曲率限制角用于状态/违例指标, 不作为力学约束")
            # 兼容性默认值 (RFT 物理不使用)
            num_segments = sample_count
            segment_length = body_length_ui
            forward_speed = 1.2
            max_turn_angle = 35.0
            backward_speed = 1.0
            turning_speed = 1.5
            curvature_limit = 45.0
            length_stiffness = 0.85
            curvature_stiffness = 0.35
            damping = 0.72

    # ==========================================================================
    # 噪声参数配置
    # ==========================================================================
    with st.expander("噪声与扰动", expanded=False):
        st.markdown("##### 运动噪声")
        
        col1, col2 = st.columns(2)
        with col1:
            position_noise = st.slider(
                "位置噪声", 0.0, 2.0, 0.1, 0.05,
                disabled=st.session_state.is_simulating,
                key="position_noise_slider",
                help="位置随机扰动强度"
            )
            angle_noise = st.slider(
                "角度噪声", 0.0, 15.0, 2.0, 0.5,
                disabled=st.session_state.is_simulating,
                key="angle_noise_slider",
                help="转向角度随机扰动(度)"
            )
            
        with col2:
            thermal_noise = st.slider(
                "温度感知噪声", 0.0, 5.0, 0.5, 0.1,
                disabled=st.session_state.is_simulating,
                key="thermal_noise_slider",
                help="温度感知的随机误差"
            )
            action_noise = st.slider(
                "动作执行噪声", 0.0, 0.3, 0.05, 0.01,
                disabled=st.session_state.is_simulating,
                key="action_noise_slider",
                help="动作执行的随机性"
            )
        
        noise_correlation = st.slider(
            "噪声时间相关性", 0.0, 0.9, 0.3, 0.1,
            disabled=st.session_state.is_simulating,
            key="noise_correlation_slider",
            help="相邻时刻噪声的相关程度"
        )

    st.markdown("---")

    # ==========================================================================
    # 标准训练模式配置
    # ==========================================================================
    if mode_choice == "标准训练模式":
        st.markdown("### 环境与策略")

        # 动作空间(方向)选择器 — 按身体模型过滤; 后续可扩展 16 方向
        if body_model_type == "active_deformation":
            # ADB (RFT 波驱动): 无离散方向动作, 动作 = 连续波参数 (波幅, 频率, 曲率偏置)
            direction_mode = "continuous"
            action_size = 4
            st.info("🌊 ADB: 动作空间固定为连续波参数 (波幅, 频率, 曲率偏置); "
                    "速度与转向由力/力矩平衡涌现, 无离散方向动作")
        else:
            if body_model_type == "worm2d":
                direction_options = {"4 方向 (离散)": "4", "8 方向 (离散)": "8"}
            else:
                direction_options = {"4 方向 (离散)": "4", "8 方向 (离散)": "8", "连续方向 (Actor-Critic)": "continuous"}
            direction_label = st.selectbox(
                "动作空间",
                list(direction_options.keys()),
                index=0,
                disabled=st.session_state.is_simulating,
                format_func=clean_label,
                key="direction_selector_standard",
                help="离散方向对应 Q-Learning / DQN / Dueling DQN；连续方向仅支持 Actor-Critic"
            )
            direction_mode = direction_options[direction_label]
            action_size = 4 if direction_mode == "continuous" else int(direction_mode)  # 连续方向不走离散Q路径

        # 学习方法选择 — 根据身体模型类型 + 动作空间显示可用算法
        if direction_mode == "continuous":
            if not PYTORCH_AVAILABLE:
                st.error("⚠️ 连续方向需要 PyTorch (Actor-Critic)")
                st.stop()
            methods = ["🎯 Actor-Critic"]
            default_method_idx = 0
        elif body_model_type == "worm2d":
            # Worm2D: Q-Learning + DQN + Dueling DQN
            methods = ["📋 Q-Learning"]
            if PYTORCH_AVAILABLE:
                methods += ["🧠 DQN", "⚡ Dueling DQN"]
            default_method_idx = 1 if PYTORCH_AVAILABLE else 0
        else:
            # 连续身体模型离散方向: Q-Learning + DQN + Dueling DQN
            methods = ["📋 Q-Learning"]
            if PYTORCH_AVAILABLE:
                methods += ["🧠 DQN", "⚡ Dueling DQN"]
            default_method_idx = 0

        method_choice = st.selectbox(
            "学习算法",
            methods,
            index=default_method_idx,
            disabled=st.session_state.is_simulating,
            format_func=clean_label,
            key=f"method_selector_standard_{body_model_type}_{direction_mode}"
        )

        # 温度场选择
        field_map = {
            "🌊 S型迷宫 (maze_thermal_channel)": "maze_thermal_channel",
            "🏰 复杂迷宫 (complex_maze_channel)": "complex_maze_channel", 
            "🎯 单热源 (single_center)": "single_center",
            "♻️ 双热源 (dual_center)": "dual_center", 
            "🔵 环形热源 (ring_hotspot)": "ring_hotspot", 
            "🔸 斑点热源 (spotty_field)": "spotty_field",
            "🌀 动态双中心 (dynamic_rotating_double_center)": "dynamic_rotating_double_center", 
            "⭐ 动态四中心 (dynamic_rotating_quad_center)": "dynamic_rotating_quad_center"
        }
        
        field_name = st.selectbox(
            "温度环境",
            list(field_map.keys()), 
            index=0, 
            disabled=st.session_state.is_simulating,
            format_func=clean_label,
            key="field_selector_standard"
        )
        field_type = field_map[field_name]
        
        enable_step_tracking = st.toggle(
            "详细分析",
            value=False,
            disabled=st.session_state.is_simulating,
            key="step_tracking_standard"
        )

    # ==========================================================================
    # 迁移学习实验配置
    # ==========================================================================
    elif mode_choice == "迁移学习实验":
        st.markdown("### 迁移设置")
        
        if not PYTORCH_AVAILABLE:
            st.error("⚠️ 需要 PyTorch")
            st.stop()
        
        method_choice = st.selectbox(
            "学习算法",
            ["🧠 DQN", "⚡ Dueling DQN"],
            index=1,
            disabled=st.session_state.is_simulating,
            format_func=clean_label,
            key="method_selector_transfer"
        )
        # 奖励函数变体（迁移学习仅 Worm2D，能量变体对其无效果，固定原版）
        reward_variant = "original"
        reward_energy_weight = 0.1
        action_size = 4  # 迁移学习仅 Worm2D，固定 4 方向
        
        transfer_options = [
            "🎯 单热源 (single_center)", "♻️ 双热源 (dual_center)", "🌊 S型迷宫 (maze_thermal_channel)", 
            "🏰 复杂迷宫 (complex_maze_channel)", "🌀 动态双中心 (dynamic_rotating_double_center)", 
            "⭐ 动态四中心 (dynamic_rotating_quad_center)"
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            source_field_display = st.selectbox(
                "源环境",
                transfer_options,
                index=0,
                disabled=st.session_state.is_simulating,
                format_func=clean_label,
                key="source_field_selector"
            )
            # 提取英文名称
            source_field = source_field_display.split('(')[1].split(')')[0]
        
        with col2:
            target_field_display = st.selectbox(
                "目标环境",
                transfer_options,
                index=2,
                disabled=st.session_state.is_simulating,
                format_func=clean_label,
                key="target_field_selector"
            )
            # 提取英文名称
            target_field = target_field_display.split('(')[1].split(')')[0]

    # ==========================================================================
    # 课程学习实验配置
    # ==========================================================================
    elif mode_choice == "课程学习实验":
        st.markdown("### 课程设置")
        
        if not PYTORCH_AVAILABLE:
            st.error("⚠️ 需要 PyTorch")
            st.stop()
        
        method_choice = "⚡ Dueling DQN"
        st.info("🎯 使用 Dueling DQN")
        
        curriculum_stages = [
            "🎯 初级单热源 (single_center)", "♻️ 双热源 (dual_center)", "🔸 斑点热源 (spotty_field)", "🔵 环形热源 (ring_hotspot)", 
            "🌊 S型迷宫 (maze_thermal_channel)", "🏰 复杂迷宫 (complex_maze_channel)", "🌀 动态双中心 (dynamic_rotating_double_center)", "⭐ 动态四中心 (dynamic_rotating_quad_center)"
        ]
        
        test_stage_name = st.selectbox(
            "测试阶段",
            curriculum_stages,
            index=len(curriculum_stages)-1,
            disabled=st.session_state.is_simulating,
            format_func=clean_label,
            key="test_stage_selector"
        )
        test_stage_idx = curriculum_stages.index(test_stage_name)
        
        env_size = st.slider(
            "环境尺寸",
            min_value=40, max_value=120, value=80, step=10,
            disabled=st.session_state.is_simulating,
            key="env_size_slider"
        )
        
        enable_step_tracking = st.toggle(
            "详细分析",
            value=False,
            disabled=st.session_state.is_simulating,
            key="step_tracking_curriculum"
        )

    # ==========================================================================
    # 高级参数配置 - 紧凑版，减少默认训练量以避免卡死
    # ==========================================================================
    if mode_choice in ["标准训练模式", "迁移学习实验"]:
        with st.expander("训练参数", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                width = st.number_input(
                    "宽度", min_value=40, max_value=200, value=40, step=10,  # 改为40，匹配8.22版本
                    disabled=st.session_state.is_simulating,
                    key="width_input"
                )
                # 匹配8.22版本的默认训练轮次
                num_rounds = st.slider(
                    "训练轮次", 5, 1000, 800, 5,  # 改为800，匹配8.22版本
                    disabled=st.session_state.is_simulating,
                    key="num_rounds_slider"
                )
                initial_epsilon = st.slider(
                    "初始探索率", 0.5, 1.0, 0.98, 0.02,
                    disabled=st.session_state.is_simulating,
                    key="initial_epsilon_slider"
                )
                min_epsilon = st.slider(
                    "最小探索率", 0.01, 0.5, 0.1, 0.01,
                    disabled=st.session_state.is_simulating,
                    key="min_epsilon_slider"
                )
                if "DQN" in method_choice or "Actor" in method_choice:
                    hidden_size = st.slider(
                        "隐藏层大小", 32, 512, 32, 32,  # 改为32，匹配8.22版本
                        disabled=st.session_state.is_simulating,
                        key="hidden_size_slider",
                        help="神经网络隐藏层神经元数量"
                    )
                else:
                    hidden_size = 32  # 改为32，匹配8.22版本
            
            with col2:
                height = st.number_input(
                    "高度", min_value=40, max_value=200, value=40, step=10,  # 改为40，匹配8.22版本
                    disabled=st.session_state.is_simulating,
                    key="height_input"
                )
                # 匹配8.22版本的每轮步数
                steps_per_round = st.slider(
                    "每轮步数", 50, 1000, 500, 25,  # 改为500，匹配8.22版本
                    disabled=st.session_state.is_simulating,
                    key="steps_per_round_slider"
                )
                learning_rate = st.slider(
                    "学习率", 0.1, 1.0, 0.4, 0.05,
                    disabled=st.session_state.is_simulating,
                    key="learning_rate_slider"
                )
                epsilon_decay = st.slider(
                    "探索率衰减", 0.001, 0.05, 0.008, 0.001, format="%.3f",
                    disabled=st.session_state.is_simulating,
                    key="epsilon_decay_slider"
                )
                discount_factor = st.slider(
                    "折扣因子", 0.8, 0.99, (0.99 if body_model_type == "continuous_centerline" and "Actor" in method_choice else 0.95), 0.01,
                    disabled=st.session_state.is_simulating,
                    key="discount_factor_slider"
                )
            
            if "DQN" in method_choice:
                neural_lr = st.slider(
                    "神经网络学习率", 0.0001, 0.01, 0.001, 0.0001, format="%.4f",
                    disabled=st.session_state.is_simulating,
                    key="neural_lr_slider"
                )
                weight_decay = st.slider(
                    "权重衰减", 0.0001, 0.01, 0.0005, 0.0001, format="%.4f",
                    disabled=st.session_state.is_simulating,
                    key="weight_decay_slider"
                )
                # state_v2 选项 — 仅 Worm2D + DQN 可用
                if body_model_type == "worm2d":
                    use_state_v2 = st.toggle(
                        "🧬 使用 state_v2 (10维)",
                        value=False,
                        disabled=st.session_state.is_simulating,
                        key="use_state_v2_toggle",
                        help="启用后 DQN 输入从 32 维 (8×4) 升级到 40 维 (10×4)：原8维 + 能量率 + 平均曲率"
                    )
                else:
                    use_state_v2 = False
            else:
                neural_lr = 0.001
                weight_decay = 0.0005
                use_state_v2 = False

            ac_random_starts = True
            ac_seed = 7
            # Actor-Critic 专用超参数
            if "Actor-Critic" in method_choice:
                st.markdown("##### Actor-Critic")
                col_ac1, col_ac2 = st.columns(2)
                with col_ac1:
                    ac_hidden_size = st.slider(
                        "AC 隐藏层大小", 32, 256, 128, 16,
                        disabled=st.session_state.is_simulating,
                        key="ac_hidden_size_slider",
                        help="Actor 和 Critic 网络的隐藏层神经元数"
                    )
                    ac_actor_lr = st.slider(
                        "Actor 学习率", 0.00001, 0.01, 0.0001, 0.00001,
                        format="%.5f",
                        disabled=st.session_state.is_simulating,
                        key="ac_actor_lr_slider",
                        help="策略网络学习率"
                    )
                with col_ac2:
                    ac_batch_size = st.slider(
                        "AC 批次大小", 16, 256, 64, 16,
                        disabled=st.session_state.is_simulating,
                        key="ac_batch_size_slider",
                        help="经验回放采样批次大小"
                    )
                    ac_critic_lr = st.slider(
                        "Critic 学习率", 0.0001, 0.01, 0.001, 0.0001,
                        format="%.4f",
                        disabled=st.session_state.is_simulating,
                        key="ac_critic_lr_slider",
                        help="价值网络学习率"
                    )
                if body_model_type == "continuous_centerline":
                    ac_random_starts = st.checkbox(
                        "单热源混合起点训练", value=True,
                        disabled=st.session_state.is_simulating, key="ac_random_starts",
                        help="每四轮保留一轮默认起点，其余使用随机位置和初始朝向，减少只记住一条路线的情况。")
                    ac_seed = st.number_input("训练随机种子", min_value=0, max_value=999999,
                                              value=7, step=1, key="ac_seed",
                                              disabled=st.session_state.is_simulating)
                    st.caption("连续转向与步长控制；测温使用连续插值，目标到达半径为 0.75 格。")
                ac_noise_scale = st.slider(
                    "探索噪声强度", 0.1, 2.0, 0.6, 0.1,
                    disabled=st.session_state.is_simulating,
                    key="ac_noise_scale_slider",
                    help="Ornstein-Uhlenbeck 噪声初始强度 (越大探索越多)"
                )
            else:
                ac_hidden_size = 128
                ac_actor_lr = 1e-4
                ac_critic_lr = 1e-3
                ac_batch_size = 64
                ac_noise_scale = 0.6

            # 奖励函数变体 — 统一 core/reward_functions.py 模块
            st.markdown("##### 奖励函数")
            reward_variant = st.selectbox(
                "奖励变体",
                (["continuous", "continuous_energy", "original", "energy"]
                 if body_model_type == "continuous_centerline" and "Actor" in method_choice
                 else ["original", "energy"]),
                index=0,
                format_func=lambda v: {"continuous": "连续趋近奖励", "continuous_energy": "连续趋近奖励 + 能耗",
                                       "original": "温度阶梯（原版对照）", "energy": "温度阶梯 + 能耗"}[v],
                disabled=st.session_state.is_simulating,
                key="reward_variant_selectbox",
                help="连续奖励使用插值温度、温度势函数与每步代价；头部进入目标 0.75 格内时成功结束。原版阶梯可作对照。"
            )
            reward_energy_weight = st.slider(
                "能量权重 w_e", 0.0, 0.5, 0.1, 0.01,
                format="%.2f",
                disabled=st.session_state.is_simulating or reward_variant not in ("energy", "continuous_energy"),
                key="reward_energy_weight_slider",
                help="单步能量惩罚系数：每步扣除 w_e × ΔE / max_energy（ΔE = 当步能量消耗）"
            )
            if body_model_type == "worm2d" and reward_variant == "energy":
                st.warning("⚠️ Worm2D 的能量从不衰减（ΔE≡0），含能量变体与原版逐值等价，仅在 CCB/ADB 上有效果。")
    else:
        # 课程学习默认值，匹配8.22版本
        width = height = 40  # 改为40，匹配8.22版本
        num_rounds = 800  # 改为800，匹配8.22版本
        initial_epsilon = 0.98
        min_epsilon = 0.1  # 改为0.1，匹配8.22版本
        epsilon_decay = 0.008
        discount_factor = 0.95
        steps_per_round = 500
        learning_rate = 0.4
        neural_lr = 0.001  # 改为0.001，匹配8.22版本
        hidden_size = 32  # 改为32，匹配8.22版本
        weight_decay = 5e-4
        # 奖励函数变体（课程学习默认原版）
        reward_variant = "original"
        reward_energy_weight = 0.1
        action_size = 4  # 课程学习仅 Worm2D，固定 4 方向

    st.markdown("---")
    
    # ==========================================================================
    # 控制按钮区域
    # ==========================================================================
    st.markdown("### 运行")
    can_run = body_model_type == "worm2d" or mode_choice == "标准训练模式"
    if not can_run:
        st.caption("此身体模型适用于标准训练。迁移与课程学习请选择多节段链条。")

    run_control = st.empty()
    with run_control.container():
        if not st.session_state.is_simulating:
            st.button("开始实验", use_container_width=True, type="primary",
                      key="start_experiment_button", on_click=request_experiment, disabled=not can_run)
        else:
            st.button("停止训练", use_container_width=True, type="secondary",
                      key="stop_experiment_button", on_click=request_stop)

# ==============================================================================
# 主界面区域 - 紧凑设计
# ==============================================================================

start_button = st.session_state.pop("run_requested", False)
if not start_button:
    if st.session_state.pop("stopped_feedback", False):
        st.info("本次训练已停止。最近一次完成的结果仍然保留。")
    if mode_choice == "标准训练模式":
        preview_field = clean_label(field_name)
    elif mode_choice == "迁移学习实验":
        preview_field = f"{clean_label(source_field_display)} → {clean_label(target_field_display)}"
    else:
        preview_field = clean_label(test_stage_name)
    if st.session_state.get("last_result"):
        render_experiment_preview(clean_label(body_model_display), clean_label(method_choice), preview_field, mode_choice)
    else:
        render_welcome(body_model_type, clean_label(body_model_display), clean_label(method_choice), preview_field, mode_choice)
    st.button("开始新实验" if st.session_state.get("last_result") else "开始实验", type="primary", key="main_start_experiment",
              on_click=request_experiment, disabled=not can_run, help="使用左侧当前配置开始训练")
    st.caption("调整左侧参数，然后开始。结果将在此处展示。")
    if st.session_state.get("last_result"):
        render_saved_results(st.session_state.last_result)

if start_button:
    st.session_state.is_simulating = True
    st.session_state.stop_requested = False
    
    # 准备实验配置
    parent_dir = os.path.join(os.path.expanduser("~"), "Desktop", "C_Elegans_Sim_Results")
    config = ExperimentConfig(experiment_name, parent_dir=parent_dir)
    st.session_state.current_config = config
    
    use_neural = "DQN" in method_choice

    # 非Worm2D身体模型不支持迁移学习和课程学习 (需要DQN)
    if body_model_type != "worm2d" and mode_choice != "标准训练模式":
        st.error("此身体模型适用于标准训练。迁移与课程学习请选择多节段链条。")
        st.session_state.is_simulating = False
        st.stop()
    
    # 身体参数字典 - 关键修复：确保参数正确传递
    body_params = {
        "action_size": action_size,
        "num_segments": num_segments,
        "sample_count": sample_count,
        "segment_length": segment_length,
        "body_length": body_length_ui,
        "head_radius": head_radius,
        "body_width": body_width,
        "max_turn_angle": max_turn_angle,
        "forward_speed": forward_speed,
        "backward_speed": backward_speed,
        "turning_speed": turning_speed,
        "curvature_limit_deg": curvature_limit,
        "angular_constraint": curvature_limit,
        "length_stiffness": length_stiffness,
        "curvature_stiffness": curvature_stiffness,
        "damping": damping,
        "wave_amplitude": wave_amplitude,
        "wave_frequency": wave_frequency,
        "wave_speed": wave_speed,
        "wave_length": wave_length,
        # RFT 波驱动参数 (仅 ADB 使用)
        "steer_bias": 0.0,
        "sub_steps": sub_steps,
        "drag_coeff": drag_coeff,
        "curriculum_freeze_steps": curriculum_freeze_steps,
        "wave_envelope": wave_envelope,
        # Actor-Critic 超参数
        "ac_hidden_size": ac_hidden_size,
        "ac_actor_lr": ac_actor_lr,
        "ac_critic_lr": ac_critic_lr,
        "ac_gamma": discount_factor,
        "ac_random_starts": locals().get("ac_random_starts", True),
        "ac_seed": int(locals().get("ac_seed", 7)),
        "ac_batch_size": ac_batch_size,
        "ac_noise_scale": ac_noise_scale,
    }
    
    # 噪声参数字典 - 关键修复：确保参数正确传递
    noise_params = {
        "position_noise": position_noise,
        "angle_noise": angle_noise,
        "thermal_noise": thermal_noise,
        "action_noise": action_noise,
        "noise_correlation": noise_correlation
    }
    
    # 显示参数确认信息
    
    # 基础训练参数 - 包含所有必需的参数
    if mode_choice == "课程学习实验":
        training_params = {
            "steps_per_round": steps_per_round,
            "num_rounds": num_rounds,
            "initial_epsilon": initial_epsilon,
            "min_epsilon": min_epsilon,
            "epsilon_decay": epsilon_decay,
            "discount_factor": discount_factor,
            "learning_rate": learning_rate,
            "method": "Dueling DQN",
            "neural_lr": neural_lr,
            "hidden_size": hidden_size,
            "weight_decay": weight_decay,
            "body_model_type": body_model_type,
            "use_state_v2": use_state_v2,
            "reward_variant": reward_variant,
            "reward_energy_weight": reward_energy_weight,
            "body_params": body_params,  # 确保传递
            "noise_params": noise_params  # 确保传递
        }
    else:
        training_params = {
            "width": width,
            "height": height,
            "num_rounds": num_rounds,
            "steps_per_round": steps_per_round,
            "initial_epsilon": initial_epsilon,
            "learning_rate": learning_rate,
            "min_epsilon": min_epsilon,  # 使用配置的最小epsilon
            "epsilon_decay": epsilon_decay,  # 使用配置的epsilon衰减
            "discount_factor": discount_factor,  # 使用配置的折扣因子
            "method": method_choice,
            "neural_lr": neural_lr,
            "hidden_size": hidden_size,  # 使用配置的隐藏层大小
            "weight_decay": weight_decay if "DQN" in method_choice else 5e-4,  # 添加权重衰减
            "body_model_type": body_model_type,
            "use_state_v2": use_state_v2,
            "reward_variant": reward_variant,
            "reward_energy_weight": reward_energy_weight,
            "body_params": body_params,  # 确保传递
            "noise_params": noise_params  # 确保传递
        }
    
    # 实验信息展示 - 紧凑版
    st.markdown("### 实验概览")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>训练方式</h4>
            <p style="font-size: 1em; margin: 0;">{clean_label(mode_choice)}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>学习算法</h4>
            <p style="font-size: 1em; margin: 0;">{clean_label(method_choice)}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        if mode_choice == "标准训练模式":
            display_text = field_name
        elif mode_choice == "迁移学习实验":
            display_text = f"{source_field}→{target_field}"
        else:
            display_text = test_stage_name
            
        st.markdown(f"""
        <div class="metric-card">
            <h4>温度环境</h4>
            <p style="font-size: 1em; margin: 0;">{clean_label(display_text)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 身体参数展示
    st.markdown("#### 身体配置")
    # 模型名称映射
    model_display_name = {
        "worm2d": "多节段链条",
        "continuous_centerline": "连续中心线",
        "active_deformation": "主动形变波",
    }.get(body_model_type, body_model_type)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="compact-card">
            <strong>模型:</strong> {model_display_name}<br>
            <strong>采样点:</strong> {sample_count}
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="compact-card">
            <strong>头径:</strong> {head_radius}px<br>
            <strong>体宽:</strong> {body_width}px
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="compact-card">
            <strong>转向角:</strong> {max_turn_angle}°<br>
            <strong>前进:</strong> {forward_speed}px
        </div>
        """, unsafe_allow_html=True)
    with col4:
        if body_model_type == "active_deformation":
            card_text = f"<strong>波幅:</strong> {wave_amplitude}<br><strong>波频:</strong> {wave_frequency}"
        elif body_model_type == "continuous_centerline":
            card_text = f"<strong>阻尼:</strong> {damping}<br><strong>最大弯折:</strong> {curvature_limit}°"
        else:
            card_text = f"<strong>后退:</strong> {backward_speed}px<br><strong>转向:</strong> {turning_speed}px"
        st.markdown(f"""
        <div class="compact-card">
            {card_text}
        </div>
        """, unsafe_allow_html=True)
    
    # 进度条和状态
    progress_bar = st.progress(0, text="准备实验…")
    status_placeholder = st.empty()
    
    # 结果展示区域 - 紧凑版
    st.markdown("### 训练结果")
    
    result_tabs = st.tabs(["曲线与轨迹", "运动回放", "运行日志"])
    
    with result_tabs[0]:
        image_placeholder = st.empty()
        image_placeholder.markdown("""
        <div class="info-box">
            <h4>训练曲线与轨迹</h4>
            <p>训练完成后显示详细分析图表。</p>
        </div>
        """, unsafe_allow_html=True)
    
    with result_tabs[1]:
        video_placeholder = st.empty()
        video_placeholder.markdown("""
        <div class="info-box">
            <h4>运动回放</h4>
            <p>展示线虫训练过程中的行为轨迹。</p>
        </div>
        """, unsafe_allow_html=True)
    
    with result_tabs[2]:
        log_placeholder = st.empty()
        log_placeholder.code("⏳ 等待实验开始...", language="text")

    # 启动仿真引擎
    completed = False
    try:
        if mode_choice == "标准训练模式":
            engine = run_standard_simulation_engine(
                config, training_params, field_type, use_neural, 
                enable_step_tracking=enable_step_tracking
            )
        elif mode_choice == "迁移学习实验":
            engine = run_transfer_simulation_engine(
                config, training_params, source_field, target_field, use_neural
            )
        elif mode_choice == "课程学习实验":
            engine = run_curriculum_simulation_engine(
                config, training_params, test_stage_idx, env_size, 
                enable_step_tracking=enable_step_tracking
            )

        # 🔧 关键修复：优化的实时处理引擎输出，添加完成检测
        log_content = deque(maxlen=50)  # 使用双端队列更高效，最多保存50条日志
        last_stats = {}
        update_counter = 0  # 更新计数器
        experiment_failed = False
        
        try:
            for current, total, message, stats in engine:
                if st.session_state.stop_requested:
                    break
                    
                if current == -1:
                    experiment_failed = True
                    error_html = f"""
                    <div class="error-box">
                        <h4>实验未完成</h4>
                        <p>{message}</p>
                    </div>
                    """
                    st.markdown(error_html, unsafe_allow_html=True)
                    break
                
                # 更新进度条 - 直接更新，无需额外调用
                progress = current / total if total > 0 else 1.0
                progress_text = clean_label(message)
                progress_bar.progress(min(progress, 1.0), text=progress_text)
                
                # 更新状态信息
                if stats:
                    last_stats.update(stats)
                    if 'phase' in stats:
                        phase_names = {"init": "准备实验", "source_training": "源环境训练", "target_testing": "目标环境评估", "training": "正在训练", "testing": "正在评估", "control": "对照实验", "results": "整理结果"}
                        phase_label = phase_names.get(stats['phase'], "正在运行")
                        status_html = f'<div class="status-box"><h4><span class="live-dot"></span>{phase_label}</h4>'

                        if 'round' in stats:
                            status_html += f"<p><strong>轮次:</strong> {stats['round']}</p>"
                        if 'reward' in stats:
                            status_html += f"<p><strong>奖励:</strong> {stats['reward']:.2f}</p>"
                        elif 'epsilon' in stats:
                            status_html += f"<p><strong>探索率:</strong> {stats['epsilon']:.3f}</p>"
                            
                        status_html += "</div>"
                        status_placeholder.markdown(status_html, unsafe_allow_html=True)
                
                # 更新日志 - 使用双端队列，自动限制长度
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                log_content.append(f"[{timestamp}] {message}")
                
                # 只在特定间隔更新日志显示，减少UI刷新频率
                update_counter += 1
                if update_counter % 3 == 0:  # 每3次更新显示一次日志
                    log_placeholder.code("\n".join(log_content), language="text")
                
                # 🔧 关键修复：检测完成状态
                if current >= total and total > 0:
                    print("🔧 调试：检测到进度完成信号，准备退出循环")
                    break
                    
        except StopIteration:
            # 🔧 添加：正常的生成器结束处理
            print("🔧 调试：生成器正常结束")
        except Exception as gen_error:
            experiment_failed = True
            print(f"❌ 生成器处理错误: {gen_error}")
            error_html = f"""
            <div class="error-box">
                <h4>处理失败</h4>
                <p>{gen_error}</p>
            </div>
            """
            st.markdown(error_html, unsafe_allow_html=True)

        # 实验完成处理
        print("🔧 调试：开始实验完成处理...")
        if experiment_failed:
            status_placeholder.error("实验未完成，请查看错误信息；没有生成有效结果时不会显示成功。")
        if not st.session_state.stop_requested and not experiment_failed:
            progress_bar.progress(1.0, text="训练完成")
            
            success_html = """
            <div class="success-box">
                <h4>训练完成</h4>
                <p>结果已保存到指定目录。</p>
            </div>
            """
            status_placeholder.markdown(success_html, unsafe_allow_html=True)
            
            st.session_state.last_result = {"output_dir": str(config.output_dir), "name": experiment_name}
            completed = True
    except Exception as e:
        st.error(f"❌ 实验运行错误: {e}")
        st.code(traceback.format_exc(), language="python")
    
    finally:
        st.session_state.is_simulating = False
        st.session_state.stop_requested = False
        if completed:
            # 结果包已保存，重新启用参数；请求标志已消费，不会再次训练。
            st.rerun()
        with run_control.container():
            st.button("开始实验", use_container_width=True, type="primary",
                      key="start_experiment_button", on_click=request_experiment)
        st.button("配置下一次实验", key="restart_button")

# 简洁页脚
st.markdown('<footer class="lab-footer"><span>C. elegans Lab</span><span>感知 · 决策 · 运动</span></footer>', unsafe_allow_html=True)
