"""
仿真引擎模块 - 完整封装所有训练模式
包含标准训练、迁移学习、课程学习的完整功能
支持 Streamlit 实时进度监控和中断控制
"""

import os
import sys
import io
import numpy as np
import time
import streamlit as st
import random
import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 🔧 使用非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 🔧 Windows GBK编码修复：强制stdout使用UTF-8，避免emoji打印崩溃
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    elif hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

# 🔧 Mac系统中文字体配置
def setup_chinese_font():
    """为Mac系统设置中文字体"""
    try:
        # Mac系统常见中文字体列表
        mac_chinese_fonts = [
            'PingFang SC',      # macOS默认中文字体
            'Hiragino Sans GB', # macOS系统字体
            'STHeiti',          # 华文黑体
            'Arial Unicode MS', # 万能Unicode字体
            'SimHei',           # Windows黑体(如果安装了)
            'Microsoft YaHei'   # 微软雅黑(如果安装了)
        ]
        
        # 检查可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font_name in mac_chinese_fonts:
            if font_name in available_fonts:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 使用中文字体: {font_name}")
                return True
        
        # 如果没有找到中文字体，使用英文标题
        print("⚠️ 未找到中文字体，将使用英文标题")
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False
        
    except Exception as e:
        print(f"⚠️ 字体设置失败: {e}")
        return False

# 在所有绘图函数开始前调用
has_chinese_font = setup_chinese_font()

# 从core文件夹中导入所有必要的模块
from core.environment import ExperimentConfig, Environment2D
from core.worm_body import create_body_model
from core.visualization import plot_training_results_2d, create_training_animation_2d, create_training_animation_2d_dynamic_mp4
from core.utils import (save_training_log, save_q_table, save_body_metrics, setup_neural_network, 
                      reset_worm_for_new_round, create_temperature_environment, 
                      generate_dynamic_rotating_double_center, generate_dynamic_rotating_quad_center)
from core.neural_networks import PYTORCH_AVAILABLE
from core.training_stats import create_step_tracker, create_dual_center_tracker


def get_start_position(field_type, width, height):
    """根据场景类型获取起始位置，支持等比例缩放"""
    scale_x = width / 80.0
    scale_y = height / 80.0
    
    base_positions = {
        'single_center': (10, 10),
        'dual_center': (70, 10),  # 修正为(70, 10)，匹配8.22版本
        'spotty_field': (40, 40),
        'ring_hotspot': (5, 5),
        'maze_thermal_channel': (15, 65),
        'complex_maze_channel': (10, 70),
        'dynamic_rotating_double_center': (10, 10),
        'dynamic_rotating_quad_center': (10, 10),
        'complex_maze_mirror': (70, 70)
    }
    
    if field_type in base_positions:
        base_x, base_y = base_positions[field_type]
        scaled_x = int(base_x * scale_x)
        scaled_y = int(base_y * scale_y)
        return (scaled_x, scaled_y)
    else:
        return (random.randint(3, width-4), random.randint(3, height-4))


def create_training_body_model(start_pos, width, height, training_params):
    """按训练参数创建身体模型，默认使用兼容的多节段链式身体。"""
    body_params = training_params.get('body_params', {})
    noise_params = training_params.get('noise_params', {})
    if not isinstance(body_params, dict):
        print(f"⚠️ 警告：body_params 不是字典类型: {type(body_params)}")
        body_params = {}
    if not isinstance(noise_params, dict):
        print(f"⚠️ 警告：noise_params 不是字典类型: {type(noise_params)}")
        noise_params = {}

    return create_body_model(
        model_type=training_params.get("body_model_type", "worm2d"),
        start_pos=start_pos,
        width=width,
        height=height,
        body_params=body_params,
        noise_params=noise_params,
    )


def run_standard_simulation_engine(config, training_params, field_type, use_neural_network, enable_step_tracking=False):
    """
    标准训练模式的引擎，使用 yield 实时报告进度。
    """
    try:
        # --- 1. 初始化阶段 ---
        yield 0, 1000, "🔄 开始初始化实验环境...", {'phase': 'init'}
        
        config.field_type = field_type
        width, height = training_params.get("width", 40), training_params.get("height", 40)  # 改为40，匹配8.22版本
        
        print(f"🔧 调试：环境尺寸 = {width}x{height}")
        yield 1, 1000, f"📐 设置环境尺寸: {width}x{height}", {'phase': 'init'}

        # 创建温度环境
        print(f"🔧 调试：创建温度环境，类型={field_type}")
        temp_array, best_point = create_temperature_environment(width, height, None, field_type)
        env = Environment2D(temp_array, best_point=best_point)
        print(f"🔧 调试：环境创建完成，最佳点={best_point}")
        yield 2, 1000, f"🌡️ 温度环境已创建: {field_type}", {'phase': 'init'}

        # 获取起始位置
        start_pos = get_start_position(field_type, width, height)
        print(f"🔧 调试：起始位置={start_pos}")
        yield 3, 1000, f"📍 设置起始位置: {start_pos}", {'phase': 'init'}

        # 🔧 关键修复：安全的参数处理
        print("🔧 调试：准备创建线虫对象...")
        
        # 确保参数字典存在且格式正确
        body_params = training_params.get('body_params', {})
        noise_params = training_params.get('noise_params', {})
        
        # 验证参数字典
        if not isinstance(body_params, dict):
            print(f"⚠️ 警告：body_params 不是字典类型: {type(body_params)}")
            body_params = {}
        
        if not isinstance(noise_params, dict):
            print(f"⚠️ 警告：noise_params 不是字典类型: {type(noise_params)}")
            noise_params = {}
            
        print(f"🔧 调试：body_params = {body_params}")
        print(f"🔧 调试：noise_params = {noise_params}")
        
        yield 4, 1000, "🐛 准备创建线虫对象...", {'phase': 'init'}
        
        # 创建线虫对象 - 使用超时保护
        try:
            print("🔧 调试：开始通过身体模型工厂创建线虫对象...")
            worm = create_training_body_model(start_pos, width, height, training_params)
            print("🔧 调试：线虫身体模型创建成功！")
            yield 5, 1000, "✅ 线虫对象创建成功", {'phase': 'init'}
            
        except Exception as worm_error:
            print(f"❌ 线虫创建失败: {worm_error}")
            yield -1, -1, f"线虫对象创建失败: {worm_error}", {}
            return

        # 设置 Actor-Critic (优先级高于 DQN / Q-Learning)
        method_name = training_params.get("method", "")
        if "Actor-Critic" in method_name:
            if hasattr(worm, 'setup_actor_critic'):
                try:
                    worm.setup_actor_critic()
                    print("🔧 调试：Actor-Critic 初始化完成")
                    yield 6, 1000, "🎯 Actor-Critic (DDPG) 已配置", {'phase': 'init'}
                except Exception as ac_error:
                    print(f"❌ Actor-Critic 初始化失败: {ac_error}")
                    yield -1, -1, f"Actor-Critic 初始化失败: {ac_error}", {}
                    return
            else:
                yield -1, -1, "所选身体模型不支持 Actor-Critic", {}
                return

        # 设置神经网络 (DQN / Dueling DQN)
        elif use_neural_network:
            print("🔧 调试：设置神经网络...")
            if not PYTORCH_AVAILABLE:
                yield -1, -1, "PyTorch 未安装，无法使用神经网络。", {}
                return
            try:
                setup_neural_network(worm, training_params)
                print("🔧 调试：神经网络设置完成")
                yield 6, 1000, "🧠 神经网络已配置", {'phase': 'init'}
            except Exception as nn_error:
                print(f"❌ 神经网络设置失败: {nn_error}")
                yield -1, -1, f"神经网络设置失败: {nn_error}", {}
                return
        else:
            worm.use_neural = False
            yield 6, 1000, "📋 使用 Q-Learning 算法", {'phase': 'init'}

        # 测试环境状态向量
        print("🔧 调试：测试环境状态向量...")
        try:
            test_state = env.get_state_vector((worm.x, worm.y), worm=worm)
            print(f"🔧 调试：状态向量测试成功，维度={test_state.shape}")
            yield 7, 1000, f"🔍 状态向量测试通过: {test_state.shape}", {'phase': 'init'}
        except Exception as state_error:
            print(f"❌ 状态向量测试失败: {state_error}")
            yield -1, -1, f"状态向量计算失败: {state_error}", {}
            return

        # --- 2. 训练准备 ---
        yield 10, 1000, "📊 初始化训练统计...", {'phase': 'init'}
        
        steps_to_reach_target = []
        average_temperatures = []
        detailed_temperature_history = []
        
        # 双热源追踪器初始化
        dual_center_tracker = None
        if field_type == 'dual_center' and enable_step_tracking:
            print("🔧 调试：初始化双热源追踪器...")
            scale_x = width / 80.0
            scale_y = height / 80.0
            
            base_center1 = (80 // 3, 80 // 3)
            base_center2 = (2 * 80 // 3, 2 * 80 // 3)
            
            high_temp_center = (int(base_center1[0] * scale_x), int(base_center1[1] * scale_y))
            low_temp_center = (int(base_center2[0] * scale_x), int(base_center2[1] * scale_y))
            
            try:
                dual_center_tracker = create_dual_center_tracker(
                    high_temp_center=high_temp_center,
                    low_temp_center=low_temp_center,
                    proximity_threshold=5.0
                )
                print("🔧 调试：双热源追踪器创建成功")
            except Exception as tracker_error:
                print(f"⚠️ 双热源追踪器创建失败: {tracker_error}")

        # --- 3. 训练循环 ---
        yield 20, 1000, "🚀 开始训练循环...", {'phase': 'training'}
        
        num_rounds = training_params.get("num_rounds", 800)  # 改为800，匹配8.22版本
        all_histories, all_rewards = [], []
        
        print(f"🔧 调试：准备进行 {num_rounds} 轮训练")

        for round_num in range(num_rounds):
            # 检查停止信号
            if st.session_state.get('stop_requested', False):
                yield round_num * 10 + 20, 1000, "⏹️ 收到停止信号，正在中断...", {}
                break

            # 更新进度和状态
            progress = 20 + round_num * 950 // num_rounds  # 20-970的范围
            progress_msg = f"正在进行第 {round_num + 1}/{num_rounds} 轮标准训练..."
            stats = {
                'phase': 'training',
                'round': round_num + 1,
                'total_rounds': num_rounds,
                'reward': worm.total_reward,  # 添加当前奖励值
                'epsilon': max(training_params["min_epsilon"], 
                             training_params["initial_epsilon"] - round_num * training_params["epsilon_decay"]),
                'method': 'Neural Network' if use_neural_network else 'Q-Learning',
                'field_type': field_type
            }
            yield progress, 1000, progress_msg, stats
            
            # 每10轮输出一次调试信息
            if (round_num + 1) % 10 == 0:
                print(f"🔧 调试：开始第 {round_num + 1} 轮训练")
            
            # 重置线虫
            try:
                start_pos = get_start_position(field_type, width, height)
                reset_worm_for_new_round(worm, env, start_pos, width, height, field_type)
                if (round_num + 1) % 10 == 0:
                    print(f"🔧 调试：线虫重置完成，位置={start_pos}")
            except Exception as reset_error:
                print(f"❌ 线虫重置失败: {reset_error}")
                continue  # 跳过这一轮
            
            epsilon = max(training_params["min_epsilon"], 
                         training_params["initial_epsilon"] - round_num * training_params["epsilon_decay"])
            
            # 步数和温度统计
            round_steps = training_params["steps_per_round"]
            round_temperature_sum = 0.0
            temperature_count = 0
            round_temp_history = []
            
            # 单轮训练
            steps_per_round = training_params.get("steps_per_round", 100)  # 使用配置的步数
            if (round_num + 1) % 10 == 0:
                print(f"🔧 调试：单轮训练 {steps_per_round} 步")
            
            for step in range(steps_per_round):
                try:
                    # ...existing code...
                    # 动态温度场更新
                    if field_type.startswith('dynamic'):
                        if field_type == 'dynamic_rotating_double_center':
                            temp_array, best_point = generate_dynamic_rotating_double_center(width, height, t=step)
                            env.update_temperature_array(temp_array, best_point)
                        elif field_type == 'dynamic_rotating_quad_center':
                            temp_array, best_point = generate_dynamic_rotating_quad_center(width, height, t=step)
                            env.update_temperature_array(temp_array, best_point)
                    # 线虫决策和移动
                    moved = worm.decide_move(env, epsilon=epsilon, 
                                           alpha=training_params["learning_rate"], 
                                           gamma=training_params["discount_factor"])
                    if not moved:
                        print(f"⚠️ 第{step}步移动失败")
                        break
                    # 温度统计
                    if enable_step_tracking and moved:
                        if hasattr(worm, 'body_temperatures') and worm.body_temperatures:
                            valid_temps = [t for t in worm.body_temperatures if t != -float('inf') and t > 0]
                            if valid_temps:
                                step_avg_temp = sum(valid_temps) / len(valid_temps)
                                round_temperature_sum += step_avg_temp
                                temperature_count += 1
                                round_temp_history.append(step_avg_temp)
                    
                    # 步数统计
                    if enable_step_tracking and round_steps == training_params["steps_per_round"]:
                        head_pos = worm.body_segments[0] if hasattr(worm, 'body_segments') and worm.body_segments else start_pos
                        target_distance = np.sqrt((head_pos[0] - best_point[0])**2 + (head_pos[1] - best_point[1])**2)
                        if target_distance <= 3.0:
                            round_steps = step + 1
                            
                except Exception as step_error:
                    print(f"❌ 第{step}步训练失败: {step_error}")
                    break

            # 记录统计数据
            if enable_step_tracking:
                steps_to_reach_target.append(round_steps)
                round_avg_temp = round_temperature_sum / temperature_count if temperature_count > 0 else 0.0
                average_temperatures.append(round_avg_temp)
                detailed_temperature_history.append(round_temp_history.copy())
            
            # 双热源追踪
            if dual_center_tracker is not None:
                try:
                    dual_center_tracker.track_round(worm)
                except Exception as track_error:
                    print(f"⚠️ 双热源追踪失败: {track_error}")

            all_histories.append(worm.history.copy())
            all_rewards.append(worm.total_reward)
            
            # 更新进度状态显示当前轮次的奖励
            current_reward = worm.total_reward
            progress_complete = 20 + (round_num + 1) * 950 // num_rounds
            stats_complete = {
                'phase': 'training',
                'round': round_num + 1,
                'total_rounds': num_rounds,
                'reward': current_reward,
                'method': 'Neural Network' if use_neural_network else 'Q-Learning',
                'field_type': field_type
            }
            yield progress_complete, 1000, f"第 {round_num + 1} 轮完成，奖励: {current_reward:.2f}", stats_complete
            
            if (round_num + 1) % 10 == 0 or round_num == num_rounds - 1:
                print(f"🔧 进度：第 {round_num + 1} 轮完成，奖励={worm.total_reward:.2f}")

        # --- 4. 生成结果 ---
        yield 980, 1000, "✅ 训练完成，正在生成结果...", {'phase': 'results'}
        
        if all_rewards:
            print(f"🔧 调试：开始生成结果，共{len(all_rewards)}轮数据")
            
            # 生成步数统计图表
            if enable_step_tracking and steps_to_reach_target:
                try:
                    print("🔧 开始生成步数统计图表...")
                    # 🔧 添加数据验证
                    if len(steps_to_reach_target) > 0:
                        generate_step_statistics_plots(config, steps_to_reach_target, all_rewards, 
                                                     field_type, average_temperatures, detailed_temperature_history)
                        print("🔧 调试：步数统计图表生成完成")
                    else:
                        print("⚠️ 步数数据为空，跳过统计图表生成")
                except Exception as stats_error:
                    print(f"❌ 步数统计图表生成失败: {stats_error}")
                    import traceback
                    traceback.print_exc()
            else:
                print("🔧 步数追踪未启用或无数据，跳过统计图表")
            
            # 双热源统计
            if dual_center_tracker is not None:
                try:
                    dual_center_tracker.print_summary()
                except Exception as summary_error:
                    print(f"⚠️ 双热源统计总结失败: {summary_error}")
            
            # 保存和可视化结果
            try:
                save_and_visualize_results(config, all_histories, all_rewards, worm, env, 
                                         training_params, temp_array, best_point)
                print("🔧 调试：结果保存和可视化完成")
            except Exception as save_error:
                print(f"❌ 结果保存失败: {save_error}")
            
            final_stats = {
                'total_rounds': len(all_rewards),
                'avg_reward': np.mean(all_rewards),
                'final_performance': np.mean(all_rewards[-10:]) if len(all_rewards) >= 10 else np.mean(all_rewards),
                'improvement': np.mean(all_rewards[-3:]) - np.mean(all_rewards[:3]) if len(all_rewards) >= 6 else 0
            }
            
            yield 1000, 1000, f"🎉 标准训练完成！平均奖励: {final_stats['avg_reward']:.2f}", final_stats
        else:
            yield 1000, 1000, "⚠️ 训练完成，但没有生成有效数据", {}

        # 🔧 关键修复：添加生成器完成信号
        print("🔧 调试：生成器即将结束，发送完成信号")
        return  # 明确结束生成器

    except Exception as e:
        import traceback
        error_msg = f"标准训练引擎错误: {e}\n{traceback.format_exc()}"
        print(f"❌ 引擎异常: {error_msg}")
        yield -1, -1, error_msg, {}
        return  # 🔧 添加：确保异常时也能正确结束


def run_transfer_simulation_engine(config, training_params, source_field, target_field, use_neural_network):
    """
    迁移学习模式的引擎，使用 yield 实时报告进度。
    """
    try:
        yield 0, 1200, "🚀 启动迁移学习实验...", {'phase': 'init', 'source_field': source_field, 'target_field': target_field}
        
        if source_field == target_field:
            yield -1, -1, "❌ 错误：源和目标温度场相同", {}
            return
        
        width, height = 80, 80
        
        # --- 第一阶段：源环境训练 ---
        yield 0, 1200, f"🏋️ 阶段1：在源环境 '{source_field}' 开始预训练...", {'phase': 'source_training'}
        
        source_temp_array, source_best_point = create_temperature_environment(width, height, None, source_field)
        source_env = Environment2D(source_temp_array, best_point=source_best_point)
        
        start_pos = get_start_position(source_field, width, height)
        source_worm = create_training_body_model(start_pos, width, height, training_params)
        
        if use_neural_network:
            if not PYTORCH_AVAILABLE:
                yield -1, -1, "PyTorch 未安装，无法使用神经网络。", {}
                return
            setup_neural_network(source_worm, training_params)
        
        # 源环境训练循环
        source_rewards = []
        source_rounds = training_params.get("num_rounds", 600)
        
        for round_num in range(source_rounds):
            if st.session_state.get('stop_requested', False):
                yield round_num, 1200, "⏹️ 收到停止信号，正在中断...", {}
                break
                
            progress_msg = f"源环境训练: {round_num + 1}/{source_rounds} 轮"
            stats = {
                'phase': 'source_training',
                'round': round_num + 1,
                'total_rounds': source_rounds,
                'reward': source_worm.total_reward,  # 添加当前奖励值
                'source_field': source_field
            }
            yield round_num, 1200, progress_msg, stats
            
            # 每轮重新获取起始位置，确保正确重置
            round_start_pos = get_start_position(source_field, width, height)
            reset_worm_for_new_round(source_worm, source_env, round_start_pos, width, height, source_field)
            epsilon = max(training_params["min_epsilon"],
                         training_params["initial_epsilon"] - round_num * training_params["epsilon_decay"])
            
            for step in range(training_params["steps_per_round"]):
                # 动态温度场更新
                if source_field.startswith('dynamic'):
                    if source_field == 'dynamic_rotating_double_center':
                        temp_array, best_point = generate_dynamic_rotating_double_center(width, height, t=step)
                        source_env.update_temperature_array(temp_array, best_point)
                    elif source_field == 'dynamic_rotating_quad_center':
                        temp_array, best_point = generate_dynamic_rotating_quad_center(width, height, t=step)
                        source_env.update_temperature_array(temp_array, best_point)
                        
                moved = source_worm.decide_move(source_env, epsilon=epsilon,
                                              alpha=training_params["learning_rate"],
                                              gamma=training_params["discount_factor"])
                if not moved:
                    break
            
            source_rewards.append(source_worm.total_reward)
        
        source_performance = np.mean(source_rewards[-20:]) if len(source_rewards) >= 20 else np.mean(source_rewards)
        
        # --- 第二阶段：目标环境测试 ---
        yield source_rounds, 1200, f"🎯 阶段2：在目标环境 '{target_field}' 进行零样本测试...", {'phase': 'target_testing'}
        
        target_temp_array, target_best_point = create_temperature_environment(width, height, None, target_field)
        target_env = Environment2D(target_temp_array, best_point=target_best_point)
        
        target_start_pos = get_start_position(target_field, width, height)
        veteran_worm = create_training_body_model(target_start_pos, width, height, training_params)
        
        if use_neural_network:
            setup_neural_network(veteran_worm, training_params)
            # 迁移权重
            veteran_worm.neural_network.load_state_dict(source_worm.neural_network.state_dict())
            veteran_worm.target_network.load_state_dict(source_worm.target_network.state_dict())
        else:
            # Q-Learning 权重迁移
            if hasattr(source_worm, 'q_table'):
                veteran_worm.q_table = source_worm.q_table.copy()
        
        # 零样本测试
        test_rewards = []
        test_histories = []
        test_rounds = 20
        
        for episode in range(test_rounds):
            if st.session_state.get('stop_requested', False):
                break
                
            progress_msg = f"目标环境测试: {episode + 1}/{test_rounds} 轮"
            stats = {
                'phase': 'target_testing',
                'episode': episode + 1,
                'reward': veteran_worm.total_reward,  # 添加当前奖励值
                'target_field': target_field
            }
            yield source_rounds + episode * 30, 1200, progress_msg, stats
            
            # 每轮重新获取起始位置，确保正确重置
            round_target_start_pos = get_start_position(target_field, width, height)
            reset_worm_for_new_round(veteran_worm, target_env, round_target_start_pos, width, height, target_field)
            
            for step in range(training_params["steps_per_round"]):
                # 动态温度场更新
                if target_field.startswith('dynamic'):
                    if target_field == 'dynamic_rotating_double_center':
                        temp_array, best_point = generate_dynamic_rotating_double_center(width, height, t=step)
                        target_env.update_temperature_array(temp_array, best_point)
                    elif target_field == 'dynamic_rotating_quad_center':
                        temp_array, best_point = generate_dynamic_rotating_quad_center(width, height, t=step)
                        target_env.update_temperature_array(temp_array, best_point)
                
                moved = veteran_worm.decide_move(target_env, epsilon=0.01, alpha=0.0,
                                               gamma=training_params["discount_factor"])
                if not moved:
                    break
            
            test_rewards.append(veteran_worm.total_reward)
            test_histories.append(veteran_worm.history.copy())
        
        transfer_performance = np.mean(test_rewards) if test_rewards else 0
        transfer_ratio = transfer_performance / source_performance if source_performance > 0 else 0
        
        # --- 保存结果 ---
        config.experiment_name = f"transfer_{source_field}_to_{target_field}_{config.experiment_name}"
        config.field_type = target_field
        
        save_and_visualize_results(config, test_histories, test_rewards, veteran_worm, 
                                  target_env, training_params, target_temp_array, target_best_point)
        
        # 最终结果
        final_stats = {
            'source_performance': source_performance,
            'transfer_performance': transfer_performance,
            'transfer_ratio': transfer_ratio,
            'source_field': source_field,
            'target_field': target_field
        }
        
        if transfer_ratio > 0.8:
            result_msg = f"🎉 卓越迁移！达到源环境 {transfer_ratio*100:.1f}% 的性能"
        elif transfer_ratio > 0.5:
            result_msg = f"✅ 良好迁移！达到源环境 {transfer_ratio*100:.1f}% 的性能"
        else:
            result_msg = f"⚠️ 有限迁移！仅达到源环境 {transfer_ratio*100:.1f}% 的性能"
        
        yield 1200, 1200, result_msg, final_stats

    except Exception as e:
        import traceback
        error_msg = f"迁移学习引擎错误: {e}\n{traceback.format_exc()}"
        yield -1, -1, error_msg, {}


def run_curriculum_simulation_engine(config, training_params, test_stage_idx, env_size, enable_step_tracking=False):
    """
    课程学习模式的引擎，使用 yield 实时报告进度。
    """
    try:
        yield 0, 5000, "🚀 启动课程学习实验（留出法交叉验证）...", {'phase': 'init'}
        
        # 定义所有可用课程阶段
        all_possible_stages = [
            {"name": "初级阶段", "field_type": "single_center", "rounds": 600, "initial_epsilon": 0.98},
            {"name": "双热源阶段", "field_type": "dual_center", "rounds": 600, "initial_epsilon": 0.9},
            {"name": "斑点阶段", "field_type": "spotty_field", "rounds": 600, "initial_epsilon": 0.85},
            {"name": "环形阶段", "field_type": "ring_hotspot", "rounds": 600, "initial_epsilon": 0.9},
            {"name": "迷宫阶段", "field_type": "maze_thermal_channel", "rounds": 600, "initial_epsilon": 0.99},
            {"name": "复杂迷宫阶段", "field_type": "complex_maze_channel", "rounds": 600, "initial_epsilon": 0.9},
            {"name": "双中心旋转阶段", "field_type": "dynamic_rotating_double_center", "rounds": 600, "initial_epsilon": 0.8},
            {"name": "终极四中心阶段", "field_type": "dynamic_rotating_quad_center", "rounds": 600, "initial_epsilon": 0.8}
        ]
        
        # 确定训练集和测试集
        test_stage = all_possible_stages[test_stage_idx]
        training_stages_with_indices = [(i, s) for i, s in enumerate(all_possible_stages) if i != test_stage_idx]
        
        # 如果测试集是复杂迷宫场，添加镜像迷宫到训练集
        if test_stage["field_type"] == "complex_maze_channel":
            mirror_stage = {
                "name": "镜像复杂迷宫阶段", 
                "field_type": "complex_maze_mirror", 
                "rounds": 600,
                "initial_epsilon": 0.9
            }
            training_stages_with_indices.append((99, mirror_stage))
        
        width = height = env_size
        base_training_params = training_params.copy()
        base_training_params.update({
            "weight_decay": 5e-4,
            "neural_lr": 0.00001
        })
        
        worm = None
        env = None
        stage_performances = []
        total_estimated_steps = sum(stage[1]["rounds"] for stage in training_stages_with_indices) + test_stage["rounds"] + 600
        current_step = 0
        
        # --- 训练阶段 ---
        if training_stages_with_indices:
            yield current_step, total_estimated_steps, f"🏋️ 开始训练阶段 ({len(training_stages_with_indices)}个阶段)", {'phase': 'training'}
            
            # 创建持久化环境对象
            initial_temp_array, initial_best_point = create_temperature_environment(width, height, None, training_stages_with_indices[0][1]["field_type"])
            env = Environment2D(initial_temp_array, best_point=initial_best_point)
            
            for stage_idx, (original_idx, stage) in enumerate(training_stages_with_indices):
                if st.session_state.get('stop_requested', False):
                    yield current_step, total_estimated_steps, "⏹️ 收到停止信号，正在中断...", {}
                    break
                    
                stage_msg = f"训练阶段 {stage_idx + 1}/{len(training_stages_with_indices)}: {stage['name']}"
                yield current_step, total_estimated_steps, stage_msg, {
                    'phase': 'training',
                    'stage_index': stage_idx,
                    'stage_name': stage['name'],
                    'field_type': stage['field_type']
                }
                
                # 更新环境
                temp_array, best_point = create_temperature_environment(width, height, None, stage["field_type"])
                env.temp_array = temp_array
                env.best_point = best_point
                
                start_pos = get_start_position(stage["field_type"], width, height)
                stage_training_params = base_training_params.copy()
                stage_training_params.update(stage)
                
                # 创建或重置线虫
                if worm is None:
                    worm = create_training_body_model(start_pos, width, height, training_params)
                    setup_neural_network(worm, stage_training_params)
                else:
                    # 重置经验池，保留神经网络权重
                    if hasattr(worm, 'experience_replay') and worm.experience_replay is not None:
                        from core.neural_networks import PrioritizedReplayBuffer
                        capacity = getattr(worm.experience_replay, 'capacity', 10000)
                        worm.experience_replay = PrioritizedReplayBuffer(capacity=capacity)
                
                # 确保网络可训练
                if hasattr(worm, 'neural_network') and worm.neural_network is not None:
                    worm.neural_network.train()
                    for param in worm.neural_network.parameters():
                        param.requires_grad = True
                
                stage_histories, stage_rewards = [], []
                
                # 单阶段训练循环
                for round_num in range(stage["rounds"]):
                    if st.session_state.get('stop_requested', False):
                        break
                        
                    current_step += 1

                    if round_num % 50 == 0:  # 每50轮更新一次进度
                        round_msg = f"{stage['name']} - 轮次 {round_num + 1}/{stage['rounds']}"
                        yield current_step, total_estimated_steps, round_msg, {
                            'phase': 'training',
                            'stage_name': stage['name'],
                            'round': round_num + 1,
                            'stage_rounds': stage['rounds'],
                            'reward': worm.total_reward  # 添加当前奖励值
                        }
                    
                    # 每轮重新获取起始位置，确保正确重置
                    round_start_pos = get_start_position(stage["field_type"], width, height)
                    reset_worm_for_new_round(worm, env, round_start_pos, width, height, stage["field_type"])
                    epsilon = max(stage_training_params["min_epsilon"], 
                                 stage_training_params.get("initial_epsilon", 0.98) - round_num * stage_training_params["epsilon_decay"])
                    
                    for step in range(stage_training_params["steps_per_round"]):
                        if 'dynamic' in stage['field_type']:
                            if stage['field_type'] == 'dynamic_rotating_double_center':
                                temp_array, best_point = generate_dynamic_rotating_double_center(width, height, t=step)
                                env.update_temperature_array(temp_array, best_point)
                            elif stage['field_type'] == 'dynamic_rotating_quad_center':
                                temp_array, best_point = generate_dynamic_rotating_quad_center(width, height, t=step)
                                env.update_temperature_array(temp_array, best_point)
                        
                        moved = worm.decide_move(env, epsilon=epsilon, 
                                               alpha=stage_training_params["learning_rate"], 
                                               gamma=stage_training_params["discount_factor"])
                    
                    stage_histories.append(worm.history.copy())
                    stage_rewards.append(worm.total_reward)
                
                # 记录阶段性能
                final_performance = np.mean(stage_rewards[-10:]) if len(stage_rewards) >= 10 else (np.mean(stage_rewards) if stage_rewards else 0)
                stage_performances.append({
                    'name': stage['name'], 
                    'performance': final_performance, 
                    'is_test': False
                })
                
                # 保存阶段结果
                if original_idx == 99:
                    subfolder_name = f"M_{stage['name']}_{stage['field_type']}"
                else:
                    subfolder_name = f"{original_idx+1:02d}_{stage['name']}_{stage['field_type']}"
                
                stage_config = ExperimentConfig(experiment_name=subfolder_name, parent_dir=config.output_dir)
                stage_config.field_type = stage['field_type']
                
                if stage_histories and stage_rewards:
                    if enable_step_tracking:
                        # 这里可以添加步数统计逻辑
                        pass
                    save_and_visualize_results(stage_config, stage_histories, stage_rewards, worm, env, 
                                             stage_training_params, temp_array, best_point)
        
        # --- 测试阶段 ---
        yield current_step, total_estimated_steps, "🏆 开始最终测试阶段", {'phase': 'testing'}
        
        if worm is None:
            start_pos = get_start_position(test_stage["field_type"], width, height)
            worm = create_training_body_model(start_pos, width, height, training_params)
            setup_neural_network(worm, base_training_params)
        
        # 准备测试环境
        temp_array, best_point = create_temperature_environment(width, height, None, test_stage["field_type"])
        env = Environment2D(temp_array, best_point=best_point)
        start_pos = get_start_position(test_stage["field_type"], width, height)
        
        test_training_params = base_training_params.copy()
        test_training_params.update(test_stage)
        
        # 执行零学习测试
        stage_histories, stage_rewards, test_results = yield from zero_shot_testing_engine(
            worm, env, test_stage, test_training_params, start_pos, width, height, current_step, total_estimated_steps
        )
        
        stage_performances.append({**test_results, 'name': test_stage['name'], 'is_test': True})
        
        # 保存测试结果
        subfolder_name = f"TEST_{test_stage_idx+1:02d}_{test_stage['name']}_{test_stage['field_type']}"
        stage_config = ExperimentConfig(experiment_name=subfolder_name, parent_dir=config.output_dir)
        stage_config.field_type = test_stage['field_type']
        
        if stage_histories and stage_rewards:
            save_and_visualize_results(stage_config, stage_histories, stage_rewards, worm, env, 
                                     test_training_params, temp_array, best_point)
        
        # --- 对照实验：新兵测试 ---
        yield current_step, total_estimated_steps, "🆚 对照实验：测试全新'新兵'的表现...", {'phase': 'control'}
        
        rookie_worm = create_training_body_model(start_pos, width, height, training_params)
        setup_neural_network(rookie_worm, base_training_params)
        
        rookie_histories, rookie_rewards, rookie_test_results = yield from zero_shot_testing_engine(
            rookie_worm, env, test_stage, test_training_params, start_pos, width, height, current_step, total_estimated_steps
        )
        
        # 最终结果分析
        veteran_performance = test_results.get('final_performance', 0)
        rookie_performance = rookie_test_results.get('final_performance', 0)
        
        final_stats = {
            'test_environment': test_stage['name'],
            'veteran_performance': veteran_performance,
            'rookie_performance': rookie_performance,
            'stage_performances': stage_performances,
            'total_training_stages': len(training_stages_with_indices)
        }
        
        if veteran_performance > rookie_performance * 1.5 and veteran_performance > 1:
            result_msg = "✅ 结论：模型表现出显著的泛化能力！"
        elif veteran_performance > rookie_performance:
            result_msg = "🤔 结论：模型表现出一定的泛化能力，但不够鲁棒。"
        else:
            result_msg = "❌ 结论：模型未能证明其泛化能力。"
        
        yield total_estimated_steps, total_estimated_steps, result_msg, final_stats

    except Exception as e:
        import traceback
        error_msg = f"课程学习引擎错误: {e}\n{traceback.format_exc()}"
        yield -1, -1, error_msg, {}


def zero_shot_testing_engine(worm, env, stage, training_params, start_pos, width, height, current_step, total_steps):
    """零学习测试引擎，支持 yield 进度报告"""
    
    # 固定网络参数
    if hasattr(worm, 'neural_network'):
        worm.neural_network.eval()
        for param in worm.neural_network.parameters():
            param.requires_grad = False
    
    if hasattr(worm, 'target_network'):
        worm.target_network.eval()
        for param in worm.target_network.parameters():
            param.requires_grad = False
    
    stage_histories = []
    stage_rewards = []
    
    # 基线测试
    baseline_rewards = []
    for test_round in range(10):
        if st.session_state.get('stop_requested', False):
            break
            
        if test_round % 5 == 0:
            yield current_step + test_round, total_steps, f"基线测试: {test_round + 1}/10", {
                'phase': 'baseline_testing',
                'round': test_round + 1,
                'reward': worm.total_reward  # 添加当前奖励值
            }
        
        # 每轮重新获取起始位置，确保正确重置
        round_start_pos = get_start_position(stage["field_type"], width, height)
        reset_worm_for_new_round(worm, env, round_start_pos, width, height, stage["field_type"])
        
        for step in range(training_params["steps_per_round"]):
            # 动态温度场更新
            if stage["field_type"].startswith('dynamic'):
                if stage["field_type"] == 'dynamic_rotating_double_center':
                    temp_array, best_point = generate_dynamic_rotating_double_center(width, height, t=step)
                    env.update_temperature_array(temp_array, best_point)
                elif stage["field_type"] == 'dynamic_rotating_quad_center':
                    temp_array, best_point = generate_dynamic_rotating_quad_center(width, height, t=step)
                    env.update_temperature_array(temp_array, best_point)
                    
            moved = worm.decide_move(env, epsilon=0.01, alpha=0.0, gamma=training_params["discount_factor"])
        
        baseline_rewards.append(worm.total_reward)
        stage_rewards.append(worm.total_reward)
        stage_histories.append(worm.history.copy())
    
    baseline_performance = np.mean(baseline_rewards) if baseline_rewards else 0
    current_step += 10
    
    # 主要测试
    remaining_rounds = stage["rounds"] - 10
    for round_num in range(10, stage["rounds"]):
        if st.session_state.get('stop_requested', False):
            break
            
        if round_num % 50 == 0:
            yield current_step + round_num - 10, total_steps, f"零学习测试: {round_num + 1}/{stage['rounds']}", {
                'phase': 'zero_shot_testing',
                'round': round_num + 1,
                'total_rounds': stage['rounds'],
                'reward': worm.total_reward  # 添加当前奖励值
            }
        
        # 每轮重新获取起始位置，确保正确重置
        round_start_pos = get_start_position(stage["field_type"], width, height)
        reset_worm_for_new_round(worm, env, round_start_pos, width, height, stage["field_type"])
        
        for step in range(training_params["steps_per_round"]):
            # 动态温度场更新
            if stage["field_type"].startswith('dynamic'):
                if stage["field_type"] == 'dynamic_rotating_double_center':
                    temp_array, best_point = generate_dynamic_rotating_double_center(width, height, t=step)
                    env.update_temperature_array(temp_array, best_point)
                elif stage["field_type"] == 'dynamic_rotating_quad_center':
                    temp_array, best_point = generate_dynamic_rotating_quad_center(width, height, t=step)
                    env.update_temperature_array(temp_array, best_point)
                    
            moved = worm.decide_move(env, epsilon=0.01, alpha=0.0, gamma=training_params["discount_factor"])
        
        stage_histories.append(worm.history.copy())
        stage_rewards.append(worm.total_reward)
    
    # 分析结果
    final_performance = np.mean(stage_rewards[-50:]) if len(stage_rewards) >= 50 else np.mean(stage_rewards)
    improvement = final_performance - baseline_performance
    stability = np.std(stage_rewards)
    success_rate = len([r for r in stage_rewards if r > 0]) / len(stage_rewards) if stage_rewards else 0
    
    test_results = {
        "final_performance": final_performance,
        "improvement": improvement,
        "baseline_performance": baseline_performance,
        "stability": stability,
        "success_rate": success_rate
    }
    
    return stage_histories, stage_rewards, test_results


# 辅助函数 - 从 main.py 移植过来的完整功能

def save_and_visualize_results(config, all_histories, all_rewards, worm, env, training_params, temp_array, best_point):
    """保存和可视化结果 - 修复版本，恢复动画生成"""
    try:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 调试：开始保存结果到 {output_dir}")
        
        # 🔧 修复：确保 config 有必要的属性
        if not hasattr(config, 'log_file'):
            config.log_file = os.path.join(output_dir, "experiment.log")
        if not hasattr(config, 'field_type'):
            config.field_type = 'unknown'
        
        # 保存训练日志和数据
        save_training_log(config, all_histories, all_rewards, worm, env, training_params)
        save_body_metrics(config, all_histories, all_rewards, worm, env)
        if hasattr(worm, 'q_table'):
            save_q_table(config, worm.q_table, env)
        
        # 🔧 修复：使用完整的训练结果可视化，包含轨迹图
        try:
            print("📊 正在生成训练结果图表...")
            plot_training_results_2d(all_histories, all_rewards, 
                                    getattr(worm, 'q_table', None), 
                                    temp_array, best_point, config)
            print("✓ 训练结果图表生成完成")
        except Exception as e:
            print(f"❌ 生成训练结果图片失败: {e}")
            # 如果完整版失败，回退到简化版
            try:
                print("🔧 回退到简化版训练结果图片...")
                plot_training_results_simple_mac(all_rewards, config)
                print("✓ 简化版训练结果图片生成完成")
            except Exception as e2:
                print(f"⚠️ 简化版训练结果图片也失败: {e2}")
            # 尝试最基础的可视化
            try:
                plot_basic_rewards_only(all_rewards, config)
            except Exception as e2:
                print(f"⚠️ 基础图片生成也失败: {e2}")
        
        # 🎬 恢复动画生成功能！
        try:
            print("🎬 开始生成训练动画...")
            
            # 判断是否为动态环境
            if hasattr(config, 'field_type') and 'dynamic' in config.field_type:
                print(f"🔧 检测到动态环境: {config.field_type}")
                width = temp_array.shape[1] if hasattr(temp_array, 'shape') else 80
                height = temp_array.shape[0] if hasattr(temp_array, 'shape') else 80
                steps_per_round = training_params.get("steps_per_round", 100)
                
                # 生成动态环境的MP4动画
                create_training_animation_2d_dynamic_mp4(
                    all_histories, width, height, best_point, config, steps_per_round
                )
                print("✓ 动态环境MP4动画生成完成")
            else:
                print(f"🔧 检测到静态环境: {config.field_type}")
                # 生成静态环境的GIF动画
                create_training_animation_2d(all_histories, temp_array, best_point, config)
                print("✓ 静态环境GIF动画生成完成")
                
        except Exception as e:
            print(f"⚠️ 动画生成失败: {e}")
            # 如果标准动画失败，尝试简化版动画
            try:
                print("🔧 尝试生成简化版动画...")
                create_simple_animation(all_histories, temp_array, config)
                print("✓ 简化版动画生成完成")
            except Exception as e2:
                print(f"⚠️ 简化版动画也失败: {e2}")
            
        print("🎉 结果保存完成！")
            
    except Exception as e:
        print(f"❌ 保存和可视化结果失败: {e}")

def plot_training_results_simple_mac(all_rewards, config):
    """Mac适配的简化版训练结果可视化"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 设置Mac中文字体
        has_chinese = setup_chinese_font()
        
        plt.figure(figsize=(12, 8))
        
        # 根据是否有中文字体选择标题
        if has_chinese:
            titles = {
                'main': '训练奖励进展',
                'dist': '奖励分布',
                'smooth': '平滑进展',
                'stats': '训练统计摘要'
            }
            labels = {
                'episode': '轮次',
                'reward': '总奖励',
                'frequency': '频次',
                'raw': '原始奖励'
            }
        else:
            titles = {
                'main': 'Training Rewards Progress',
                'dist': 'Reward Distribution', 
                'smooth': 'Smoothed Progress',
                'stats': 'Training Statistics'
            }
            labels = {
                'episode': 'Episode',
                'reward': 'Total Reward',
                'frequency': 'Frequency',
                'raw': 'Raw Rewards'
            }
        
        # 奖励变化图
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(all_rewards) + 1), all_rewards, 'b-o', linewidth=2, markersize=6)
        plt.title(titles['main'], fontsize=14, fontweight='bold')
        plt.xlabel(labels['episode'], fontsize=12)
        plt.ylabel(labels['reward'], fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 奖励统计
        plt.subplot(2, 2, 2)
        plt.hist(all_rewards, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(titles['dist'], fontsize=14, fontweight='bold')
        plt.xlabel(labels['reward'], fontsize=12)
        plt.ylabel(labels['frequency'], fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 滑动平均
        plt.subplot(2, 2, 3)
        if len(all_rewards) >= 3:
            window_size = min(5, len(all_rewards) // 2)
            moving_avg = []
            for i in range(len(all_rewards)):
                start_idx = max(0, i - window_size + 1)
                end_idx = i + 1
                moving_avg.append(np.mean(all_rewards[start_idx:end_idx]))
            plt.plot(range(1, len(all_rewards) + 1), moving_avg, 'r-', linewidth=3, label=f'Moving Avg (window={window_size})')
            plt.plot(range(1, len(all_rewards) + 1), all_rewards, 'b-', alpha=0.3, label=labels['raw'])
            plt.legend()
        else:
            plt.plot(range(1, len(all_rewards) + 1), all_rewards, 'b-o')
        plt.title(titles['smooth'], fontsize=14, fontweight='bold')
        plt.xlabel(labels['episode'], fontsize=12)
        plt.ylabel(labels['reward'], fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 统计信息
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        # 使用英文统计文本确保显示正常
        stats_text = f"""
Training Summary

Total Rounds: {len(all_rewards)}
Average Reward: {np.mean(all_rewards):.2f}
Best Reward: {np.max(all_rewards):.2f}
Worst Reward: {np.min(all_rewards):.2f}
Std Deviation: {np.std(all_rewards):.2f}
"""
        if len(all_rewards) >= 6:
            early_avg = np.mean(all_rewards[:3])
            late_avg = np.mean(all_rewards[-3:])
            improvement = late_avg - early_avg
            stats_text += f"Improvement: {improvement:.2f}"
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        results_image = os.path.join(config.output_dir, "training_results_simple.png")
        plt.savefig(results_image, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Mac适配训练结果图已保存: {results_image}")
        
    except Exception as e:
        print(f"⚠️ Mac适配可视化失败: {e}")
        raise

def plot_basic_rewards_only(all_rewards, config):
    """最基础的奖励图表"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(all_rewards) + 1), all_rewards, 'b-o', linewidth=2, markersize=6)
        plt.title('Training Rewards', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Total Reward', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        avg_reward = np.mean(all_rewards)
        plt.axhline(y=avg_reward, color='r', linestyle='--', alpha=0.7, label=f'Average: {avg_reward:.2f}')
        plt.legend()
        
        plt.tight_layout()
        
        # 保存图片
        results_image = os.path.join(config.output_dir, "basic_rewards.png")
        plt.savefig(results_image, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 基础奖励图已保存: {results_image}")
        
    except Exception as e:
        print(f"⚠️ 基础可视化失败: {e}")
        raise

def create_simple_animation(all_histories, temp_array, config):
    """创建简化版动画 - 如果标准动画失败时的备用方案"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from PIL import Image
        import io
        
        print("🔧 开始创建简化版动画...")
        
        if not all_histories:
            print("⚠️ 没有历史数据，无法创建动画")
            return
        
        # 只取前几轮，避免文件过大
        max_rounds = min(5, len(all_histories))
        selected_histories = all_histories[:max_rounds]
        
        frames = []
        
        # 为每一轮创建一帧
        for round_idx, history in enumerate(selected_histories):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 绘制温度背景
            if temp_array is not None:
                im = ax.imshow(temp_array, cmap='hot', alpha=0.7, origin='lower')
                plt.colorbar(im, ax=ax, label='Temperature (°C)')
            
            # 绘制线虫轨迹
            if history:
                # 取最后几步的轨迹
                max_steps = min(50, len(history))
                recent_history = history[-max_steps:]
                
                for step_idx, body_segments in enumerate(recent_history):
                    alpha = 0.3 + 0.7 * (step_idx / len(recent_history))  # 越新越不透明
                    
                    if body_segments:
                        # 绘制身体段
                        x_coords = [seg[0] for seg in body_segments]
                        y_coords = [seg[1] for seg in body_segments]
                        
                        # 身体
                        ax.plot(x_coords, y_coords, 'b-', linewidth=3, alpha=alpha)
                        
                        # 头部
                        if len(body_segments) > 0:
                            head_x, head_y = body_segments[0]
                            ax.plot(head_x, head_y, 'ro', markersize=8, alpha=alpha)
            
            ax.set_title(f'线虫训练轨迹 - 第{round_idx + 1}轮', fontsize=14, fontweight='bold')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True, alpha=0.3)
            
            # 保存为图像
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            frames.append(img)
            
            plt.close(fig)
            buf.close()
        
        # 保存为GIF
        if frames:
            gif_path = os.path.join(config.output_dir, "simple_training_animation.gif")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000,  # 每帧1秒
                loop=0
            )
            print(f"✓ 简化版GIF动画已保存: {gif_path}")
        
    except Exception as e:
        print(f"❌ 简化版动画创建失败: {e}")
        raise

def generate_step_statistics_plots(config, steps_to_reach_target, all_rewards, 
                                 field_type, average_temperatures, detailed_temperature_history):
    """
    生成步数统计图表 - 新增函数
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        print("🔧 开始生成步数统计图表...")
        
        # 创建多子图布局
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'训练统计分析 - {field_type}', fontsize=16, fontweight='bold')
        
        # 1. 步数变化趋势
        ax1 = axes[0, 0]
        ax1.plot(range(1, len(steps_to_reach_target) + 1), steps_to_reach_target, 'b-o', linewidth=2, markersize=4)
        ax1.set_title('到达目标的步数变化', fontsize=12, fontweight='bold')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('步数')
        ax1.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(steps_to_reach_target) > 1:
            z = np.polyfit(range(len(steps_to_reach_target)), steps_to_reach_target, 1)
            p = np.poly1d(z)
            ax1.plot(range(1, len(steps_to_reach_target) + 1), p(range(len(steps_to_reach_target))), 
                    "r--", alpha=0.8, label=f'趋势线 (斜率: {z[0]:.2f})')
            ax1.legend()
        
        # 2. 奖励vs步数散点图
        ax2 = axes[0, 1]
        if len(all_rewards) == len(steps_to_reach_target):
            ax2.scatter(steps_to_reach_target, all_rewards, alpha=0.6, s=50)
            ax2.set_title('奖励 vs 步数关系', fontsize=12, fontweight='bold')
            ax2.set_xlabel('到达目标步数')
            ax2.set_ylabel('总奖励')
            ax2.grid(True, alpha=0.3)
            
            # 计算相关系数
            if len(steps_to_reach_target) > 1:
                correlation = np.corrcoef(steps_to_reach_target, all_rewards)[0, 1]
                ax2.text(0.05, 0.95, f'相关系数: {correlation:.3f}', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, '数据长度不匹配', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('奖励 vs 步数关系', fontsize=12, fontweight='bold')
        
        # 3. 平均温度变化
        ax3 = axes[0, 2]
        if average_temperatures:
            ax3.plot(range(1, len(average_temperatures) + 1), average_temperatures, 'g-o', linewidth=2, markersize=4)
            ax3.set_title('平均体感温度变化', fontsize=12, fontweight='bold')
            ax3.set_xlabel('轮次')
            ax3.set_ylabel('平均温度 (°C)')
            ax3.grid(True, alpha=0.3)
            
            # 添加温度区间标记
            ax3.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='优秀温度线 (80°C)')
            ax3.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='良好温度线 (60°C)')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, '无温度数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('平均体感温度变化', fontsize=12, fontweight='bold')
        
        # 4. 步数分布直方图
        ax4 = axes[1, 0]
        ax4.hist(steps_to_reach_target, bins=min(10, len(steps_to_reach_target)), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_title('步数分布', fontsize=12, fontweight='bold')
        ax4.set_xlabel('步数')
        ax4.set_ylabel('频次')
        ax4.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_steps = np.mean(steps_to_reach_target)
        std_steps = np.std(steps_to_reach_target)
        ax4.axvline(mean_steps, color='r', linestyle='--', label=f'平均: {mean_steps:.1f}')
        ax4.axvline(mean_steps + std_steps, color='orange', linestyle=':', alpha=0.7, label=f'+1σ: {mean_steps + std_steps:.1f}')
        ax4.axvline(mean_steps - std_steps, color='orange', linestyle=':', alpha=0.7, label=f'-1σ: {mean_steps - std_steps:.1f}')
        ax4.legend()
        
        # 5. 学习效率分析
        ax5 = axes[1, 1]
        if len(steps_to_reach_target) >= 3:
            # 计算滑动平均
            window_size = min(5, len(steps_to_reach_target) // 2)
            moving_avg_steps = []
            for i in range(len(steps_to_reach_target)):
                start_idx = max(0, i - window_size + 1)
                end_idx = i + 1
                moving_avg_steps.append(np.mean(steps_to_reach_target[start_idx:end_idx]))
            
            ax5.plot(range(1, len(steps_to_reach_target) + 1), steps_to_reach_target, 
                    'b-', alpha=0.3, label='原始步数')
            ax5.plot(range(1, len(moving_avg_steps) + 1), moving_avg_steps, 
                    'r-', linewidth=3, label=f'滑动平均 (窗口={window_size})')
            ax5.set_title('学习效率分析', fontsize=12, fontweight='bold')
            ax5.set_xlabel('轮次')
            ax5.set_ylabel('步数')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.plot(range(1, len(steps_to_reach_target) + 1), steps_to_reach_target, 'b-o')
            ax5.set_title('学习效率分析', fontsize=12, fontweight='bold')
            ax5.set_xlabel('轮次')
            ax5.set_ylabel('步数')
            ax5.grid(True, alpha=0.3)
        
        # 6. 综合统计信息
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # 计算各种统计指标
        total_rounds = len(steps_to_reach_target)
        min_steps = np.min(steps_to_reach_target) if steps_to_reach_target else 0
        max_steps = np.max(steps_to_reach_target) if steps_to_reach_target else 0
        mean_steps = np.mean(steps_to_reach_target) if steps_to_reach_target else 0
        median_steps = np.median(steps_to_reach_target) if steps_to_reach_target else 0
        std_steps = np.std(steps_to_reach_target) if steps_to_reach_target else 0
        
        # 学习趋势
        if len(steps_to_reach_target) > 1:
            early_avg = np.mean(steps_to_reach_target[:3]) if len(steps_to_reach_target) >= 3 else steps_to_reach_target[0]
            late_avg = np.mean(steps_to_reach_target[-3:]) if len(steps_to_reach_target) >= 3 else steps_to_reach_target[-1]
            improvement = early_avg - late_avg  # 步数减少表示改善
            improvement_pct = (improvement / early_avg * 100) if early_avg > 0 else 0
        else:
            improvement = 0
            improvement_pct = 0
        
        stats_text = f"""
综合统计摘要

总训练轮数: {total_rounds}
最少步数: {min_steps:.0f}
最多步数: {max_steps:.0f}
平均步数: {mean_steps:.1f}
中位数步数: {median_steps:.1f}
标准差: {std_steps:.1f}

学习效果:
步数改善: {improvement:.1f} 步
改善百分比: {improvement_pct:.1f}%
"""
        
        if average_temperatures:
            avg_temp = np.mean(average_temperatures)
            max_temp = np.max(average_temperatures)
            stats_text += f"""
温度表现:
平均温度: {avg_temp:.1f}°C
最高温度: {max_temp:.1f}°C
"""
        
        # 根据表现给出评价
        if improvement_pct > 20:
            evaluation = "🎉 学习效果优秀!"
        elif improvement_pct > 10:
            evaluation = "✅ 学习效果良好!"
        elif improvement_pct > 0:
            evaluation = "🤔 学习效果一般"
        else:
            evaluation = "❌ 需要调整策略"
        
        stats_text += f"\n评价: {evaluation}"
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        stats_image = os.path.join(config.output_dir, "step_statistics.png")
        plt.savefig(stats_image, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 步数统计图表已保存: {stats_image}")
        
    except Exception as e:
        print(f"⚠️ 步数统计图表生成失败: {e}")
        # 创建一个简化版本
        try:
            create_simple_step_stats(config, steps_to_reach_target, all_rewards)
        except Exception as e2:
            print(f"⚠️ 简化版步数统计也失败: {e2}")

def create_simple_step_stats(config, steps_to_reach_target, all_rewards):
    """创建简化版步数统计"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # 步数变化
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(steps_to_reach_target) + 1), steps_to_reach_target, 'b-o', linewidth=2)
        plt.title('步数变化趋势')
        plt.xlabel('轮次')
        plt.ylabel('步数')
        plt.grid(True, alpha=0.3)
        
        # 奖励变化
        plt.subplot(2, 2, 2)
        plt.plot(range(1, len(all_rewards) + 1), all_rewards, 'g-o', linewidth=2)
        plt.title('奖励变化趋势')
        plt.xlabel('轮次')
        plt.ylabel('奖励')
        plt.grid(True, alpha=0.3)
        
        # 步数分布
        plt.subplot(2, 2, 3)
        plt.hist(steps_to_reach_target, bins=10, alpha=0.7, color='skyblue')
        plt.title('步数分布')
        plt.xlabel('步数')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        
        # 统计信息
        plt.subplot(2, 2, 4)
        plt.axis('off')
        stats_text = f"""
简化统计:

总轮数: {len(steps_to_reach_target)}
平均步数: {np.mean(steps_to_reach_target):.1f}
平均奖励: {np.mean(all_rewards):.2f}
"""
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存
        simple_stats_image = os.path.join(config.output_dir, "simple_step_statistics.png")
        plt.savefig(simple_stats_image, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 简化版步数统计已保存: {simple_stats_image}")
        
    except Exception as e:
        print(f"❌ 简化版步数统计失败: {e}")

# 在 simulation_engine.py 中修复动画生成函数

def create_training_animation_2d_fixed(all_histories, temp_array, best_point, config):
    """修复版动画生成 - 解决闪烁问题"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from PIL import Image
        import numpy as np
        
        print("🎬 开始生成修复版训练动画...")
        
        if not all_histories or len(all_histories) == 0:
            print("⚠️ 没有历史数据，跳过动画生成")
            return
        
        # 🔧 修复1：稳定的图像参数
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')  # 设置背景色
        
        # 🔧 修复2：固定坐标轴范围，避免跳动
        height, width = temp_array.shape
        ax.set_xlim(0, width-1)
        ax.set_ylim(0, height-1)
        ax.set_aspect('equal')
        
        # 🔧 修复3：预先绘制温度背景，避免每帧重绘
        im_bg = ax.imshow(temp_array, cmap='hot', alpha=0.6, origin='lower', 
                         extent=[0, width-1, 0, height-1], interpolation='bilinear')
        
        # 绘制目标点
        if best_point:
            ax.plot(best_point[0], best_point[1], 'g*', markersize=20, 
                   markeredgecolor='black', markeredgewidth=2, label='Target')
        
        # 🔧 修复4：预创建绘图对象，避免重复创建
        trail_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, label='Trail')
        body_line, = ax.plot([], [], 'r-', linewidth=4, label='Body')
        head_point, = ax.plot([], [], 'ro', markersize=10, markeredgecolor='black', 
                             markeredgewidth=2, label='Head')
        
        ax.set_title('C. elegans Training Animation', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 🔧 修复5：优化数据处理，减少计算量
        max_rounds = min(10, len(all_histories))  # 限制轮数
        selected_histories = all_histories[:max_rounds]
        
        # 预处理所有帧数据
        frames_data = []
        for round_idx, history in enumerate(selected_histories):
            if not history:
                continue
                
            # 限制每轮的步数，避免动画过长
            max_steps = min(100, len(history))
            round_history = history[:max_steps]
            
            for step_idx, body_segments in enumerate(round_history):
                if body_segments and len(body_segments) > 0:
                    frames_data.append({
                        'round': round_idx,
                        'step': step_idx,
                        'body_segments': body_segments.copy(),
                        'frame_index': len(frames_data)
                    })
        
        print(f"🔧 预处理完成，总帧数: {len(frames_data)}")
        
        # 🔧 修复6：稳定的动画更新函数
        def animate(frame_idx):
            if frame_idx >= len(frames_data):
                return trail_line, body_line, head_point
                
            frame_data = frames_data[frame_idx]
            body_segments = frame_data['body_segments']
            
            if not body_segments or len(body_segments) == 0:
                return trail_line, body_line, head_point
            
            # 提取坐标
            x_coords = [seg[0] for seg in body_segments if len(seg) >= 2]
            y_coords = [seg[1] for seg in body_segments if len(seg) >= 2]
            
            if len(x_coords) == 0 or len(y_coords) == 0:
                return trail_line, body_line, head_point
            
            # 🔧 修复7：显示轨迹痕迹（最近的几步）
            trail_length = 20
            start_frame = max(0, frame_idx - trail_length)
            trail_x, trail_y = [], []
            
            for i in range(start_frame, frame_idx + 1):
                if i < len(frames_data):
                    past_segments = frames_data[i]['body_segments']
                    if past_segments and len(past_segments) > 0:
                        trail_x.append(past_segments[0][0])  # 头部位置
                        trail_y.append(past_segments[0][1])
            
            # 更新绘图对象
            trail_line.set_data(trail_x, trail_y)
            body_line.set_data(x_coords, y_coords)
            
            if len(x_coords) > 0 and len(y_coords) > 0:
                head_point.set_data([x_coords[0]], [y_coords[0]])
            
            return trail_line, body_line, head_point
        
        # 🔧 修复8：稳定的动画参数
        total_frames = len(frames_data)
        if total_frames == 0:
            print("⚠️ 没有有效的动画帧数据")
            plt.close(fig)
            return
        
        # 创建动画 - 使用较慢的帧率避免闪烁
        anim = animation.FuncAnimation(
            fig, animate, frames=total_frames,
            interval=200,  # 200ms per frame (5 FPS) - 较慢避免闪烁
            blit=True,     # 使用blitting提高性能
            repeat=True,
            repeat_delay=1000  # 循环间隔1秒
        )
        
        # 保存动画
        try:
            # 保存为MP4 (推荐)
            gif_path = os.path.join(config.output_dir, "training_animation_fixed.gif")
            anim.save(gif_path, writer='pillow', fps=5)
            print(f"✓ 修复版GIF动画已保存: {gif_path}")
            
        except Exception as save_error:
            print(f"⚠️ 动画保存失败: {save_error}")
        
        plt.close(fig)
        
    except Exception as e:
        print(f"❌ 修复版动画生成失败: {e}")
        import traceback
        traceback.print_exc()
