"""
工具函数模块 - 包含日志保存、Q表保存等辅助功能
"""
import sys
import io
from collections import deque
import json
import os
import numpy as np

# 🔧 Windows GBK编码修复
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    elif hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass


def save_training_log(config, all_histories, all_rewards, worm, env, training_params):
    """保存训练日志"""
    with open(config.log_file, 'w', encoding='utf-8') as f:
        f.write(f"秀丽隐杆线虫二维学习实验日志\n")
        f.write(f"实验名称: {config.experiment_name}\n")
        f.write(f"学习方法: {training_params.get('method', 'Q-Learning')}\n")
        f.write(f"时间戳: {config.timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.write("训练参数:\n")
        for key, value in training_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        f.write("训练结果摘要:\n")
        f.write(f"  总训练轮次: {len(all_rewards)}\n")
        f.write(f"  前3轮平均奖励: {np.mean(all_rewards[:3]):.2f}\n")
        f.write(f"  后3轮平均奖励: {np.mean(all_rewards[-3:]):.2f}\n")
        f.write(f"  最佳奖励: {max(all_rewards):.2f} (第{all_rewards.index(max(all_rewards))+1}轮)\n")
        f.write(f"  最差奖励: {min(all_rewards):.2f} (第{all_rewards.index(min(all_rewards))+1}轮)\n")
        f.write("\n")
        f.write("每轮详细奖励:\n")
        for i, reward in enumerate(all_rewards):
            f.write(f"  轮次{i+1:3d}: {reward:8.2f}\n")
        f.write("\n")
        f.write("最终位置分析:\n")
        
        # 正确处理多节段身体格式
        final_positions = []
        for history in all_histories[-5:]:
            if history and len(history) > 0:
                last_segment = history[-1]
                
                # 检查多节段身体格式
                if (isinstance(last_segment, list) and 
                    len(last_segment) > 2 and 
                    all(isinstance(seg, list) and len(seg) == 2 for seg in last_segment)):
                    
                    # 多节段身体格式：取头部位置（第一个节段）
                    head_pos = last_segment[0]
                    final_positions.append(tuple(head_pos))  # 确保转换为元组
                    
                elif isinstance(last_segment, list) and len(last_segment) == 2:
                    # 简单线段格式：检查是否为坐标对
                    if all(isinstance(coord, (int, float)) for coord in last_segment):
                        # 单个坐标对
                        final_positions.append(tuple(last_segment))
                    else:
                        # 头尾坐标对：取头部
                        head_pos = last_segment[0]
                        final_positions.append(tuple(head_pos))
                else:
                    # 其他格式：跳过
                    continue
        
        # 确保所有位置都是有效的坐标对
        valid_final_positions = []
        for pos in final_positions:
            if (isinstance(pos, (tuple, list)) and 
                len(pos) == 2 and 
                all(isinstance(coord, (int, float)) for coord in pos)):
                valid_final_positions.append(pos)
        
        final_positions = valid_final_positions
        
        if final_positions:
            final_temps = [env.get_temperature(int(pos[0]), int(pos[1])) for pos in final_positions]
            best_x, best_y = env.best_point
            final_errors = [np.sqrt((pos[0]-best_x)**2 + (pos[1]-best_y)**2) for pos in final_positions]
            f.write("  后5轮最终位置及温度:\n")
            for i, (pos, temp, error) in enumerate(zip(final_positions, final_temps, final_errors)):
                round_num = len(all_histories) - len(final_positions) + i + 1
                f.write(f"    轮次{round_num}: 位置({int(pos[0]):2d},{int(pos[1]):2d}), 温度{temp:5.1f}℃, 距离{error:5.2f}\n")
            f.write(f"  平均最终距离: {np.mean(final_errors):.2f}\n")
        else:
            f.write("  无有效的最终位置数据\n")
            
        # 身体特性分析 (兼容多种身体模型)
        f.write(f"\n身体特性:\n")
        f.write(f"  节段/采样点数: {getattr(worm, 'num_segments', len(getattr(worm, 'body_segments', [])))}\n")
        seg_dist = getattr(worm, 'segment_distance', 0.0)
        f.write(f"  节段间距: {seg_dist}\n")
        if hasattr(worm, 'body_length'):
            f.write(f"  身体总长度: {worm.body_length:.1f}格\n")
        elif hasattr(worm, 'num_segments'):
            f.write(f"  身体总长度: {(worm.num_segments - 1) * seg_dist:.1f}格\n")
        # Worm2D-specific body properties
        for attr, label in [
            ('body_flexibility', '身体柔韧性'),
            ('cuticle_stiffness', '表皮刚性'),
            ('max_bend_angle', '最大弯曲角度'),
            ('hydrostatic_pressure', '液压恢复力'),
            ('pressure_recovery_rate', '液压恢复速度'),
        ]:
            if hasattr(worm, attr):
                f.write(f"  {label}: {getattr(worm, attr)}\n")
        
        # 🔧 【标黄-待改进】肌肉收缩波系统状态记录 - 新增功能，尚待完善
        # ⚠️ TODO: 下一步将添加更详细的肌肉系统日志，包括：
        # - 每轮训练中的肌肉激活模式统计
        # - 肌肉波频率和幅度的变化趋势
        # - 背腹肌肉协调性的量化分析
        # - 肌肉疲劳和能量消耗的追踪记录
        f.write(f"\n🔧 肌肉收缩波系统状态:\n")
        f.write(f"  当前肌肉波相位: {getattr(worm, 'muscle_wave_phase', 0.0):.3f} 弧度\n")
        f.write(f"  背侧肌肉状态: {getattr(worm, 'dorsal_muscle_state', 0.0):.3f} (激活强度)\n")
        f.write(f"  腹侧肌肉状态: {getattr(worm, 'ventral_muscle_state', 0.0):.3f} (激活强度)\n")
        f.write(f"  肌肉波幅度: {getattr(worm, 'muscle_wave_amplitude', 0.0):.3f}\n")
        f.write(f"  肌肉波频率: {getattr(worm, 'muscle_wave_frequency', 0.0):.3f}\n")
        mc = getattr(worm, 'muscle_coordination', None)
        if mc is not None:
            f.write(f"  肌肉协调性: {mc:.3f}\n")

        # 增强的肌肉动力学参数 (Worm2D-specific)
        for attr, label in [
            ('base_muscle_wave_frequency', '基础肌肉波频率'),
            ('frequency_adaptation_rate', '频率适应速度'),
            ('movement_frequency_boost', '运动频率提升系数'),
            ('energy_frequency_factor', '能量频率影响系数'),
        ]:
            val = getattr(worm, attr, None)
            if val is not None:
                f.write(f"  {label}: {val:.3f}\n")

        # 计算肌肉系统健康指标
        muscle_balance = abs(getattr(worm, 'dorsal_muscle_state', 0.0) - getattr(worm, 'ventral_muscle_state', 0.0))
        muscle_activity = abs(getattr(worm, 'dorsal_muscle_state', 0.0)) + abs(getattr(worm, 'ventral_muscle_state', 0.0))
        energy_ratio = getattr(worm, 'energy', 100.0) / max(getattr(worm, 'max_energy', 100.0), 1.0)
        f.write(f"\n🔧 肌肉系统健康指标:\n")
        f.write(f"  肌肉平衡度: {muscle_balance:.3f} (越小越平衡)\n")
        f.write(f"  肌肉活跃度: {muscle_activity:.3f}\n")
        f.write(f"  当前能量比例: {energy_ratio:.3f} ({energy_ratio*100:.1f}%)\n")
        f.write("  ⚠️ 注：以上肌肉系统参数为微小改进版本，持续完善中\n")
        
        f.write(f"\n肌肉疲劳系统:\n")
        f.write(f"  当前疲劳水平: {getattr(worm, 'muscle_fatigue_level', 0.0):.3f}\n")
        f.write(f"  疲劳累积速度: {getattr(worm, 'fatigue_accumulation_rate', 0.0):.3f}\n")
        f.write(f"  疲劳恢复速度: {getattr(worm, 'fatigue_recovery_rate', 0.0):.3f}\n")
        f.write(f"  疲劳阈值: {getattr(worm, 'fatigue_threshold', 0.0):.3f}\n")
        f.write(f"  最大疲劳惩罚: {getattr(worm, 'max_fatigue_penalty', 0.0):.3f}\n")

        # 计算疲劳健康指标
        fatigue_level = getattr(worm, 'muscle_fatigue_level', 0.0)
        fatigue_threshold = getattr(worm, 'fatigue_threshold', 0.3)
        max_fatigue_penalty = getattr(worm, 'max_fatigue_penalty', 0.6)
        if fatigue_level > fatigue_threshold:
            fatigue_status = "疲劳"
            excess_fatigue = fatigue_level - fatigue_threshold
            fatigue_penalty = (excess_fatigue / max(0.01, 1.0 - fatigue_threshold)) * max_fatigue_penalty
            muscle_efficiency = 1.0 - fatigue_penalty
        else:
            fatigue_status = "正常"
            muscle_efficiency = 1.0
        
        f.write(f"  疲劳状态: {fatigue_status}\n")
        f.write(f"  当前肌肉效率: {muscle_efficiency:.3f} ({muscle_efficiency*100:.1f}%)\n")
        
        f.write(f"\n温度感应肌肉系统:\n")
        current_temp = getattr(worm, 'current_temp', None)
        if current_temp is not None:
            f.write(f"  当前体温: {current_temp:.1f}°C\n")
            f.write(f"  最适肌肉温度: {getattr(worm, 'optimal_muscle_temperature', 75.0):.1f}°C\n")
            f.write(f"  温度肌肉敏感度: {getattr(worm, 'temperature_muscle_sensitivity', 0.8):.2f}\n")
            f.write(f"  肌肉温度适应速度: {getattr(worm, 'muscle_temp_adaptation_rate', 0.1):.2f}\n")

            # 计算温度对肌肉的实际影响
            temp_diff = abs(current_temp - getattr(worm, 'optimal_muscle_temperature', 75.0))
            if temp_diff <= 5.0:
                temp_status = "最适温度"
                temp_efficiency = 1.0
            elif temp_diff <= 15.0:
                temp_status = "轻微偏差"
                temp_efficiency = 1.0 - (temp_diff - 5.0) * 0.02
            elif temp_diff <= 30.0:
                temp_status = "明显偏差"
                temp_efficiency = 0.8 - (temp_diff - 15.0) * 0.01
            else:
                temp_status = "极端温度"
                temp_efficiency = max(0.3, 0.65 - (temp_diff - 30.0) * 0.005)
            
            f.write(f"  温度适应状态: {temp_status}\n")
            f.write(f"  温度肌肉效率: {temp_efficiency:.3f} ({temp_efficiency*100:.1f}%)\n")
        else:
            f.write(f"  温度感应系统: 未初始化\n")
        
        f.write(f"\n身体弹性约束系统:\n")
        if hasattr(worm, 'segment_elasticity') or hasattr(worm, 'angular_constraint'):
            f.write(f"  节段弹性系数: {getattr(worm, 'segment_elasticity', 0.85):.2f}\n")
            f.write(f"  最大拉伸倍数: {getattr(worm, 'max_segment_stretch', 1.5):.2f}\n")
            f.write(f"  最小压缩倍数: {getattr(worm, 'min_segment_compression', 0.7):.2f}\n")
            f.write(f"  最大角度约束: {getattr(worm, 'angular_constraint', 45.0):.1f}°\n")
            f.write(f"  弹性恢复速度: {getattr(worm, 'elastic_recovery_rate', 0.12):.3f}\n")
            
            # 🔧 计算当前身体张力统计
            if hasattr(worm, 'segment_tensions'):
                avg_tension = sum(abs(t) for t in worm.segment_tensions) / len(worm.segment_tensions)
                max_tension = max(abs(t) for t in worm.segment_tensions)
                stretch_segments = sum(1 for t in worm.segment_tensions if t > 0.1)
                compress_segments = sum(1 for t in worm.segment_tensions if t < -0.1)
                
                f.write(f"  当前平均张力: {avg_tension:.3f}\n")
                f.write(f"  最大张力: {max_tension:.3f}\n")
                f.write(f"  拉伸节段数: {stretch_segments}/{len(worm.segment_tensions)}\n")
                f.write(f"  压缩节段数: {compress_segments}/{len(worm.segment_tensions)}\n")
                
                # 身体柔韧性评估
                if avg_tension < 0.2:
                    flexibility_status = "非常柔韧"
                elif avg_tension < 0.4:
                    flexibility_status = "柔韧"
                elif avg_tension < 0.6:
                    flexibility_status = "一般"
                else:
                    flexibility_status = "僵硬"
                
                f.write(f"  身体柔韧性状态: {flexibility_status}\n")
        else:
            f.write(f"  弹性约束系统: 未初始化\n")
        
    print(f"✓ 训练日志已保存到: {config.log_file}")


def _to_json_safe(value):
    """把 NumPy 类型和元组转成 JSON 可写的数据。"""
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_json_safe(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _polyline_metrics_from_segments(segments, target_segment_length, target_body_length, curvature_limit_deg):
    points = []
    for point in segments or []:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            points.append(np.array([float(point[0]), float(point[1])], dtype=float))

    segment_lengths = [
        float(np.linalg.norm(points[i + 1] - points[i]))
        for i in range(len(points) - 1)
    ]
    turn_angles = []
    for i in range(1, len(points) - 1):
        prev_vector = points[i] - points[i - 1]
        next_vector = points[i + 1] - points[i]
        prev_norm = float(np.linalg.norm(prev_vector))
        next_norm = float(np.linalg.norm(next_vector))
        if prev_norm <= 1e-9 or next_norm <= 1e-9:
            continue
        cosine = float(np.dot(prev_vector, next_vector) / (prev_norm * next_norm))
        angle = float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))
        turn_angles.append(angle)

    body_length = float(sum(segment_lengths)) if segment_lengths else 0.0
    segment_errors = [abs(length - target_segment_length) for length in segment_lengths]
    curvature_violations = sum(1 for angle in turn_angles if angle > curvature_limit_deg)

    return {
        "actual_body_length": body_length,
        "body_length_error": float(body_length - target_body_length),
        "body_length_error_abs": float(abs(body_length - target_body_length)),
        "average_segment_length": float(np.mean(segment_lengths)) if segment_lengths else 0.0,
        "mean_segment_length_error": float(np.mean(segment_errors)) if segment_errors else 0.0,
        "max_segment_length_error": float(max(segment_errors)) if segment_errors else 0.0,
        "curvature_mean_deg": float(np.mean(turn_angles)) if turn_angles else 0.0,
        "curvature_max_deg": float(max(turn_angles)) if turn_angles else 0.0,
        "curvature_violation_count": int(curvature_violations),
        "curvature_violation_rate": float(curvature_violations / len(turn_angles)) if turn_angles else 0.0,
    }


def save_body_metrics(config, all_histories, all_rewards, worm, env=None):
    """保存身体模型指标，供不同身体表征做横向比较。"""
    output_dir = getattr(config, "output_dir", ".")
    metrics_file = getattr(config, "body_metrics_file", os.path.join(output_dir, "body_metrics.json"))

    final_metrics = worm.get_metrics(env=env) if hasattr(worm, "get_metrics") else {}
    geometry = worm.get_geometry() if hasattr(worm, "get_geometry") else {}
    target_segment_length = float(final_metrics.get("target_segment_length", getattr(worm, "segment_distance", 0.0)))
    target_body_length = float(final_metrics.get("target_body_length", getattr(worm, "body_length", 0.0)))
    curvature_limit_deg = float(final_metrics.get("curvature_limit_deg", getattr(worm, "angular_constraint", 0.0)))

    round_metrics = []
    for index, history in enumerate(all_histories or []):
        if not history:
            continue
        final_segments = history[-1]
        if not (
            isinstance(final_segments, list)
            and final_segments
            and isinstance(final_segments[0], (list, tuple))
        ):
            continue
        metrics = _polyline_metrics_from_segments(
            final_segments,
            target_segment_length,
            target_body_length,
            curvature_limit_deg,
        )
        metrics["round"] = index + 1
        if index < len(all_rewards):
            metrics["reward"] = float(all_rewards[index])
        round_metrics.append(metrics)

    summary_fields = [
        "body_length_error_abs",
        "mean_segment_length_error",
        "max_segment_length_error",
        "curvature_mean_deg",
        "curvature_max_deg",
        "curvature_violation_rate",
    ]
    summary = {"round_count": len(round_metrics)}
    for field in summary_fields:
        values = [item[field] for item in round_metrics if field in item]
        if values:
            summary[f"mean_{field}"] = float(np.mean(values))
            summary[f"max_{field}"] = float(max(values))

    payload = {
        "experiment_name": getattr(config, "experiment_name", None),
        "field_type": getattr(config, "field_type", None),
        "timestamp": getattr(config, "timestamp", None),
        "model": final_metrics.get("model", geometry.get("model", "unknown")),
        "final_metrics": final_metrics,
        "geometry": geometry,
        "round_metrics": round_metrics,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(metrics_file) or ".", exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(payload), f, ensure_ascii=False, indent=2)
    print(f"身体指标已保存到: {metrics_file}")
    return metrics_file


def save_q_table(config, q_table, env):
    """保存Q表数据"""
    with open(config.q_table_file, 'w', encoding='utf-8') as f:
        f.write(f"Q表数据 - {config.experiment_name}\n")
        f.write("位置(x,y)\t温度\t上Q\t下Q\t左Q\t右Q\t偏好动作\n")
        f.write("-" * 60 + "\n")
        for y in range(env.height):
            for x in range(env.width):
                q_up, q_down, q_left, q_right = q_table[y][x]
                temp = env.get_temperature(x, y)
                q_list = [q_up, q_down, q_left, q_right]
                actions = ["上","下","左","右"]
                max_q = max(q_list)
                if q_list.count(max_q) == 1:
                    preference = actions[q_list.index(max_q)]
                else:
                    preference = "无偏好"
                f.write(f"({x:2d},{y:2d})\t{temp:6.1f}\t{q_up:8.3f}\t{q_down:8.3f}\t{q_left:8.3f}\t{q_right:8.3f}\t{preference}\n")
    print(f"Q表已保存到: {config.q_table_file}")


def setup_neural_network(worm, training_params):
    """设置神经网络组件 - 添加DuelingDQN支持"""
    # 导入 PrioritizedReplayBuffer。支持作为 core 包导入，也兼容旧的脚本式运行。
    try:
        from .neural_networks import (
            PYTORCH_AVAILABLE,
            SimpleNeuralNetwork,
            DuelingDQN,
            ExperienceReplay,
            PrioritizedReplayBuffer,
        )
    except ImportError:
        from neural_networks import (
            PYTORCH_AVAILABLE,
            SimpleNeuralNetwork,
            DuelingDQN,
            ExperienceReplay,
            PrioritizedReplayBuffer,
        )
    
    if not PYTORCH_AVAILABLE:
        return False
        
    import torch.optim as optim
    
    print("🔧 初始化持久神经网络组件...")
    worm.use_neural = True
    
    method = training_params.get("method", "Neural Network")
    
    if method == "Dueling DQN":
        NetworkClass = DuelingDQN
        network_name = "Dueling DQN"
    else:
        NetworkClass = SimpleNeuralNetwork
        network_name = "标准 DQN"
    
    worm.neural_network = NetworkClass(
        input_size=32,
        hidden_size=training_params["hidden_size"],
        output_size=4
    )
    worm.target_network = NetworkClass(
        input_size=32,
        hidden_size=training_params["hidden_size"],
        output_size=4
    )
    
    worm.optimizer = optim.Adam(
        worm.neural_network.parameters(),
        lr=training_params["neural_lr"],
        weight_decay=training_params.get("weight_decay", 5e-4)  # 改为5e-4，匹配8.22版本
    )
    # ⭐️ 使用新的优先经验回放缓冲区
    worm.experience_replay = PrioritizedReplayBuffer(capacity=50000)
    worm.target_network.load_state_dict(worm.neural_network.state_dict())
    worm.target_update_freq = 100
    worm.scheduler = optim.lr_scheduler.StepLR(worm.optimizer, step_size=1000, gamma=0.95)
    
    print(f"✓ 神经网络组件已准备就绪 - 使用 {network_name} 架构")
    print("✓ 学习将跨轮次持续积累")
    return True


def reset_worm_for_new_round(worm, env, start_pos, width, height, field_type):
    """
    为新一轮重置线虫状态，并自动预热状态缓冲区。
    核心功能：根据环境类型智能设置初始朝向。
    """
    
    worm.x, worm.y = start_pos
    x, y = start_pos

    worm.body_segments = []

    # 🔧 修改：所有迷宫类型都使用垂直排列，其他环境使用水平排列
    if field_type in ['maze_thermal_channel', 'complex_maze_channel', 'complex_maze_mirror']:
        # 对于所有迷宫类型，设置垂直朝向，让线虫准备好进入迷宫
        for i in range(worm.num_segments):
            segment_x = x
            # 身体节段在头部"后方"（Y坐标增加，因为入口在上方Y坐标小的地方）
            segment_y = max(0, min(height - 1, y + i * worm.segment_distance))
            worm.body_segments.append([int(segment_x), int(segment_y)])
            
    else:
        # 对于所有非迷宫环境，使用默认的水平排列
        for i in range(worm.num_segments):
            segment_x = max(0, min(width - 1, x - i * worm.segment_distance))
            segment_y = y
            worm.body_segments.append([int(segment_x), int(segment_y)])

    # --- 后续所有通用重置逻辑保持不变 ---
    # 同步连续中心线模型的内部状态
    if hasattr(worm, '_sync_centerline_from_public_segments'):
        worm._sync_centerline_from_public_segments()
    worm.body_segment = [worm.body_segments[0], worm.body_segments[-1]]
    if hasattr(worm, '_sync_public_state'):
        try:
            worm._sync_public_state()
        except Exception:
            pass
    worm.history = [worm.body_segments.copy()]
    worm.total_reward = 0
    
    worm.recent_temperatures.clear()
    worm.visited_positions.clear()
    worm.energy = worm.max_energy
    
    # 重置肌肉和物理状态
    worm.muscle_wave_phase = 0.0
    worm.dorsal_muscle_state = 0.0
    worm.ventral_muscle_state = 0.0
    worm.muscle_wave_frequency = worm.base_muscle_wave_frequency
    if hasattr(worm, 'segment_tensions'):
        worm.segment_tensions = [0.0] * worm.num_segments
    # ⭐️ 关键修复：将状态预热逻辑统一整合进此函数
    if hasattr(worm, 'state_buffer'):
        worm.state_buffer.clear()
        # 确保 env 对象存在才进行状态获取
        if env:
            for _ in range(4):
                state = env.get_state_vector((worm.x, worm.y), worm=worm)
                worm.state_buffer.append(state)


def create_temperature_environment(width, height, seed, field_type):
    """创建可缩放的温度环境 - 完整健壮版本"""
    
    # 基准尺寸 40x40，所有温度场基于此设计
    base_width, base_height = 40, 40
    scale_x = width / base_width
    scale_y = height / base_height
    
    def scale_position(x, y):
        """将基准坐标缩放到目标尺寸"""
        return int(x * scale_x), int(y * scale_y)
    
    def scale_radius(radius):
        """缩放半径，使用平均缩放比例"""
        return radius * (scale_x + scale_y) / 2
    
    # --- 1. 单热源环境 ---
    if field_type == 'single_center':
        temp_array = np.zeros((height, width))
        cx, cy = scale_position(20, 20)  # 基准中心位置
        max_temp = 120.0
        base_temp = 20.0
        decay_factor = scale_radius(15.0)
        
        for i in range(height):
            for j in range(width):
                distance = np.sqrt((i - cy)**2 + (j - cx)**2)
                temp_contrib = (max_temp - base_temp) * np.exp(-distance / decay_factor)
                temp_array[i, j] = base_temp + temp_contrib
        
        best_point = (cx, cy)
        print(f"✓ 单热源温度场已生成 ({width}x{height})")
        print(f"  🎯 中心点: {best_point}, 最高温度: {max_temp}°C")
        return temp_array, best_point
    
    # --- 2. 线性梯度环境 ---
    elif field_type == 'linear_gradient':
        temp_array = np.zeros((height, width))
        
        # 从两边向中间递增温度
        for i in range(height):
            for j in range(width):
                # 计算距离中线的归一化值
                distance_to_center = abs(j - width // 2) / (width // 2)
                temp_array[i, j] = 20.0 + (1.0 - distance_to_center) * 80.0  # 两边低，中间高
        
        best_point = (width - 2, height // 2)
        
        print(f"✓ 两边低中间高的线性梯度温度场已生成 ({width}x{height})")
        print(f"  🎯 最佳点: {best_point}")
        
        return temp_array, best_point
    
    # --- 🔧 新增：动态温度场类型 ---
    elif field_type == 'dynamic_rotating_double_center':
        # 对于动态温度场，返回初始状态（t=0）
        temp_array, best_point = generate_dynamic_rotating_double_center(width, height, t=0)
        print(f"✓ 双中心旋转温度场已生成 ({width}x{height})")
        print(f"  🔄 动态旋转温度场 - 初始状态")
        print(f"  🎯 初始最佳点: {best_point}")
        return temp_array, best_point
    
    elif field_type == 'dynamic_rotating_quad_center':
        # 对于动态温度场，返回初始状态（t=0）
        temp_array, best_point = generate_dynamic_rotating_quad_center(width, height, t=0)
        print(f"✓ 四中心旋转温度场已生成 ({width}x{height})")
        print(f"  🔄 动态旋转温度场 - 初始状态")
        print(f"  🎯 初始最佳点: {best_point}")
        return temp_array, best_point
    
    # --- 3. 斑点热源环境 ---
    elif field_type == 'spotty_field':
        if seed is None:
            seed = 42
        np.random.seed(seed)
        
        temp_array = np.full((height, width), 20.0)  # 初始化背景温度为 20°C
        
        # 基准斑点位置（40x40基础），确保4个斑点，间距适中
        base_spots = [
            {"center": (7.5, 7.5), "intensity": 95, "radius": 8},
            {"center": (12.5, 27.5), "intensity": 92, "radius": 12},
            {"center": (27.5, 12.5), "intensity": 98, "radius": 8},
            {"center": (32.5, 32.5), "intensity": 90, "radius": 10},
        ]
        
        # 缩放斑点到目标尺寸
        for spot in base_spots:
            cx, cy = scale_position(*spot["center"])
            intensity = spot["intensity"]
            radius = scale_radius(spot["radius"])
            
            for x in range(max(0, int(cx-radius*2)), min(width, int(cx+radius*2+1))):
                for y in range(max(0, int(cy-radius*2)), min(height, int(cy+radius*2+1))):
                    distance = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if distance <= radius:
                        # 衰减速率减半，扩大辐射范围
                        temp_contrib = intensity * np.exp(-distance / (radius * 2))
                        temp_array[y, x] = max(temp_array[y, x], temp_contrib)
        
        # 设置最佳点为某个斑点的中心
        best_point = scale_position(15, 30)
        print(f"✓ 斑点热源温度场已生成 ({width}x{height})")
        return temp_array, best_point
    
    # --- 4. 双热源环境（修改为竖直分布） ---
    elif field_type == 'dual_center':
        temp_array = np.zeros((height, width))
        
        # 🔧 修改：基准热源中心位置改为竖直分布（40x40基础）
        # 上热源：在场地上方1/4处
        base_center1 = (20, 10)  # (20, 10) - 水平居中，垂直上方
        # 下热源：在场地下方3/4处  
        base_center2 = (20, 30)  # (20, 30) - 水平居中，垂直下方
        
        center1_x, center1_y = scale_position(*base_center1)
        center2_x, center2_y = scale_position(*base_center2)
        
        max_temp1 = 120.0
        max_temp2 = 100.0
        base_temp = 20.0
        
        # 🔧 修改：为高温热源设置更快的衰减速度
        # 高温热源衰减更快，范围更小
        decay_factor1 = 6.0 * (scale_x + scale_y) / 2  # 高温热源：快衰减（原12.0改为6.0）
        decay_factor2 = 15.0 * (scale_x + scale_y) / 2  # 低温热源：保持原衰减速度
        
        for i in range(height):
            for j in range(width):
                distance1 = np.sqrt((i - center1_y)**2 + (j - center1_x)**2)
                distance2 = np.sqrt((i - center2_y)**2 + (j - center2_x)**2)
                
                # 🔧 修改：使用不同的衰减参数
                temp_contrib1 = max_temp1 * np.exp(-distance1 / decay_factor1)  # 高温热源快衰减
                temp_contrib2 = max_temp2 * np.exp(-distance2 / decay_factor2)  # 低温热源慢衰减
                
                total_temp = base_temp + temp_contrib1 + temp_contrib2
                temp_array[i, j] = min(total_temp, 120.0)
        
        max_temp_idx = np.unravel_index(np.argmax(temp_array), temp_array.shape)
        best_point = (max_temp_idx[1], max_temp_idx[0])
        
        print(f"✓ 竖直分布双中心温度场已生成 ({width}x{height})")
        print(f"  热源1中心: ({center1_x}, {center1_y}) - 最高{max_temp1}°C (上方，快衰减)")
        print(f"  热源2中心: ({center2_x}, {center2_y}) - 最高{max_temp2}°C (下方，慢衰减)")
        print(f"  衰减参数: 高温{decay_factor1:.1f} vs 低温{decay_factor2:.1f}")
        print(f"  🎯 最佳点: {best_point}")
        
        return temp_array, best_point

    # --- 5. 环形热源环境（改进版） ---
    elif field_type == 'ring_hotspot':
        # 🔧 创建从环形向外递减的温度场，而不是统一低温背景
        cx, cy = scale_position(20, 20)  # 基准中心位置
        ring_radius = scale_radius(min(40, 40) // 3)  # 基准环半径
        ring_width = scale_radius(3)  # 基准环宽度
        
        # 创建基础温度场
        temp_array = np.zeros((height, width))
        
        # 🔧 从环形开始向外递减的温度分布
        max_ring_temp = 120.0  # 环形最高温度
        center_temp = 45.0     # 环心温度
        edge_temp = 20.0       # 边缘温度
        
        for i in range(height):
            for j in range(width):
                dist_to_center = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)
                
                # 环形区域：高温
                if ring_radius - ring_width <= dist_to_center <= ring_radius + ring_width:
                    # 环形内部温度分布
                    ring_distance = abs(dist_to_center - ring_radius)
                    ring_factor = max(0, 1 - ring_distance / ring_width)
                    temp_array[i, j] = max_ring_temp * ring_factor + center_temp * (1 - ring_factor)
                
                # 环心区域：中等温度，从环形向内递减
                elif dist_to_center < ring_radius - ring_width:
                    # 环心内部，温度从环边界向中心递减
                    inner_boundary = ring_radius - ring_width
                    center_factor = dist_to_center / inner_boundary if inner_boundary > 0 else 1
                    temp_array[i, j] = center_temp + (max_ring_temp - center_temp) * center_factor * 0.6
                
                # 环外区域：从环形向外递减
                else:
                    # 环外，温度从环形向边缘递减
                    outer_boundary = ring_radius + ring_width
                    max_outer_dist = max(width, height)  # 到边缘的最大可能距离
                    
                    if dist_to_center <= outer_boundary + max_outer_dist * 0.3:
                        # 近外围区域：温和递减
                        outer_factor = (dist_to_center - outer_boundary) / (max_outer_dist * 0.3)
                        outer_factor = min(1.0, max(0.0, outer_factor))
                        temp_array[i, j] = max_ring_temp * 0.7 * (1 - outer_factor) + edge_temp * outer_factor
                    else:
                        # 远外围区域：接近边缘温度
                        temp_array[i, j] = edge_temp + np.random.normal(0, 2)  # 轻微随机波动
        
        # 🔧 添加温和的空间噪声，增加环境复杂性
        np.random.seed(44)
        noise = np.random.normal(0, 2.0, (height, width))
        
        # 应用高斯平滑噪声
        try:
            from scipy import ndimage
            smoothed_noise = ndimage.gaussian_filter(noise, sigma=1.5)
        except ImportError:
            smoothed_noise = noise.copy()
            for i in range(1, height-1):
                for j in range(1, width-1):
                    smoothed_noise[i,j] = np.mean(noise[i-1:i+2, j-1:j+2])
        
        temp_array += smoothed_noise
        temp_array = np.clip(temp_array, edge_temp, max_ring_temp + 5)
        
        max_temp_idx = np.unravel_index(np.argmax(temp_array), temp_array.shape)
        best_point = (max_temp_idx[1], max_temp_idx[0])
        
        print(f"✓ 环形热源温度场已生成 ({width}x{height})")
        print(f"  🔥 环心: ({cx}, {cy}), 半径: {ring_radius:.1f}")
        print(f"  🌡️ 温度分布: {edge_temp}°C(边缘) → {center_temp}°C(环心) → {max_ring_temp}°C(环形)")
        print(f"  🎯 最高温点: {best_point}")
        
        return temp_array, best_point
    
    # --- 6. 迷宫温度通道（改进版） ---
    elif field_type == 'maze_thermal_channel':
        # 🔧 创建从中心向四周递减的背景温度场
        cx, cy = width // 2, height // 2  # 中心点
        max_bg_temp = 35.0  # 中心最高背景温度
        min_bg_temp = 25.0  # 边缘最低背景温度
        
        # 计算每个点到中心的距离，并据此设置背景温度
        temp_array = np.zeros((height, width))
        max_dist = np.sqrt((width//2)**2 + (height//2)**2)  # 到角落的最大距离
        
        for i in range(height):
            for j in range(width):
                # 计算到中心的距离
                dist_to_center = np.sqrt((j - cx)**2 + (i - cy)**2)
                # 基于距离设置背景温度（中心高，边缘低）
                normalized_dist = min(1.0, dist_to_center / max_dist)
                base_temp = max_bg_temp - (max_bg_temp - min_bg_temp) * normalized_dist
                temp_array[i, j] = base_temp
        
        # 🔧 添加温和的背景噪声，增加环境复杂性
        np.random.seed(42)
        background_noise = np.random.normal(0, 1.5, (height, width))
        
        # 🔧 添加空间平滑
        try:
            from scipy import ndimage
            smoothed_noise = ndimage.gaussian_filter(background_noise, sigma=1.5)
        except ImportError:
            smoothed_noise = background_noise.copy()
            for i in range(1, height-1):
                for j in range(1, width-1):
                    smoothed_noise[i,j] = np.mean(background_noise[i-1:i+2, j-1:j+2])
        
        temp_array += smoothed_noise
        temp_array = np.clip(temp_array, min_bg_temp, max_bg_temp)
        
        # 基准路径点（40x40基础）
        base_waypoints = [
            (8, 40 - 8),    # (8, 32)
            (8, 40 // 2),   # (8, 20)
            (40 - 8, 40 // 2),  # (32, 20)
            (40 - 8, 8)     # (32, 8)
        ]
        
        # 缩放路径点
        waypoints = [scale_position(x, y) for x, y in base_waypoints]
        best_point = waypoints[-1]
        
        # 缩放通道宽度
        channel_width = scale_radius(8.0)
        
        def point_to_line_distance(px, py, x1, y1, x2, y2):
            A = px - x1
            B = py - y1
            C = x2 - x1
            D = y2 - y1
            
            dot = A * C + B * D
            len_sq = C * C + D * D
            
            if len_sq == 0:
                return np.sqrt(A * A + B * B)
            
            param = dot / len_sq
            
            if param < 0:
                xx, yy = x1, y1
            elif param > 1:
                xx, yy = x2, y2
            else:
                xx = x1 + param * C
                yy = y1 + param * D
            
            dx = px - xx
            dy = py - yy
            return np.sqrt(dx * dx + dy * dy)
        
        # 计算总路径长度
        total_length = 0
        for i in range(len(waypoints) - 1):
            x1, y1 = waypoints[i]
            x2, y2 = waypoints[i + 1]
            total_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # 生成温度通道
        for y in range(height):
            for x in range(width):
                min_dist = float('inf')
                closest_progress = 0.0
                current_length = 0
                
                for i in range(len(waypoints) - 1):
                    x1, y1 = waypoints[i]
                    x2, y2 = waypoints[i + 1]
                    
                    dist = point_to_line_distance(x, y, x1, y1, x2, y2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        segment_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        if segment_length > 0:
                            A = x - x1
                            B = y - y1
                            C = x2 - x1
                            D = y2 - y1
                            dot = A * C + B * D
                            param = max(0, min(1, dot / (segment_length**2)))
                            
                            segment_progress = current_length + param * segment_length
                            closest_progress = segment_progress / total_length
                    
                    current_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if min_dist < channel_width:
                    progress_temp = 50.0 + 50.0 * closest_progress
                    distance_factor = max(0.6, 1.0 - min_dist / channel_width)
                    final_temp = progress_temp * distance_factor
                    temp_array[y, x] = max(temp_array[y, x], final_temp)
        
        # 设置起点引导区域
        start_x, start_y = waypoints[0]
        guide_radius = scale_radius(8)
        for dy in range(int(-guide_radius), int(guide_radius) + 1):
            for dx in range(int(-guide_radius), int(guide_radius) + 1):
                ny, nx = int(start_y + dy), int(start_x + dx)
                if 0 <= ny < height and 0 <= nx < width:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= guide_radius:
                        next_x, next_y = waypoints[1]
                        direction_x = next_x - start_x
                        direction_y = next_y - start_y
                        rel_x, rel_y = dx, dy
                        
                        dot_product = rel_x * direction_x + rel_y * direction_y
                        direction_factor = max(0, dot_product / (dist * np.sqrt(direction_x**2 + direction_y**2) + 1))
                        distance_factor = max(0, 1 - dist / guide_radius)
                        
                        if direction_factor > 0.3:
                            guide_temp = 35 + 10 * distance_factor + 8 * direction_factor
                        else:
                            guide_temp = 32 + 5 * distance_factor
                        
                        temp_array[ny, nx] = max(temp_array[ny, nx], guide_temp)
        
        # 增强终点区域
        end_x, end_y = best_point
        end_radius = scale_radius(10)
        for dy in range(int(-end_radius), int(end_radius) + 1):
            for dx in range(int(-end_radius), int(end_radius) + 1):
                ny, nx = int(end_y + dy), int(end_x + dx)
                if 0 <= ny < height and 0 <= nx < width:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= end_radius:
                        temp_factor = max(0, 1 - dist / end_radius)
                        goal_temp_final = 85 + 25 * temp_factor
                        temp_array[ny, nx] = max(temp_array[ny, nx], goal_temp_final)
        
        temp_array = np.clip(temp_array, min_bg_temp, 115)
        
        print(f"✓ 迷宫温度通道已生成 ({width}x{height})")
        print(f"  🎯 起点: {waypoints[0]}")
        print(f"  🏆 终点: {best_point}")
        print(f"  🛤️ 通道宽度: {channel_width*2:.1f}格")
        print(f"  🌡️ 背景温度: {min_bg_temp}°C(边缘) → {max_bg_temp}°C(中心)")
        
        return temp_array, best_point
    
    # --- 7. 复杂迷宫温度通道（改进版） ---
    elif field_type == 'complex_maze_channel':
        # 🔧 创建从中心向四周递减的背景温度场
        cx, cy = width // 2, height // 2  # 中心点
        max_bg_temp = 35.0  # 中心最高背景温度
        min_bg_temp = 25.0  # 边缘最低背景温度
        
        # 计算每个点到中心的距离，并据此设置背景温度
        temp_array = np.zeros((height, width))
        max_dist = np.sqrt((width//2)**2 + (height//2)**2)  # 到角落的最大距离
        
        for i in range(height):
            for j in range(width):
                # 计算到中心的距离
                dist_to_center = np.sqrt((j - cx)**2 + (i - cy)**2)
                # 基于距离设置背景温度（中心高，边缘低）
                normalized_dist = min(1.0, dist_to_center / max_dist)
                base_temp = max_bg_temp - (max_bg_temp - min_bg_temp) * normalized_dist
                temp_array[i, j] = base_temp
        
        # 🔧 添加温和的背景噪声
        np.random.seed(43)
        background_noise = np.random.normal(0, 1.8, (height, width))
        
        # 🔧 添加空间平滑
        try:
            from scipy import ndimage
            smoothed_noise = ndimage.gaussian_filter(background_noise, sigma=1.8)
        except ImportError:
            smoothed_noise = background_noise.copy()
            for i in range(1, height-1):
                for j in range(1, width-1):
                    smoothed_noise[i,j] = np.mean(background_noise[i-1:i+2, j-1:j+2])
        
        temp_array += smoothed_noise
        temp_array = np.clip(temp_array, min_bg_temp, max_bg_temp)
        
        # 基准多转角路径点（40x40基础）
        base_waypoints = [
            (5, 40 - 5),      # (5, 35)
            (5, 40 - 15),     # (5, 25)
            (15, 40 - 15),    # (15, 25)
            (15, 40 - 20),    # (15, 20)
            (22, 40 - 20),    # (22, 20)
            (22, 40 - 12),    # (22, 12)
            (40 - 5, 40 - 12), # (35, 12)
            (40 - 5, 5)       # (35, 5)
        ]
        
        waypoints = [scale_position(x, y) for x, y in base_waypoints]
        best_point = waypoints[-1]
        channel_width = scale_radius(8.0)
        
        def point_to_line_distance(px, py, x1, y1, x2, y2):
            A = px - x1
            B = py - y1
            C = x2 - x1
            D = y2 - y1
            dot = A * C + B * D
            len_sq = C * C + D * D
            if len_sq == 0:
                return np.sqrt(A * A + B * B)
            param = dot / len_sq
            if param < 0:
                xx, yy = x1, y1
            elif param > 1:
                xx, yy = x2, y2
            else:
                xx = x1 + param * C
                yy = y1 + param * D
            dx = px - xx
            dy = py - yy
            return np.sqrt(dx * dx + dy * dy)
        
        # 计算总路径长度并生成温度通道
        total_length = sum(np.sqrt((waypoints[i+1][0] - waypoints[i][0])**2 + 
                                 (waypoints[i+1][1] - waypoints[i][1])**2) 
                         for i in range(len(waypoints) - 1))
        
        for y in range(height):
            for x in range(width):
                min_dist = float('inf')
                closest_progress = 0.0
                current_length = 0
                
                for i in range(len(waypoints) - 1):
                    x1, y1 = waypoints[i]
                    x2, y2 = waypoints[i + 1]
                    
                    dist = point_to_line_distance(x, y, x1, y1, x2, y2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        segment_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        if segment_length > 0:
                            A = x - x1
                            B = y - y1
                            C = x2 - x1
                            D = y2 - y1
                            dot = A * C + B * D
                            param = max(0, min(1, dot / (segment_length**2)))
                            segment_progress = current_length + param * segment_length
                            closest_progress = segment_progress / total_length
                    
                    current_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if min_dist < channel_width:
                    progress_temp = 50.0 + 50.0 * closest_progress
                    distance_factor = max(0.6, 1.0 - min_dist / channel_width)
                    final_temp = progress_temp * distance_factor
                    temp_array[y, x] = max(temp_array[y, x], final_temp)
        
        # 设置起点和终点区域（缩放）
        start_x, start_y = waypoints[0]
        end_x, end_y = best_point
        
        for pos, desc in [(waypoints[0], "start"), (best_point, "end")]:
            px, py = pos
            radius = scale_radius(8 if desc == "start" else 10)
            for dy in range(int(-radius), int(radius) + 1):
                for dx in range(int(-radius), int(radius) + 1):
                    ny, nx = int(py + dy), int(px + dx)
                    if 0 <= ny < height and 0 <= nx < width:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= radius:
                            if desc == "start":
                                distance_factor = max(0, 1 - dist / radius)
                                guide_temp = 35 + 10 * distance_factor
                            else:
                                temp_factor = max(0, 1 - dist / radius)
                                guide_temp = 85 + 25 * temp_factor
                            temp_array[ny, nx] = max(temp_array[ny, nx], guide_temp)
        
        temp_array = np.clip(temp_array, min_bg_temp, 115)
        
        print(f"✓ 复杂迷宫温度通道已生成 ({width}x{height})")
        print(f"  🎯 起点: {waypoints[0]}")
        print(f"  🏆 终点: {best_point}")
        print(f"  🛤️ {len(waypoints)}个转折点")
        print(f"  🌡️ 背景温度: {min_bg_temp}°C(边缘) → {max_bg_temp}°C(中心)")
        
        return temp_array, best_point
    
    # --- 🔧 添加：镜像迷宫温度通道 ---
    elif field_type == 'complex_maze_mirror':
        # 🔧 创建从中心向四周递减的背景温度场
        cx, cy = width // 2, height // 2  # 中心点
        max_bg_temp = 35.0  # 中心最高背景温度
        min_bg_temp = 25.0  # 边缘最低背景温度
        
        # 计算每个点到中心的距离，并据此设置背景温度
        temp_array = np.zeros((height, width))
        max_dist = np.sqrt((width//2)**2 + (height//2)**2)  # 到角落的最大距离
        
        for i in range(height):
            for j in range(width):
                # 计算到中心的距离
                dist_to_center = np.sqrt((j - cx)**2 + (i - cy)**2)
                # 基于距离设置背景温度（中心高，边缘低）
                normalized_dist = min(1.0, dist_to_center / max_dist)
                base_temp = max_bg_temp - (max_bg_temp - min_bg_temp) * normalized_dist
                temp_array[i, j] = base_temp
        
        # 🔧 添加温和的背景噪声
        np.random.seed(143)
        background_noise = np.random.normal(0, 1.6, (height, width))
        
        # 🔧 添加空间平滑
        try:
            from scipy import ndimage
            smoothed_noise = ndimage.gaussian_filter(background_noise, sigma=1.6)
        except ImportError:
            smoothed_noise = background_noise.copy()
            for i in range(1, height-1):
                for j in range(1, width-1):
                    smoothed_noise[i,j] = np.mean(background_noise[i-1:i+2, j-1:j+2])
        
        temp_array += smoothed_noise
        temp_array = np.clip(temp_array, min_bg_temp, max_bg_temp)
        
        # 基准镜像路径点（对原复杂迷宫进行水平镜像变换：x' = 40-x, y' = y）
        # 原始路径：(5,35) → (5,25) → (15,25) → (15,20) → (22,20) → (22,12) → (35,12) → (35,5)
        # 镜像路径：(35,35) → (35,25) → (25,25) → (25,20) → (18,20) → (18,12) → (5,12) → (5,5)
        base_waypoints = [
            (40 - 5, 40 - 5),   # (35, 35) - 镜像起点对应原始(5,35)
            (40 - 5, 40 - 15),  # (35, 25) - 镜像对应原始(5,25)
            (40 - 15, 40 - 15), # (25, 25) - 镜像对应原始(15,25)
            (40 - 15, 40 - 20), # (25, 20) - 镜像对应原始(15,20)
            (40 - 22, 40 - 20), # (18, 20) - 镜像对应原始(22,20)
            (40 - 22, 40 - 12), # (18, 12) - 镜像对应原始(22,12)
            (40 - 35, 40 - 12), # (5, 12) - 镜像对应原始(35,12)
            (40 - 35, 5)        # (5, 5) - 镜像终点对应原始(35,5)
        ]
        
        waypoints = [scale_position(x, y) for x, y in base_waypoints]
        best_point = waypoints[-1]
        channel_width = scale_radius(8.0)
        
        def point_to_line_distance(px, py, x1, y1, x2, y2):
            A = px - x1
            B = py - y1
            C = x2 - x1
            D = y2 - y1
            dot = A * C + B * D
            len_sq = C * C + D * D
            if len_sq == 0:
                return np.sqrt(A * A + B * B)
            param = dot / len_sq
            if param < 0:
                xx, yy = x1, y1
            elif param > 1:
                xx, yy = x2, y2
            else:
                xx = x1 + param * C
                yy = y1 + param * D
            dx = px - xx
            dy = py - yy
            return np.sqrt(dx * dx + dy * dy)
        
        # 计算总路径长度并生成温度通道
        total_length = sum(np.sqrt((waypoints[i+1][0] - waypoints[i][0])**2 + 
                                 (waypoints[i+1][1] - waypoints[i][1])**2) 
                         for i in range(len(waypoints) - 1))
        
        for y in range(height):
            for x in range(width):
                min_dist = float('inf')
                closest_progress = 0.0
                current_length = 0
                
                for i in range(len(waypoints) - 1):
                    x1, y1 = waypoints[i]
                    x2, y2 = waypoints[i + 1]
                    
                    dist = point_to_line_distance(x, y, x1, y1, x2, y2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        segment_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        if segment_length > 0:
                            A = x - x1
                            B = y - y1
                            C = x2 - x1
                            D = y2 - y1
                            dot = A * C + B * D
                            param = max(0, min(1, dot / (segment_length**2)))
                            segment_progress = current_length + param * segment_length
                            closest_progress = segment_progress / total_length
                    
                    current_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if min_dist < channel_width:
                    progress_temp = 50.0 + 50.0 * closest_progress
                    distance_factor = max(0.6, 1.0 - min_dist / channel_width)
                    final_temp = progress_temp * distance_factor
                    temp_array[y, x] = max(temp_array[y, x], final_temp)
        
        # 设置起点和终点区域
        start_x, start_y = waypoints[0]
        end_x, end_y = best_point
        
        for pos, desc in [(waypoints[0], "start"), (best_point, "end")]:
            px, py = pos
            radius = scale_radius(8 if desc == "start" else 10)
            for dy in range(int(-radius), int(radius) + 1):
                for dx in range(int(-radius), int(radius) + 1):
                    ny, nx = int(py + dy), int(px + dx)
                    if 0 <= ny < height and 0 <= nx < width:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= radius:
                            if desc == "start":
                                distance_factor = max(0, 1 - dist / radius)
                                guide_temp = 35 + 10 * distance_factor
                            else:
                                temp_factor = max(0, 1 - dist / radius)
                                guide_temp = 85 + 25 * temp_factor
                            temp_array[ny, nx] = max(temp_array[ny, nx], guide_temp)
        
        temp_array = np.clip(temp_array, min_bg_temp, 115)
        
        print(f"✓ 镜像复杂迷宫温度通道已生成 ({width}x{height})")
        print(f"  🎯 起点: {waypoints[0]} (右上)")
        print(f"  🏆 终点: {best_point} (左下)")
        print(f"  🛤️ {len(waypoints)}个转折点 (镜像路径)")
        print(f"  🌡️ 背景温度: {min_bg_temp}°C(边缘) → {max_bg_temp}°C(中心)")
        
        return temp_array, best_point
    
    # --- 默认情况：未知温度场类型 ---
    else:
        print(f"⚠️ 未知的温度场类型: {field_type}")
        print(f"  🔧 使用默认单热源温度场")
        # 默认创建一个简单的单热源环境
        temp_array = np.zeros((height, width))
        cx, cy = width // 2, height // 2
        max_temp = 120.0
        base_temp = 20.0
        
        for i in range(height):
            for j in range(width):
                distance = np.sqrt((i - cy)**2 + (j - cx)**2)
                temp_contrib = (max_temp - base_temp) * np.exp(-distance / 15.0)
                temp_array[i, j] = base_temp + temp_contrib
        
        best_point = (cx, cy)
        print(f"✓ 默认单热源温度场已生成 ({width}x{height})")
        return temp_array, best_point

def get_stacked_state(state_buffer):
    """从状态缓冲区获取并拼接成一个大的状态向量。"""
    if not state_buffer:
        return np.zeros(8 * 4)
    
    frames = list(state_buffer)
    while len(frames) < 4:
        frames.insert(0, frames[0])
        
    return np.concatenate(frames)


def generate_dynamic_rotating_double_center(width, height, t, omega_deg=11, temp1=120, temp2=100):
    """双中心旋转温度场，t为帧数或秒，omega_deg为每秒旋转角度，热源距离为场边长的1/2"""
    cx, cy = width // 2, height // 2
    r = min(width, height) // 4  # 距离为场边长的1/2，即半径为1/4
    omega = np.deg2rad(omega_deg)
    theta = omega * t
    hot1 = (cx + r * np.cos(theta), cy + r * np.sin(theta))
    hot2 = (cx - r * np.cos(theta), cy - r * np.sin(theta))
    y_idx, x_idx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # 🔧 修复：使用指数衰减而不是线性衰减，避免全场高温
    base_decay = min(width, height) / 15.0  # 基础衰减系数
    
    dist1 = np.sqrt((x_idx - hot1[0])**2 + (y_idx - hot1[1])**2)
    dist2 = np.sqrt((x_idx - hot2[0])**2 + (y_idx - hot2[1])**2)
    
    temp1_val = temp1 * np.exp(-dist1 / base_decay) + 20
    temp2_val = temp2 * np.exp(-dist2 / base_decay) + 20
    
    temp_array = np.maximum(temp1_val, temp2_val)
    temp_array = np.clip(temp_array, 20, 140)  # 限制温度范围
    abs_diff = np.abs(temp_array - 120)
    min_idx = np.unravel_index(np.argmin(abs_diff), temp_array.shape)
    best_point = (min_idx[1], min_idx[0])
    return temp_array, best_point


def generate_dynamic_rotating_quad_center(width, height, t, omega_deg=1, temp1=120, temp2=100, temp3=80, temp4=70):
    """四中心旋转温度场，t为帧数或秒，omega_deg为每秒旋转角度，热源间距适中，衰减适当"""
    cx, cy = width // 2, height // 2
    r = min(width, height) // 4  # 适中的热源间距
    omega = np.deg2rad(omega_deg)
    theta = omega * t
    
    # 四个热源，以正方形分布，绕中心旋转
    hot1 = (cx + r * np.cos(theta), cy + r * np.sin(theta))  # 右上
    hot2 = (cx - r * np.sin(theta), cy + r * np.cos(theta))  # 左上
    hot3 = (cx - r * np.cos(theta), cy - r * np.sin(theta))  # 左下
    hot4 = (cx + r * np.sin(theta), cy - r * np.cos(theta))  # 右下
    
    y_idx, x_idx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # 🔧 修复：使用合适的衰减系数，避免全场高温
    # 基础衰减系数根据尺寸调整
    base_decay = min(width, height) / 20.0  # 基础衰减系数
    
    # 计算四个热源的温度贡献，使用指数衰减而不是线性衰减
    dist1 = np.sqrt((x_idx - hot1[0])**2 + (y_idx - hot1[1])**2)
    dist2 = np.sqrt((x_idx - hot2[0])**2 + (y_idx - hot2[1])**2)
    dist3 = np.sqrt((x_idx - hot3[0])**2 + (y_idx - hot3[1])**2)
    dist4 = np.sqrt((x_idx - hot4[0])**2 + (y_idx - hot4[1])**2)
    
    temp1_val = temp1 * np.exp(-dist1 / base_decay) + 20
    temp2_val = temp2 * np.exp(-dist2 / (base_decay * 1.2)) + 20
    temp3_val = temp3 * np.exp(-dist3 / base_decay) + 20  
    temp4_val = temp4 * np.exp(-dist4 / (base_decay * 0.8)) + 20
    
    # 取四个热源中的最高温度
    temp_array = np.maximum.reduce([temp1_val, temp2_val, temp3_val, temp4_val])
    temp_array = np.clip(temp_array, 20, 140)  # 限制温度范围
    
    # 找到最接近最高温度的点作为最佳点
    abs_diff = np.abs(temp_array - temp1)
    min_idx = np.unravel_index(np.argmin(abs_diff), temp_array.shape)
    best_point = (min_idx[1], min_idx[0])
    
    return temp_array, best_point
