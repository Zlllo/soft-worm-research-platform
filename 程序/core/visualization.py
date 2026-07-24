"""
可视化模块 - 包含训练结果图表和动画生成功能
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')

# 配置中文字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # macOS中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数显示问题
plt.rcParams['font.size'] = 12


def plot_training_results_2d(all_histories, all_rewards, final_q_table, temp_array, best_point, config):
    """生成训练结果图表"""
    plt.figure(figsize=(15, 12))
    plt.subplot(2,1,1)
    temp_img = plt.imshow(temp_array, cmap='coolwarm', origin='lower')
    plt.colorbar(temp_img, label='温度')
    
    # 只绘制最后5轮，只绘制最终线虫身体和轨迹
    for i, history in enumerate(all_histories[-5:]):
        color = plt.cm.viridis(i / max(1, 4))
        if not history or len(history) == 0:
            continue
        try:
            # ⭐️ 绘制头部轨迹
            traj_x = [seg[0][0] for seg in history if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], (list, tuple))]
            traj_y = [seg[0][1] for seg in history if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], (list, tuple))]
            plt.plot(traj_x, traj_y, '--', color=color, linewidth=2, alpha=0.5, label=f'轨迹{len(all_histories)-4+i}')
            
            # 只绘制最终位置的线虫身体
            if len(history) > 0:
                final_segment = history[-1]
                if (isinstance(final_segment, list) and 
                    len(final_segment) > 2 and 
                    all(isinstance(seg, list) and len(seg) == 2 for seg in final_segment)):
                    x_coords = [seg[0] for seg in final_segment]
                    y_coords = [seg[1] for seg in final_segment]
                    plt.plot(x_coords, y_coords, color=color, linewidth=6, alpha=0.9, 
                           label=f'线虫{len(all_histories)-4+i}')
                    head_pos = final_segment[0]
                    plt.scatter(head_pos[0], head_pos[1], color=color, s=120, 
                              marker='o', edgecolors='black', linewidth=2, zorder=5)
                    tail_pos = final_segment[-1]
                    plt.scatter(tail_pos[0], tail_pos[1], color=color, s=80, 
                              marker='s', edgecolors='black', linewidth=1, zorder=4)
        except Exception as e:
            print(f"❌ 绘制出错: {e}")
            continue
    
    plt.title('线虫二维最终位置及轨迹（最后5轮）')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(all_rewards, linewidth=2)
    plt.xlabel('训练轮次')
    plt.ylabel('累计奖励')
    plt.title('每轮累计奖励')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 先保存图片，再显示（注释掉show避免卡住）
    try:
        plt.savefig(config.results_image, dpi=300, bbox_inches='tight')
        print(f"✓ 训练结果图片已保存到: {config.results_image}")
    except Exception as e:
        print(f"❌ 保存图片失败: {e}")
    
    # 注释掉 plt.show()，避免在无GUI环境下卡住
    # plt.show()
    plt.close()  # 释放内存


def create_training_animation_2d(all_round_histories, temp_array, best_point, config):
    """创建训练动画"""
    # 只使用最后3轮，减少动画复杂度
    last_histories = all_round_histories[-3:] if len(all_round_histories) > 3 else all_round_histories
    start_round = len(all_round_histories) - len(last_histories) + 1
    fig, ax = plt.subplots(figsize=(12, 10))
    temp_img = ax.imshow(temp_array, cmap='coolwarm', origin='lower')
    plt.colorbar(temp_img, ax=ax, label='温度')
    
    # 存储当前线虫身体组件
    current_worm_lines = []  # 当前线虫线段（身体）
    current_worm_heads = []  # 当前线虫头部
    current_worm_tails = []  # 当前线虫尾部
    
    viridis = plt.cm.get_cmap('viridis')
    for i, history in enumerate(last_histories):
        color = viridis(i / max(1, len(last_histories)-1))
        
        # 当前线虫身体（实线，粗线）
        worm_line, = ax.plot([], [], '-', lw=8, color=color, alpha=0.9,
                           label=f'线虫{start_round+i}')
        current_worm_lines.append(worm_line)
        
        # 使用正确的scatter参数
        worm_head = ax.scatter([], [], c=[color], s=150, marker='o', 
                              edgecolors='black', linewidth=2, zorder=5)
        current_worm_heads.append(worm_head)
        
        worm_tail = ax.scatter([], [], c=[color], s=100, marker='s', 
                              edgecolors='black', linewidth=1, zorder=4)
        current_worm_tails.append(worm_tail)
    
    ax.set_xlim(-1, temp_array.shape[1])
    ax.set_ylim(-1, temp_array.shape[0])
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_title(f'线虫身体运动动画（最后{len(last_histories)}轮）\n粗线=线虫身体，圆点=头部，方块=尾部')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 限制最大帧数，避免动画过长
    max_len = min(150, max(len(h) for h in last_histories))  # 最多80帧
    
    def init():
        for line in current_worm_lines:
            line.set_data([], [])
        for head in current_worm_heads:
            head.set_offsets(np.empty((0, 2)))
        for tail in current_worm_tails:
            tail.set_offsets(np.empty((0, 2)))
        return current_worm_lines + current_worm_heads + current_worm_tails
    
    def animate(frame):
        for i, history in enumerate(last_histories):
            # 统一的数据验证
            if not history or len(history) == 0 or frame >= len(history):
                current_worm_lines[i].set_data([], [])
                current_worm_heads[i].set_offsets(np.empty((0, 2)))
                current_worm_tails[i].set_offsets(np.empty((0, 2)))
                continue
            
            current_segment = history[frame]
            
            # 检查是否为多节段身体格式
            if (isinstance(current_segment, list) and 
                len(current_segment) > 2 and 
                all(isinstance(seg, list) and len(seg) == 2 for seg in current_segment)):
                
                # 多节段身体：绘制完整的身体曲线
                try:
                    x_coords = [seg[0] for seg in current_segment]
                    y_coords = [seg[1] for seg in current_segment]
                    
                    # 绘制身体曲线（连接所有6个节段）
                    current_worm_lines[i].set_data(x_coords, y_coords)
                    
                    # 绘制头部（第一个节段）
                    head_pos = current_segment[0]
                    current_worm_heads[i].set_offsets([head_pos])
                    
                    # 绘制尾部（最后一个节段）
                    tail_pos = current_segment[-1]
                    current_worm_tails[i].set_offsets([tail_pos])
                except (ValueError, TypeError, IndexError):
                    # 数据转换失败时清空显示
                    current_worm_lines[i].set_data([], [])
                    current_worm_heads[i].set_offsets(np.empty((0, 2)))
                    current_worm_tails[i].set_offsets(np.empty((0, 2)))
                
            elif isinstance(current_segment, list) and len(current_segment) == 2:
                # 简单线段格式：向后兼容
                try:
                    head_pos, tail_pos = current_segment[0], current_segment[1]
                    current_worm_lines[i].set_data([head_pos[0], tail_pos[0]], 
                                                 [head_pos[1], tail_pos[1]])
                    current_worm_heads[i].set_offsets([head_pos])
                    current_worm_tails[i].set_offsets([tail_pos])
                except (ValueError, TypeError, IndexError):
                    # 数据转换失败时清空显示
                    current_worm_lines[i].set_data([], [])
                    current_worm_heads[i].set_offsets(np.empty((0, 2)))
                    current_worm_tails[i].set_offsets(np.empty((0, 2)))
                    
            else:
                # 其他格式：清空显示
                current_worm_lines[i].set_data([], [])
                current_worm_heads[i].set_offsets(np.empty((0, 2)))
                current_worm_tails[i].set_offsets(np.empty((0, 2)))
        
        return current_worm_lines + current_worm_heads + current_worm_tails
    
    # 创建动画
    anim = FuncAnimation(fig, animate, frames=max_len + 5, init_func=init, 
                        blit=True, interval=200, repeat=True)
    
    # 保存动画 — 使用 Pillow 输出 GIF (无需 FFmpeg)
    try:
        print(f"正在保存动画到: {config.animation_gif}")
        anim.save(config.animation_gif, writer='pillow', fps=4, dpi=120)
        print(f"✓ 训练动画已保存到: {config.animation_gif}")
    except Exception as e:
        print(f"⚠️ Pillow 保存失败 ({e})，尝试 ffmpeg...")
        try:
            anim.save(config.animation_video, writer='ffmpeg', fps=4, dpi=120, bitrate=2000)
            print(f"✓ MP4 动画已保存到: {config.animation_video}")
        except Exception as e2:
            print(f"❌ 无法保存动画: pillow={e}, ffmpeg={e2}")
            raise
    
    # 注释掉 plt.show()，避免在无GUI环境下卡住
    # plt.show()
    plt.close()  # 释放内存
    return anim  # 返回动画对象，防止被垃圾回收


# 🔧 第十四步：添加温度-肌肉状态的综合可视化
def get_temperature_muscle_color(worm):
    """根据温度和肌肉状态返回颜色信息"""
    if not hasattr(worm, 'current_temp') or not hasattr(worm, 'muscle_fatigue_level'):
        return 'blue', 1.0  # 默认颜色和透明度
    
    # 温度效应颜色
    temp_diff = abs(worm.current_temp - getattr(worm, 'optimal_muscle_temperature', 75.0))
    if temp_diff <= 5.0:
        temp_color = 'green'  # 最适温度
    elif temp_diff <= 15.0:
        temp_color = 'yellow'  # 轻微偏差
    elif temp_diff <= 30.0:
        temp_color = 'orange'  # 明显偏差
    else:
        temp_color = 'red'    # 极端温度
    
    # 疲劳效应透明度
    fatigue_alpha = max(0.4, 1.0 - worm.muscle_fatigue_level * 0.4)
    
    return temp_color, fatigue_alpha

def get_muscle_activity_intensity(worm):
    """计算肌肉活跃度强度用于可视化"""
    if not hasattr(worm, 'dorsal_muscle_state'):
        return 1.0
    
    muscle_activity = (abs(getattr(worm, 'dorsal_muscle_state', 0)) + 
                     abs(getattr(worm, 'ventral_muscle_state', 0))) / 2.0
    return max(0.3, min(1.0, muscle_activity * 2.0))  # 放大显示效果

# 🔧 第十五步：添加身体张力状态可视化
def get_segment_tension_color(worm, segment_index):
    """根据节段张力返回颜色强度"""
    if not hasattr(worm, 'segment_tensions') or segment_index >= len(worm.segment_tensions):
        return 'blue', 1.0  # 默认颜色
    
    tension = worm.segment_tensions[segment_index]
    
    if abs(tension) < 0.1:
        return 'green', 1.0      # 放松状态，绿色
    elif tension > 0.1:
        # 拉伸状态，红色，强度根据张力
        intensity = min(1.0, abs(tension))
        return 'red', 0.6 + 0.4 * intensity
    else:
        # 压缩状态，蓝色，强度根据张力
        intensity = min(1.0, abs(tension))
        return 'blue', 0.6 + 0.4 * intensity

def get_body_flexibility_indicator(worm):
    """计算身体整体柔韧性指标"""
    if not hasattr(worm, 'segment_tensions'):
        return 1.0
    
    avg_tension = sum(abs(t) for t in worm.segment_tensions) / len(worm.segment_tensions)
    flexibility = max(0.3, 1.0 - avg_tension)  # 张力越大，柔韧性越低
    return flexibility

def create_training_animation_2d_dynamic_mp4(all_round_histories, width, height, best_point, config, steps_per_round):
    """动态温度场动画，每帧温度场随时间变化，保存为mp4"""
    try:
        from .utils import generate_dynamic_rotating_double_center, generate_dynamic_rotating_quad_center
    except ImportError:
        from utils import generate_dynamic_rotating_double_center, generate_dynamic_rotating_quad_center
    # 🔧 根据温度场类型选择正确的生成函数
    if hasattr(config, 'field_type') and config.field_type == 'dynamic_rotating_quad_center':
        generate_temp_field = generate_dynamic_rotating_quad_center
        field_name = "四中心旋转"
    else:
        generate_temp_field = generate_dynamic_rotating_double_center
        field_name = "双中心旋转"
    
    last_histories = all_round_histories[-3:] if len(all_round_histories) > 3 else all_round_histories
    start_round = len(all_round_histories) - len(last_histories) + 1
    fig, ax = plt.subplots(figsize=(10, 10))
    temp_array, _ = generate_temp_field(width, height, t=0)
    temp_img = ax.imshow(temp_array, cmap='coolwarm', origin='lower', vmin=20, vmax=100)
    plt.colorbar(temp_img, ax=ax, label='温度')
    current_worm_lines = []
    current_worm_heads = []
    current_worm_tails = []
    viridis = plt.cm.get_cmap('viridis')
    for i, history in enumerate(last_histories):
        color = viridis(i / max(1, len(last_histories)-1))
        # 🔧 修复：保持与静态动画一致的线条设置
        worm_line, = ax.plot([], [], '-', lw=9, color=color, alpha=0.95, 
                           label=f'线虫{start_round+i}', solid_capstyle='round')
        current_worm_lines.append(worm_line)
        # 🔧 修复：保持与静态动画一致的scatter设置
        worm_head = ax.scatter([], [], c=[color], s=160, marker='o', edgecolors='black', linewidth=2.5, zorder=5)
        current_worm_heads.append(worm_head)
        worm_tail = ax.scatter([], [], c=[color], s=110, marker='s', edgecolors='black', linewidth=1.5, zorder=4)
        current_worm_tails.append(worm_tail)
    ax.set_xlim(-1, width)
    ax.set_ylim(-1, height)
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_title(f'线虫{field_name}温度场动画（最后{len(last_histories)}轮）')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    max_len = min(150, max(len(h) for h in last_histories))
    def init():
        for line in current_worm_lines:
            line.set_data([], [])
        for head in current_worm_heads:
            head.set_offsets(np.empty((0, 2)))
        for tail in current_worm_tails:
            tail.set_offsets(np.empty((0, 2)))
        temp_array, _ = generate_temp_field(width, height, t=0)
        temp_img.set_data(temp_array)
        return current_worm_lines + current_worm_heads + current_worm_tails + [temp_img]
    def animate(frame):
        temp_array, _ = generate_temp_field(width, height, t=frame)
        temp_img.set_data(temp_array)
        for i, history in enumerate(last_histories):
            if not history or len(history) == 0:
                continue
            current_frame = min(frame, len(history) - 1)
            if current_frame >= 0:
                current_segment = history[current_frame]
                if (isinstance(current_segment, list) and len(current_segment) > 2 and all(isinstance(seg, list) and len(seg) == 2 for seg in current_segment)):
                    x_coords = [seg[0] for seg in current_segment]
                    y_coords = [seg[1] for seg in current_segment]
                    current_worm_lines[i].set_data(x_coords, y_coords)
                    head_pos = current_segment[0]
                    current_worm_heads[i].set_offsets([head_pos])
                    tail_pos = current_segment[-1]
                    current_worm_tails[i].set_offsets([tail_pos])
                elif isinstance(current_segment, list) and len(current_segment) == 2:
                    head_pos, tail_pos = current_segment[0], current_segment[1]
                    current_worm_lines[i].set_data([head_pos[0], tail_pos[0]], [head_pos[1], tail_pos[1]])
                    current_worm_heads[i].set_offsets([head_pos])
                    current_worm_tails[i].set_offsets([tail_pos])
                else:
                    current_worm_lines[i].set_data([], [])
                    current_worm_heads[i].set_offsets(np.empty((0, 2)))
                    current_worm_tails[i].set_offsets(np.empty((0, 2)))
            else:
                current_worm_lines[i].set_data([], [])
                current_worm_heads[i].set_offsets(np.empty((0, 2)))
                current_worm_tails[i].set_offsets(np.empty((0, 2)))
        return current_worm_lines + current_worm_heads + current_worm_tails + [temp_img]
    # 🔧 修复：减少interval提高流畅度，提高fps减少掉帧
    anim = FuncAnimation(fig, animate, frames=max_len + 5, init_func=init, blit=True, interval=150, repeat=True)
    try:
        print(f"正在保存动态温度场动画为GIF: {config.animation_gif}")
        anim.save(config.animation_gif, writer='pillow', fps=7, dpi=120)
        print(f"✓ 动态温度场GIF已保存到: {config.animation_gif}")
    except Exception as e:
        print(f"⚠️ Pillow 保存失败 ({e})，尝试 ffmpeg...")
        try:
            anim.save(config.animation_video, writer='ffmpeg', fps=7, dpi=120, bitrate=3000)
            print(f"✓ 动态温度场MP4已保存到: {config.animation_video}")
        except Exception as e2:
            print(f"❌ 无法保存动画: pillow={e}, ffmpeg={e2}")
            raise
    plt.close()
    return anim
