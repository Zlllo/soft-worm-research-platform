"""
统计分析模块 - 用于分析训练过程中的性能指标
"""
import matplotlib
matplotlib.use('Agg')  # 关键修复：强制使用非GUI后端，防止Streamlit环境下卡死
import numpy as np
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class DualCenterTracker:
    """双热源追踪器 - 统计线虫到达高温和低温热源的次数"""
    
    def __init__(self, high_temp_center, low_temp_center, proximity_threshold=5.0):
        """
        初始化双热源追踪器
        Args:
            high_temp_center: 高温热源中心坐标 (x, y)
            low_temp_center: 低温热源中心坐标 (x, y)
            proximity_threshold: 接近距离阈值（像素距离）
        """
        self.high_temp_center = high_temp_center
        self.low_temp_center = low_temp_center
        self.proximity_threshold = proximity_threshold
        
        # 统计数据
        self.high_temp_visits = []  # 每轮是否到达高温源 (True/False)
        self.low_temp_visits = []   # 每轮是否到达低温源 (True/False)
        self.round_destinations = []  # 每轮最终目标 ('high', 'low', 'none')
        
    def track_round(self, worm):
        """
        跟踪一轮训练中线虫的热源访问情况
        Args:
            worm: 线虫对象
        Returns:
            str: 本轮最终到达的热源类型 ('high', 'low', 'none')
        """
        if not hasattr(worm, 'history') or len(worm.history) == 0:
            self.high_temp_visits.append(False)
            self.low_temp_visits.append(False)
            self.round_destinations.append('none')
            return 'none'
        
        visited_high = False
        visited_low = False
        last_destination = 'none'
        
        # 检查历史轨迹中每个位置（使用头部位置）
        for step, body_positions in enumerate(worm.history):
            if len(body_positions) > 0:
                head_pos = body_positions[0]  # 获取头部位置
                
                if isinstance(head_pos, (list, tuple)) and len(head_pos) >= 2:
                    x, y = head_pos[0], head_pos[1]
                    
                    # 计算到两个热源的距离
                    dist_to_high = np.sqrt((x - self.high_temp_center[0])**2 + 
                                         (y - self.high_temp_center[1])**2)
                    dist_to_low = np.sqrt((x - self.low_temp_center[0])**2 + 
                                        (y - self.low_temp_center[1])**2)
                    
                    # 检查是否接近高温源
                    if dist_to_high <= self.proximity_threshold:
                        visited_high = True
                        last_destination = 'high'
                    
                    # 检查是否接近低温源
                    if dist_to_low <= self.proximity_threshold:
                        visited_low = True
                        last_destination = 'low'
        
        # 记录统计结果
        self.high_temp_visits.append(visited_high)
        self.low_temp_visits.append(visited_low)
        self.round_destinations.append(last_destination)
        
        return last_destination
    
    def get_statistics(self):
        """
        获取双热源访问统计结果
        Returns:
            dict: 包含各种统计指标的字典
        """
        total_rounds = len(self.round_destinations)
        high_visits = sum(self.high_temp_visits)
        low_visits = sum(self.low_temp_visits)
        
        # 计算最终选择统计
        high_final = sum(1 for dest in self.round_destinations if dest == 'high')
        low_final = sum(1 for dest in self.round_destinations if dest == 'low')
        none_final = sum(1 for dest in self.round_destinations if dest == 'none')
        
        stats = {
            'total_rounds': total_rounds,
            'high_temp_visits': high_visits,
            'low_temp_visits': low_visits,
            'high_temp_final_choices': high_final,
            'low_temp_final_choices': low_final,
            'no_target_rounds': none_final,
            'high_visit_rate': high_visits / total_rounds if total_rounds > 0 else 0,
            'low_visit_rate': low_visits / total_rounds if total_rounds > 0 else 0,
            'high_preference_rate': high_final / total_rounds if total_rounds > 0 else 0,
            'low_preference_rate': low_final / total_rounds if total_rounds > 0 else 0,
            'destinations_per_round': self.round_destinations.copy()
        }
        
        return stats
    
    def print_summary(self):
        """打印双热源访问统计摘要"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("🔥 双热源访问统计摘要")
        print("="*60)
        print(f"总训练轮次: {stats['total_rounds']}")
        print(f"高温源访问次数: {stats['high_temp_visits']} ({stats['high_visit_rate']:.1%})")
        print(f"低温源访问次数: {stats['low_temp_visits']} ({stats['low_visit_rate']:.1%})")
        print("\n🎯 最终选择偏好:")
        print(f"  偏好高温源: {stats['high_temp_final_choices']}次 ({stats['high_preference_rate']:.1%})")
        print(f"  偏好低温源: {stats['low_temp_final_choices']}次 ({stats['low_preference_rate']:.1%})")
        print(f"  无明确偏好: {stats['no_target_rounds']}次")
        
        # 判断偏好倾向
        if stats['high_preference_rate'] > 0.6:
            print("📈 结论: 线虫显示出对高温源的明显偏好")
        elif stats['low_preference_rate'] > 0.6:
            print("📉 结论: 线虫显示出对低温源的明显偏好")
        else:
            print("⚖️  结论: 线虫在两个热源间没有明显偏好，表现均衡")
        print("="*60)


class StepTracker:
    """步数跟踪器 - 统计线虫到达最佳温度点的步数"""
    
    def __init__(self, temperature_tolerance=2.0):
        """
        初始化步数跟踪器
        Args:
            temperature_tolerance: 温度误差容忍度（摄氏度）
        """
        self.temperature_tolerance = temperature_tolerance
        self.steps_per_round = []  # 每轮到达目标的步数
        self.success_per_round = []  # 每轮是否成功到达
        
    def track_round(self, worm, env, best_point):
        """
        跟踪一轮训练中线虫到达最佳温度点的步数
        Args:
            worm: 线虫对象
            env: 环境对象  
            best_point: 最佳温度点坐标 (x, y)
        Returns:
            int: 到达目标的步数，如果未到达返回-1
        """
        if not hasattr(worm, 'history') or len(worm.history) == 0:
            self.steps_per_round.append(-1)
            self.success_per_round.append(False)
            return -1
            
        best_temp = env.temp_array[best_point[1], best_point[0]]
        target_temp_min = best_temp - self.temperature_tolerance
        
        # 检查历史轨迹中每个位置（使用头部位置，即第一个身体段）
        for step, body_positions in enumerate(worm.history):
            # body_positions是一个包含所有身体段位置的列表
            # 头部位置是第一个元素
            if len(body_positions) > 0:
                head_pos = body_positions[0]  # 获取头部位置
                
                # 确保head_pos是有效的坐标
                if isinstance(head_pos, (list, tuple)) and len(head_pos) >= 2:
                    x, y = int(head_pos[0]), int(head_pos[1])
                    
                    # 确保坐标在有效范围内
                    if 0 <= y < env.temp_array.shape[0] and 0 <= x < env.temp_array.shape[1]:
                        current_temp = env.temp_array[y, x]
                        # 检查是否到达目标温度范围
                        if current_temp >= target_temp_min:
                            self.steps_per_round.append(step + 1)  # step从0开始，所以+1
                            self.success_per_round.append(True)
                            return step + 1
        
        # 未到达目标
        self.steps_per_round.append(-1)
        self.success_per_round.append(False)
        return -1
    
    def get_statistics(self):
        """
        获取统计结果
        Returns:
            dict: 包含各种统计指标的字典
        """
        successful_steps = [s for s in self.steps_per_round if s > 0]
        total_rounds = len(self.steps_per_round)
        success_count = len(successful_steps)
        
        stats = {
            'total_rounds': total_rounds,
            'success_count': success_count,
            'success_rate': success_count / total_rounds if total_rounds > 0 else 0,
            'steps_per_round': self.steps_per_round.copy(),
            'successful_steps': successful_steps.copy(),
            'avg_steps_to_target': np.mean(successful_steps) if successful_steps else 0,
            'min_steps_to_target': min(successful_steps) if successful_steps else 0,
            'max_steps_to_target': max(successful_steps) if successful_steps else 0
        }
        
        return stats
    
    def plot_steps_curve(self, save_path=None, experiment_name="实验"):
        """
        绘制到达步数随轮次变化的曲线图
        Args:
            save_path: 保存路径，如果为None则显示图像
            experiment_name: 实验名称，用于图表标题
        """
        if not self.steps_per_round:
            print("⚠️  没有数据可以绘制")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        rounds = list(range(1, len(self.steps_per_round) + 1))
        
        # 上图：步数曲线（成功的轮次）
        successful_rounds = []
        successful_steps = []
        failed_rounds = []
        
        for i, steps in enumerate(self.steps_per_round):
            if steps > 0:
                successful_rounds.append(i + 1)
                successful_steps.append(steps)
            else:
                failed_rounds.append(i + 1)
        
        ax1.plot(successful_rounds, successful_steps, 'bo-', linewidth=2, markersize=4, 
                 label=f'成功到达 ({len(successful_steps)}次)')
        
        if failed_rounds:
            ax1.scatter(failed_rounds, [max(successful_steps) * 1.1] * len(failed_rounds) if successful_steps else [100] * len(failed_rounds), 
                       color='red', marker='x', s=50, label=f'未到达 ({len(failed_rounds)}次)')
        
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('到达目标的步数')
        ax1.set_title(f'{experiment_name} - 到达最佳温度点的步数变化')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 下图：移动平均趋势
        if len(successful_steps) > 5:
            window_size = min(10, len(successful_steps) // 3)
            if window_size >= 2:
                moving_avg = []
                moving_rounds = []
                for i in range(window_size - 1, len(successful_steps)):
                    avg = np.mean(successful_steps[i-window_size+1:i+1])
                    moving_avg.append(avg)
                    moving_rounds.append(successful_rounds[i])
                
                ax2.plot(moving_rounds, moving_avg, 'g-', linewidth=3, 
                        label=f'{window_size}轮移动平均')
                ax2.scatter(successful_rounds, successful_steps, alpha=0.3, color='blue', s=20)
                
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('平均步数')
        ax2.set_title('步数趋势分析（移动平均）')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 步数变化曲线已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_summary(self):
        """打印统计摘要"""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print("📊 到达目标步数统计摘要")
        print("="*50)
        print(f"总训练轮次: {stats['total_rounds']}")
        print(f"成功到达次数: {stats['success_count']}")
        print(f"成功率: {stats['success_rate']:.1%}")
        
        if stats['successful_steps']:
            print(f"平均到达步数: {stats['avg_steps_to_target']:.1f}")
            print(f"最少步数: {stats['min_steps_to_target']}")
            print(f"最多步数: {stats['max_steps_to_target']}")
        else:
            print("⚠️  没有成功到达目标的轮次")
        print("="*50)


def create_step_tracker(temperature_tolerance=2.0):
    """
    创建步数跟踪器实例
    Args:
        temperature_tolerance: 温度误差容忍度
    Returns:
        StepTracker: 步数跟踪器实例
    """
    return StepTracker(temperature_tolerance)


def create_dual_center_tracker(high_temp_center, low_temp_center, proximity_threshold=5.0):
    """
    创建双热源追踪器实例
    Args:
        high_temp_center: 高温热源中心坐标 (x, y)
        low_temp_center: 低温热源中心坐标 (x, y)
        proximity_threshold: 接近距离阈值
    Returns:
        DualCenterTracker: 双热源追踪器实例
    """
    return DualCenterTracker(high_temp_center, low_temp_center, proximity_threshold)
