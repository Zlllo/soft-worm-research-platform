"""
环境模块 - 处理温度场和实验配置
"""

import numpy as np
import os
from pathlib import Path
import datetime

class ExperimentConfig:
    """实验配置类 - Mac适配版"""
    def __init__(self, experiment_name, parent_dir=None):
        # 原有属性...
        self.experiment_name = experiment_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder_name = f"{self.timestamp}_{experiment_name}"
        
        if parent_dir is None:
            parent_dir = os.path.join(os.path.expanduser("~"), "Desktop", "C_Elegans_Sim_Results")
        
        self.output_dir = os.path.join(parent_dir, self.folder_name)
        
        # 🔧 添加缺失的属性
        self.log_file = os.path.join(self.output_dir, "experiment.log")
        self.results_image = os.path.join(self.output_dir, "training_results.png")
        self.animation_gif = os.path.join(self.output_dir, "training_animation.gif")
        self.animation_video = os.path.join(self.output_dir, "training_animation.mp4")  # 🔧 添加这个
        self.q_table_file = os.path.join(self.output_dir, "q_table.npy")
        
        # 确保目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        

class Environment2D:
    """2D温度环境类"""
    
    def __init__(self, temp_array, best_point=None):
        """
        初始化环境
        
        Args:
            temp_array: 温度数组 (numpy array)
            best_point: 最佳温度点坐标 (x, y)
        """
        self.temp_array = np.array(temp_array, dtype=np.float32)
        self.height, self.width = self.temp_array.shape
        
        if best_point is None:
            # 自动找到最高温度点
            max_idx = np.unravel_index(np.argmax(self.temp_array), self.temp_array.shape)
            self.best_point = (max_idx[1], max_idx[0])  # (x, y)
        else:
            self.best_point = best_point
    
    def get_temperature(self, x, y):
        """
        获取指定位置的温度
        
        Args:
            x, y: 坐标位置
            
        Returns:
            float: 该位置的温度值
        """
        # 🔧 关键修复：确保坐标是整数类型，并进行边界检查
        try:
            # 将坐标转换为整数
            x_int = int(round(x))
            y_int = int(round(y))
            
            # 边界检查和处理
            if x_int < 0:
                x_int = 0
            elif x_int >= self.width:
                x_int = self.width - 1
                
            if y_int < 0:
                y_int = 0
            elif y_int >= self.height:
                y_int = self.height - 1
            
            # 返回温度值
            return float(self.temp_array[y_int, x_int])
            
        except (TypeError, ValueError) as e:
            # 如果坐标转换失败，返回最低温度
            print(f"警告：坐标转换失败 ({x}, {y}): {e}")
            return 0.0
        except IndexError as e:
            # 如果仍然有索引错误，返回最低温度
            print(f"警告：索引超出边界 ({x}, {y}): {e}")
            return 0.0
    
    def is_valid_position(self, x, y):
        """
        检查位置是否在环境范围内
        
        Args:
            x, y: 坐标位置
            
        Returns:
            bool: 位置是否有效
        """
        try:
            x_int = int(round(x))
            y_int = int(round(y))
            return 0 <= x_int < self.width and 0 <= y_int < self.height
        except (TypeError, ValueError):
            return False
    
    def get_temperature_safe(self, x, y, default_temp=0.0):
        """
        安全获取温度的方法，包含错误处理
        
        Args:
            x, y: 坐标位置
            default_temp: 默认温度值
            
        Returns:
            float: 温度值
        """
        if self.is_valid_position(x, y):
            return self.get_temperature(x, y)
        else:
            return default_temp
    
    def get_gradient(self, x, y, delta=1.0):
        """
        计算指定位置的温度梯度
        
        Args:
            x, y: 坐标位置
            delta: 梯度计算的步长
            
        Returns:
            tuple: (grad_x, grad_y) 温度梯度
        """
        try:
            # 计算x方向梯度
            temp_left = self.get_temperature_safe(x - delta, y)
            temp_right = self.get_temperature_safe(x + delta, y)
            grad_x = (temp_right - temp_left) / (2 * delta)
            
            # 计算y方向梯度
            temp_down = self.get_temperature_safe(x, y - delta)
            temp_up = self.get_temperature_safe(x, y + delta)
            grad_y = (temp_up - temp_down) / (2 * delta)
            
            return grad_x, grad_y
            
        except Exception as e:
            print(f"警告：梯度计算失败 ({x}, {y}): {e}")
            return 0.0, 0.0
    
    def get_local_average_temperature(self, x, y, radius=2):
        """
        获取指定位置周围的平均温度
        
        Args:
            x, y: 中心坐标
            radius: 采样半径
            
        Returns:
            float: 平均温度
        """
        try:
            x_int = int(round(x))
            y_int = int(round(y))
            
            temps = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    temp_x = x_int + dx
                    temp_y = y_int + dy
                    if self.is_valid_position(temp_x, temp_y):
                        temps.append(self.get_temperature(temp_x, temp_y))
            
            return np.mean(temps) if temps else 0.0
            
        except Exception as e:
            print(f"警告：局部平均温度计算失败 ({x}, {y}): {e}")
            return 0.0
    
    def get_distance_to_best(self, x, y):
        """
        计算到最佳温度点的距离
        
        Args:
            x, y: 当前坐标
            
        Returns:
            float: 到最佳点的欧几里得距离
        """
        try:
            dx = x - self.best_point[0]
            dy = y - self.best_point[1]
            return np.sqrt(dx * dx + dy * dy)
        except Exception as e:
            print(f"警告：距离计算失败 ({x}, {y}): {e}")
            return float('inf')
    
    def get_normalized_temperature(self, x, y):
        """
        获取归一化的温度值 (0-1范围)
        
        Args:
            x, y: 坐标位置
            
        Returns:
            float: 归一化温度值
        """
        try:
            temp = self.get_temperature(x, y)
            min_temp = np.min(self.temp_array)
            max_temp = np.max(self.temp_array)
            
            if max_temp == min_temp:
                return 0.5  # 避免除零
            
            return (temp - min_temp) / (max_temp - min_temp)
            
        except Exception as e:
            print(f"警告：归一化温度计算失败 ({x}, {y}): {e}")
            return 0.0
    
    def get_state_vector(self, position, worm=None):
        """
        🔧 关键修复：获取固定8维状态向量，与神经网络兼容
        
        Args:
            position: 当前位置 (x, y) 或 tuple
            worm: 线虫对象（可选）
            
        Returns:
            numpy.ndarray: 固定8维状态向量
        """
        try:
            # 解析位置并转换为整数
            if isinstance(position, tuple):
                x, y = position
            else:
                x, y = position[0], position[1]
            
            x_int, y_int = int(round(x)), int(round(y))
            current_temp = self.get_temperature(x_int, y_int)
            
            state = []
            
            # 1-4. 四个方向的温度梯度（上下左右）
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dx, dy in directions:
                nx, ny = x_int + dx, y_int + dy
                if self.is_valid_position(nx, ny):
                    neighbor_temp = self.get_temperature(nx, ny)
                    gradient = (neighbor_temp - current_temp) / 20.0  # 归一化梯度
                    state.append(gradient)
                else:
                    state.append(-2.0)  # 边界标记
        
            # 5. 当前温度（归一化）
            if current_temp != -float('inf'):
                state.append(current_temp / 120.0)  # 假设最大温度为120
            else:
                state.append(-1.0)
        
            # 6. 身体姿态与温度梯度的关系
            if worm and hasattr(worm, 'segments') and worm.segments:
                try:
                    head_pos = (worm.x, worm.y)
                    tail_pos = worm.segments[-1] if worm.segments else head_pos
                    
                    body_dx = head_pos[0] - tail_pos[0]
                    body_dy = head_pos[1] - tail_pos[1]
                    body_length = np.sqrt(body_dx**2 + body_dy**2)
                    
                    if body_length > 0:
                        gradients = state[:4]
                        valid_gradients = [g for g in gradients if g > -2.0]
                        if valid_gradients:
                            best_grad_idx = np.argmax(valid_gradients)
                            grad_direction = directions[best_grad_idx]
                            alignment = (body_dx * grad_direction[0] + body_dy * grad_direction[1]) / body_length
                            state.append(alignment)
                        else:
                            state.append(0.0)
                    else:
                        state.append(0.0)
                except Exception:
                    state.append(0.0)
            else:
                state.append(0.0)
        
            # 7. 温度变化趋势
            if worm and hasattr(worm, 'recent_temperatures') and len(worm.recent_temperatures) > 2:
                try:
                    recent_avg = np.mean(list(worm.recent_temperatures)[-3:])
                    temp_trend = (current_temp - recent_avg) / 10.0
                    state.append(temp_trend)
                except Exception:
                    state.append(0.0)
            else:
                state.append(0.0)
        
            # 8. 距离最佳点的归一化距离
            try:
                distance = self.get_distance_to_best(x_int, y_int)
                max_distance = np.sqrt(self.width**2 + self.height**2)
                normalized_distance = distance / max_distance if max_distance > 0 else 0.0
                state.append(normalized_distance)
            except Exception:
                state.append(1.0)  # 最远距离
        
            # 🔧 确保精确返回8维向量
            state_array = np.array(state[:8], dtype=np.float32)
            if len(state_array) < 8:
                padding = np.zeros(8 - len(state_array), dtype=np.float32)
                state_array = np.concatenate([state_array, padding])
        
            return state_array
            
        except Exception as e:
            print(f"警告：状态向量计算失败 ({position}): {e}")
            return np.zeros(8, dtype=np.float32)
    
    def get_environment_info(self):
        """
        获取环境信息
        
        Returns:
            dict: 环境信息字典
        """
        return {
            'width': self.width,
            'height': self.height,
            'min_temp': float(np.min(self.temp_array)),
            'max_temp': float(np.max(self.temp_array)),
            'mean_temp': float(np.mean(self.temp_array)),
            'best_point': self.best_point,
            'temp_range': float(np.max(self.temp_array) - np.min(self.temp_array))
        }
    
    def update_temperature_array(self, new_temp_array, new_best_point=None):
        """
        更新温度数组（用于动态环境）
        
        Args:
            new_temp_array: 新的温度数组
            new_best_point: 新的最佳点坐标
        """
        try:
            self.temp_array = np.array(new_temp_array, dtype=np.float32)
            self.height, self.width = self.temp_array.shape
            
            if new_best_point is not None:
                self.best_point = new_best_point
            else:
                # 重新计算最佳点
                max_idx = np.unravel_index(np.argmax(self.temp_array), self.temp_array.shape)
                self.best_point = (max_idx[1], max_idx[0])  # (x, y)
                
        except Exception as e:
            print(f"警告：温度数组更新失败: {e}")
    
    def __str__(self):
        """字符串表示"""
        info = self.get_environment_info()
        return f"Environment2D({info['width']}x{info['height']}, temp_range={info['temp_range']:.2f})"
    
    def __repr__(self):
        """详细字符串表示"""
        return self.__str__()
