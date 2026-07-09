# 软体仿生线虫仿真平台

本项目是一个基于 Python 的秀丽隐杆线虫二维趋温仿真平台。当前版本包含温度场、Q-Learning、DQN、Dueling DQN、迁移学习、课程学习和 Streamlit 可视化界面。

## 环境要求

- Python 3.8 到 3.11
- 核心测试最低依赖：`numpy`
- 完整界面依赖：`streamlit`、`matplotlib`、`Pillow`
- 深度学习依赖：`torch`
- 动画导出建议安装 `ffmpeg`

## 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

如果只想先检查核心仿真，不需要安装完整依赖；当前核心冒烟测试只依赖 `numpy`。

## 运行核心冒烟测试

```bash
python3 程序/test_simple.py
```

测试覆盖：

- 单热源温度场生成
- 温度读取和 8 维状态向量
- `Worm2D` 创建
- 环境感知
- 单步移动
- 2 轮 x 10 步 Q-Learning 短训练

## 启动可视化界面

```bash
streamlit run 程序/app.py
```

界面支持标准训练、迁移学习和课程学习。使用 DQN 或 Dueling DQN 前，请确认 `torch` 已安装。

## 当前第一阶段状态

- `程序/test_simple.py` 已与当前 `Worm2D` 构造函数对齐。
- 仓库根目录已提供 `requirements.txt`。
- 核心测试不依赖 Streamlit 和 Matplotlib，方便先验证仿真逻辑。
- 后续身体表征、连续控制和群体交互还未在本阶段实现。
