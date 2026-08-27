"""
Streamlit Web 应用 - 秀丽隐杆线虫仿真系统
支持标准训练、迁移学习、课程学习三种模式
与 simulation_engine.py 和 core 文件夹完全兼容
采用白色背景的蓝灰渐变主题，支持身体参数和噪声调节
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

# --- 修复白色文字可见性的CSS样式 ---
st.markdown("""
<style>
    /* Streamlit 全局背景设置为白色 */
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* 主容器背景和文字颜色 */
    .main .block-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #6c757d !important;  /* 普通文字保持灰色 */
    }
    
    /* 全局背景和字体 - 白色背景，普通文字灰色 */
    .main {
        background-color: #ffffff !important;
        color: #6c757d !important;  /* 普通文字保持灰色 */
    }
    
    /* 普通文字保持灰色 */
    .stApp {
        color: #6c757d !important;  /* 普通文字保持灰色 */
    }
    
    /* 确保普通文本都是灰色 */
    .stMarkdown, .stText, p, div, span {
        color: #6c757d !important;  /* 普通文字保持灰色 */
    }
    
    /* 🔵 主要标题改为蓝色系 - 不同深度的蓝色 */
    h1 {
        color: #0d47a1 !important;  /* 深蓝色 - 最重要的标题 */
    }
    
    h2 {
        color: #1565c0 !important;  /* 中深蓝色 - 次级标题 */
    }
    
    h3 {
        color: #1976d2 !important;  /* 标准蓝色 - 三级标题 */
    }
    
    h4 {
        color: #1e88e5 !important;  /* 中蓝色 - 四级标题 */
    }
    
    h5 {
        color: #2196f3 !important;  /* 亮蓝色 - 五级标题 */
    }
    
    h6 {
        color: #42a5f5 !important;  /* 浅蓝色 - 六级标题 */
    }
    
    /* 侧边栏样式 - 保持蓝灰渐变背景 */
    .css-1d391kg, .css-1v3fvcr, section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, 
            #f8faff 0%,    /* 浅蓝白 */
            #f0f4f8 25%,   /* 浅灰蓝 */
            #e8f2ff 50%,   /* 中蓝白 */
            #e0f0f7 75%,   /* 蓝灰 */
            #d8ecf4 100%   /* 深蓝灰 */
        ) !important;
        border-right: 3px solid #b3d9ff;
        width: 280px !important;
        min-width: 280px !important;
        box-shadow: 2px 0 8px rgba(33, 150, 243, 0.1);
    }
    
    /* 侧边栏文字颜色 - 普通文字保持灰色 */
    .css-1d391kg, .css-1lcbmhc, .css-1y4p8pa {
        color: #6c757d !important;  /* 普通文字保持灰色 */
    }
    
    /* 侧边栏标题改为蓝色 */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, 
    .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: #1976d2 !important;  /* 侧边栏标题用标准蓝色 */
    }
    
    /* 侧边栏内容背景 - 保持灰色主题 */
    .css-1d391kg .css-1v3fvcr {
        background: rgba(255, 255, 255, 0.7) !important;
        border-radius: 8px;
        border: 1px solid rgba(108, 117, 125, 0.15);  /* 边框保持灰色 */
        margin: 0.5rem;
        backdrop-filter: blur(5px);
    }
    
    /* 🔵 主标题样式 - 改为深蓝色渐变 */
    .main-header {
        font-size: 2.2rem;
        color: #0d47a1 !important;  /* 深蓝色 */
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(13, 71, 161, 0.3);  /* 深蓝色阴影 */
        font-weight: 700;
        background: linear-gradient(45deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%);  /* 蓝色渐变 */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* 🔵 副标题样式 - 改为中蓝色 */
    .sub-header {
        color: #1976d2 !important;  /* 标准蓝色 */
        text-align: center;
        font-size: 1rem;
        margin-bottom: 1rem;
        opacity: 0.9;
    }
    
    /* 状态框样式 - 保持灰色背景，但标题改为蓝色 */
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #868e96;  /* 保持灰色边框 */
        background: linear-gradient(135deg, rgba(108, 117, 125, 0.1) 0%, rgba(134, 142, 150, 0.05) 100%);  /* 保持灰色背景 */
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(108, 117, 125, 0.15);  /* 保持灰色阴影 */
        color: #6c757d !important;  /* 普通文字保持灰色 */
    }
    
    .status-box h4 {
        color: #1976d2 !important;  /* 状态框标题改为蓝色 */
    }
    
    .status-box p {
        color: #6c757d !important;  /* 普通文字保持灰色 */
    }
    
    /* 错误框样式 - 保持红色 */
    .error-box {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #dc3545;
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(189, 33, 48, 0.05) 100%);
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(220, 53, 69, 0.15);
        color: #dc3545 !important;
    }
    
    .error-box * {
        color: #dc3545 !important;
    }
    
    /* 成功框样式 - 保持绿色 */
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #28a745;
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(32, 134, 55, 0.05) 100%);
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(40, 167, 69, 0.15);
        color: #28a745 !important;
    }
    
    .success-box * {
        color: #28a745 !important;
    }
    
    /* 信息框样式 - 保持灰色背景，但标题改为蓝色 */
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #6c757d;  /* 保持灰色边框 */
        background: linear-gradient(135deg, rgba(108, 117, 125, 0.1) 0%, rgba(134, 142, 150, 0.05) 100%);  /* 保持灰色背景 */
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(108, 117, 125, 0.15);  /* 保持灰色阴影 */
        color: #6c757d !important;  /* 普通文字保持灰色 */
    }
    
    .info-box h3, .info-box h4 {
        color: #1976d2 !important;  /* 信息框标题改为蓝色 */
    }
    
    .info-box p, .info-box li {
        color: #6c757d !important;  /* 普通文字保持灰色 */
    }
    
    /* 警告框样式 - 保持橙色 */
    .warning-box {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #fd7e14;
        background: linear-gradient(135deg, rgba(253, 126, 20, 0.1) 0%, rgba(230, 113, 18, 0.05) 100%);
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(253, 126, 20, 0.15);
        color: #fd7e14 !important;
    }
    
    .warning-box * {
        color: #fd7e14 !important;
    }
    
    /* 卡片样式 - 保持灰色背景，但标题改为蓝色 */
    .feature-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.95) 0%, rgba(248,249,250,1) 100%);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(108, 117, 125, 0.2);  /* 保持灰色边框 */
        box-shadow: 0 4px 15px rgba(108, 117, 125, 0.1);  /* 保持灰色阴影 */
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card h4 {
        color: #1976d2 !important;  /* 卡片标题改为蓝色 */
    }
    
    .feature-card p, .feature-card li {
        color: #6c757d !important;  /* 普通文字保持灰色 */
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(108, 117, 125, 0.2);  /* 保持灰色悬停阴影 */
        border-color: #1976d2;  /* 悬停时边框改为蓝色 */
    }
    
    /* 指标卡片样式 - 保持灰色背景，但标题改为白色以确保对比度 */
    .metric-card {
        background: linear-gradient(135deg, rgba(108, 117, 125, 0.9) 0%, rgba(134, 142, 150, 0.9) 100%);  /* 保持灰色渐变 */
        border-radius: 10px;
        padding: 1rem;
        margin: 0.25rem;
        border: 1px solid rgba(108, 117, 125, 0.3);  /* 保持灰色边框 */
        box-shadow: 0 2px 10px rgba(108, 117, 125, 0.2);  /* 保持灰色阴影 */
        text-align: center;
        color: #ffffff !important;  /* 保持白色文字以确保对比度 */
    }
    
    .metric-card h4, .metric-card p {
        color: #ffffff !important;  /* 保持白色文字 */
        margin: 0;
    }

    /* 紧凑型参数卡片 - 保持灰色背景，但标题改为蓝色 */
    .compact-card {
        background: rgba(248, 249, 250, 0.9);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border: 1px solid rgba(108, 117, 125, 0.2);  /* 保持灰色边框 */
        box-shadow: 0 2px 8px rgba(108, 117, 125, 0.1);  /* 保持灰色阴影 */
        color: #6c757d !important;  /* 普通文字保持灰色 */
    }
    
    .compact-card strong {
        color: #1976d2 !important;  /* 强调文字改为蓝色 */
    }
    
    /* 进度条 - 改为蓝色 */
    .stProgress > div > div > div {
        background-color: #1976d2 !important;  /* 蓝色进度条 */
    }
    
    /* 按钮样式 - 保持灰色主题 */
    .stButton > button {
        background-color: #6c757d !important;  /* 保持灰色按钮背景 */
        color: #ffffff !important;  /* 白色按钮文字 */
        border: 1px solid #6c757d !important;  /* 保持灰色边框 */
    }
    
    .stButton > button:hover {
        background-color: #5a6268 !important;  /* 深灰色悬停 */
        border: 1px solid #5a6268 !important;
    }
    
    /* 主要按钮样式 - 改为蓝色背景黑色文字 */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"],
    .stButton button[kind="primary"],
    button[kind="primary"] {
        background-color: #1976d2 !important;  /* 蓝色主按钮 */
        color: #000000 !important;  /* 黑色按钮文字 */
        border: 1px solid #1976d2 !important;
        font-weight: bold !important;  /* 加粗字体 */
        font-size: 16px !important;  /* 增大字号 */
    }
    
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover,
    .stButton button[kind="primary"]:hover,
    button[kind="primary"]:hover {
        background-color: #1565c0 !important;  /* 深蓝色悬停 */
        color: #000000 !important;  /* 悬停时保持黑色文字 */
        border: 1px solid #1565c0 !important;
    }
    
    /* 次要按钮样式 - 保持灰色 */
    .stButton > button[kind="secondary"] {
        background-color: #adb5bd !important;  /* 保持浅灰色次按钮 */
        color: #495057 !important;  /* 深灰色文字 */
        border: 1px solid #adb5bd !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: #95a5a6 !important;
        border: 1px solid #95a5a6 !important;
    }
    
    /* 输入框和选择框 - 保持灰色主题 */
    .stSelectbox > div > div {
        border: 1px solid #ced4da !important;  /* 保持浅灰色边框 */
    }
    
    .stNumberInput > div > div > input {
        border: 1px solid #ced4da !important;  /* 保持浅灰色边框 */
        color: #6c757d !important;  /* 保持灰色文字 */
    }
    
    .stTextInput > div > div > input {
        border: 1px solid #ced4da !important;  /* 保持浅灰色边框 */
        color: #6c757d !important;  /* 保持灰色文字 */
    }
    
    /* 滑块样式 - 改为浅蓝色 */
    .stSlider > div > div > div > div {
        background-color: #dee2e6 !important;  /* 保持浅灰色滑轨 */
    }
    
    .stSlider > div > div > div > div > div {
        background-color: #64b5f6 !important;  /* 浅蓝色滑块手柄 */
    }
    
    /* 标签页样式 - 改为蓝色主题 */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa !important;  /* 保持浅灰色标签页背景 */
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #6c757d !important;  /* 保持灰色标签页文字 */
        border-bottom: 2px solid transparent !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #1976d2 !important;  /* 蓝色激活标签页 */
        border-bottom: 2px solid #1976d2 !important;  /* 蓝色下划线 */
    }
    
    /* 展开器样式 - 保持灰色背景，标题改为蓝色 */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;  /* 保持浅灰色展开器头部 */
        color: #1976d2 !important;  /* 蓝色文字 */
        border: 1px solid #dee2e6 !important;  /* 保持浅灰色边框 */
    }
    
    /* 切换开关样式 - 改为蓝色 */
    .stCheckbox > label > div {
        background-color: #dee2e6 !important;  /* 保持浅灰色背景 */
    }
    
    .stCheckbox > label > div[data-checked="true"] {
        background-color: #1976d2 !important;  /* 蓝色激活状态 */
    }
    
    /* 单选按钮样式 - 改为蓝色 */
    .stRadio > label > div {
        color: #6c757d !important;  /* 保持灰色文字 */
    }
    
    .stRadio > label > div[data-checked="true"] {
        color: #1976d2 !important;  /* 蓝色选中状态 */
    }
    
    /* 代码块样式 - 保持灰色主题 */
    .stCode {
        background-color: #f8f9fa !important;  /* 保持浅灰色背景 */
        border: 1px solid #dee2e6 !important;  /* 保持浅灰色边框 */
        color: #6c757d !important;  /* 保持灰色文字 */
    }
    
    /* 页脚样式 - 标题改为蓝色 */
    .footer-style {
        text-align: center; 
        color: #6c757d !important;  /* 普通文字保持灰色 */
        padding: 1rem 0;
    }
    
    .footer-style h5 {
        color: #1976d2 !important;  /* 页脚标题改为蓝色 */
    }
    
    .footer-style p {
        color: #868e96 !important;  /* 副标题保持浅灰色 */
    }
    
    /* 🔧 修改选择框内的文字为黑色 */
    .stSelectbox label {
        color: #000000 !important;  /* 选择框标签改为黑色 */
    }
    
    .stSelectbox > div > div {
        border: 1px solid #6c8dbf !important;  /* 框框改为蓝灰色边框 */
        background-color: #f0f4f8 !important;  /* 框框背景改为蓝灰色 */
    }
    
    .stSelectbox > div > div > div {
        color: #000000 !important;  /* 选择框内的文字改为黑色 */
        background-color: #f0f4f8 !important;  /* 保持蓝灰色背景 */
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: #f0f4f8 !important;  /* 蓝灰色背景 */
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #000000 !important;  /* 黑色文字 */
        background-color: #f0f4f8 !important;  /* 蓝灰色背景 */
    }
    
    /* 🔧 修改输入框内的文字为黑色 */
    .stTextInput label {
        color: #000000 !important;  /* 输入框标签改为黑色 */
    }
    
    .stTextInput > div > div > input {
        border: 1px solid #6c8dbf !important;  /* 框框改为蓝灰色边框 */
        background-color: #f0f4f8 !important;  /* 框框背景改为蓝灰色 */
        color: #000000 !important;  /* 输入框内的文字改为黑色 */
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #495057 !important;  /* 占位符文字为深灰色 */
    }
    
    /* 🔧 修改数字输入框 */
    .stNumberInput label {
        color: #000000 !important;  /* 数字输入框标签改为黑色 */
    }
    
    .stNumberInput > div > div > input {
        border: 1px solid #6c8dbf !important;  /* 框框改为蓝灰色边框 */
        background-color: #f0f4f8 !important;  /* 框框背景改为蓝灰色 */
        color: #000000 !important;  /* 数字输入框内的文字改为黑色 */
    }
    
    /* 🔧 修改滑块标签文字为黑色 */
    .stSlider label {
        color: #000000 !important;  /* 滑块标签改为黑色 */
    }
    
    .stSlider > div > div > div {
        color: #000000 !important;  /* 滑块数值显示为黑色 */
    }
    
    /* 🔧 确保滑块所有文字都是黑色 */
    .stSlider * {
        color: #000000 !important;  /* 滑块内所有文字改为黑色 */
    }
    
    .stSlider [data-baseweb="slider"] {
        color: #000000 !important;  /* 滑块组件文字改为黑色 */
    }
    
    /* 🔧 修改单选按钮标签为黑色 */
    .stRadio label {
        color: #000000 !important;  /* 单选按钮标签改为黑色 */
    }
    
    .stRadio > div > label > div {
        color: #000000 !important;  /* 单选按钮选项文字改为黑色 */
    }
    
    /* 🔧 修改复选框标签为黑色 */
    .stCheckbox label {
        color: #000000 !important;  /* 复选框标签改为黑色 */
    }
    
    .stCheckbox > label > div {
        color: #000000 !important;  /* 复选框文字改为黑色 */
    }
    
    /* 🔧 修改切换开关标签为黑色 */
    .stToggle label {
        color: #000000 !important;  /* 切换开关标签改为黑色 */
    }
    
    /* 🔧 修改展开器内容为黑色 */
    .streamlit-expanderHeader,
    .stExpander > div > div > div > div,
    [data-testid="stExpander"] > div > div > div > div,
    .stExpander .streamlit-expanderHeader {
        background-color: #f0f4f8 !important;  /* 展开器头部改为蓝灰色背景 */
        color: #000000 !important;  /* 展开器标题改为黑色文字 */
        border: 1px solid #6c8dbf !important;  /* 蓝灰色边框 */
        font-size: 1.4em !important;  /* 显著增大展开器标题字号 */
        font-weight: bold !important;  /* 加粗字体使标题更突出 */
        padding: 12px 16px !important;  /* 增加内边距 */
        line-height: 1.2 !important;  /* 调整行高 */
    }
    
    .streamlit-expanderContent,
    .stExpander > div > div:last-child,
    [data-testid="stExpander"] > div > div:last-child {
        background-color: #f8faff !important;  /* 展开器内容区域浅蓝灰色背景 */
    }
    
    .streamlit-expanderContent label,
    .stExpander label,
    [data-testid="stExpander"] label {
        color: #000000 !important;  /* 展开器内的标签改为黑色 */
    }
    
    /* 额外的展开器样式确保生效 */
    .stExpander summary,
    [data-testid="stExpander"] summary,
    .stExpander > div > div > summary {
        font-size: 1.4em !important;  /* 展开器标题字号 */
        font-weight: bold !important;  /* 加粗 */
        color: #000000 !important;  /* 黑色文字 */
        background-color: #f0f4f8 !important;  /* 蓝灰背景 */
        padding: 12px 16px !important;  /* 内边距 */
        border-radius: 4px !important;  /* 圆角 */
    }
    
    /* 🔧 修改侧边栏内的标签文字为黑色 */
    .css-1d391kg label, 
    .css-1lcbmhc label,
    .css-1y4p8pa label {
        color: #000000 !important;  /* 侧边栏标签改为黑色 */
    }
    
    /* 🔧 确保侧边栏内的输入框和选择框也应用蓝灰色背景和黑色文字 */
    .css-1d391kg .stSelectbox > div > div,
    .css-1d391kg .stTextInput > div > div > input,
    .css-1d391kg .stNumberInput > div > div > input {
        background-color: #f0f4f8 !important;  /* 蓝灰色背景 */
        color: #000000 !important;  /* 黑色文字 */
        border: 1px solid #6c8dbf !important;  /* 蓝灰色边框 */
    }
    
    .css-1d391kg .stSelectbox [data-baseweb="select"] > div {
        color: #000000 !important;  /* 侧边栏选择框文字为黑色 */
        background-color: #f0f4f8 !important;  /* 蓝灰色背景 */
    }
    
    /* 🔧 修改下拉菜单选项的颜色 */
    [data-baseweb="menu"] {
        background-color: #f0f4f8 !important;  /* 下拉菜单背景为蓝灰色 */
    }
    
    [data-baseweb="menu"] [role="option"] {
        color: #000000 !important;  /* 下拉菜单选项文字为黑色 */
        background-color: #f0f4f8 !important;  /* 选项背景为蓝灰色 */
    }
    
    [data-baseweb="menu"] [role="option"]:hover {
        background-color: #e8f2ff !important;  /* 悬停时背景为更浅的蓝灰色 */
        color: #000000 !important;  /* 悬停时文字仍为黑色 */
    }
    
    /* 🔧 修改实验名称输入框的特定样式 */
    input[aria-label*="实验名称"] {
        background-color: #f0f4f8 !important;  /* 蓝灰色背景 */
        color: #000000 !important;  /* 黑色文字 */
        border: 1px solid #6c8dbf !important;  /* 蓝灰色边框 */
    }
    
    /* 🔧 确保所有表单控件的焦点状态也保持正确的颜色 */
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #1976d2 !important;  /* 焦点时边框为标准蓝色 */
        color: #000000 !important;  /* 焦点时文字仍为黑色 */
        background-color: #f0f4f8 !important;  /* 焦点时背景仍为蓝灰色 */
    }
</style>
""", unsafe_allow_html=True)

# --- 初始化 Session State ---
if 'is_simulating' not in st.session_state:
    st.session_state.is_simulating = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'current_config' not in st.session_state:
    st.session_state.current_config = None
if 'selected_mode_index' not in st.session_state:
    st.session_state.selected_mode_index = 0

# --- 主标题区域 ---
st.markdown('<h1 class="main-header">🐛 C. elegans 仿真系统</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">基于深度强化学习的多节段身体趋温行为建模平台</p>', unsafe_allow_html=True)

# ==============================================================================
# 侧边栏配置区域 - 蓝灰渐变主题
# ==============================================================================
with st.sidebar:
    st.markdown("### 🔬 实验配置")
    
    # 实验名称
    exp_name_default = f"exp_{datetime.datetime.now().strftime('%m%d_%H%M%S')}"
    experiment_name = st.text_input(
        "🎯 实验名称", 
        value=exp_name_default, 
        disabled=st.session_state.is_simulating,
        help="实验标识符"
    )
    
    # 实验模式选择
    st.markdown("### 🚀 训练模式")
    mode_options = ["🎓 标准训练", "🔄 迁移学习", "📚 课程学习"]
    
    if not st.session_state.is_simulating:
        selected_mode = st.radio(
            "选择模式",
            mode_options,
            index=st.session_state.selected_mode_index,
            label_visibility="collapsed",
            key="mode_selector"
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
            
        st.success(f"✅ {mode_choice}")
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
        st.info(f"🔄 {mode_choice}")

    st.markdown("---")

    # ==========================================================================
    # 身体参数配置
    # ==========================================================================
    with st.expander("🐛 身体形态设计", expanded=False):
        # 身体模型类型选择器
        body_model_map = {
            "🪱 多节段链条 (Worm2D)": "worm2d",
            "📏 连续中心线 (Continuous)": "continuous_centerline",
            "🌊 主动形变波 (Active Wave)": "active_deformation",
        }
        body_model_display = st.selectbox(
            "🧬 身体模型",
            list(body_model_map.keys()),
            index=0,
            disabled=st.session_state.is_simulating,
            key="body_model_selector",
            help="选择线虫身体的数学模型"
        )
        body_model_type = body_model_map[body_model_display]

        # ---- Worm2D 参数 ----
        if body_model_type == "worm2d":
            st.markdown("##### 🔧 身体结构参数")
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
            st.markdown("##### 🎯 运动参数")
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

        # ---- ContinuousCenterlineBody 参数 ----
        elif body_model_type == "continuous_centerline":
            st.markdown("##### 🔧 中心线参数")
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
                    help="中心线总弧长(像素)"
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
            st.markdown("##### 🔗 约束参数")
            col3, col4 = st.columns(2)
            with col3:
                curvature_limit = st.slider(
                    "曲率限制角", 15, 90, 45, 5,
                    disabled=st.session_state.is_simulating,
                    key="cc_curvature_limit",
                    help="相邻段最大弯折角度(度)"
                )
                length_stiffness = st.slider(
                    "长度刚度", 0.3, 1.0, 0.85, 0.05,
                    disabled=st.session_state.is_simulating,
                    key="cc_length_stiffness",
                    help="长度保持约束的刚度"
                )
            with col4:
                curvature_stiffness = st.slider(
                    "曲率刚度", 0.1, 1.0, 0.35, 0.05,
                    disabled=st.session_state.is_simulating,
                    key="cc_curvature_stiffness",
                    help="曲率约束的刚度"
                )
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

        # ---- ActiveDeformationBody 参数 ----
        else:  # active_deformation
            st.markdown("##### 🔧 中心线参数")
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
                    help="中心线总弧长(像素)"
                )
                head_radius = st.slider(
                    "头部半径", 2.0, 6.0, 3.0, 0.5,
                    disabled=st.session_state.is_simulating,
                    key="ad_head_radius",
                    help="头部圆形半径(像素)"
                )
            with col2:
                body_width = st.slider(
                    "身体宽度", 1.0, 4.0, 2.0, 0.25,
                    disabled=st.session_state.is_simulating,
                    key="ad_body_width",
                    help="身体节段宽度(像素)"
                )
                forward_speed = st.slider(
                    "前进速度", 0.5, 5.0, 1.2, 0.1,
                    disabled=st.session_state.is_simulating,
                    key="ad_forward_speed",
                    help="每步前进基础步长"
                )
                max_turn_angle = st.slider(
                    "最大转向角", 10, 60, 35, 5,
                    disabled=st.session_state.is_simulating,
                    key="ad_max_turn_angle",
                    help="每步最大转向角度(度)"
                )
            st.markdown("##### 🔗 约束参数")
            col3, col4 = st.columns(2)
            with col3:
                curvature_limit = st.slider(
                    "曲率限制角", 15, 90, 45, 5,
                    disabled=st.session_state.is_simulating,
                    key="ad_curvature_limit",
                    help="相邻段最大弯折角度(度)"
                )
                length_stiffness = st.slider(
                    "长度刚度", 0.3, 1.0, 0.85, 0.05,
                    disabled=st.session_state.is_simulating,
                    key="ad_length_stiffness",
                    help="长度保持约束的刚度"
                )
            with col4:
                curvature_stiffness = st.slider(
                    "曲率刚度", 0.1, 1.0, 0.35, 0.05,
                    disabled=st.session_state.is_simulating,
                    key="ad_curvature_stiffness",
                    help="曲率约束的刚度"
                )
                damping = st.slider(
                    "运动阻尼", 0.3, 0.95, 0.72, 0.05,
                    disabled=st.session_state.is_simulating,
                    key="ad_damping",
                    help="速度平滑阻尼系数"
                )
            st.markdown("##### 🌊 主动波参数")
            col5, col6 = st.columns(2)
            with col5:
                wave_amplitude = st.slider(
                    "波幅", 0.1, 3.0, 1.0, 0.1,
                    disabled=st.session_state.is_simulating,
                    key="ad_wave_amplitude",
                    help="正弦波侧向摆动幅度"
                )
                wave_frequency = st.slider(
                    "波频率", 0.05, 1.0, 0.25, 0.05,
                    disabled=st.session_state.is_simulating,
                    key="ad_wave_frequency",
                    help="肌肉波频率"
                )
            with col6:
                wave_speed = st.slider(
                    "波传播速度", 0.1, 3.0, 1.0, 0.1,
                    disabled=st.session_state.is_simulating,
                    key="ad_wave_speed",
                    help="波沿身体传播速度"
                )
                wave_length = st.slider(
                    "波长", 3.0, 30.0, 12.0, 0.5,
                    disabled=st.session_state.is_simulating,
                    key="ad_wave_length",
                    help="正弦波的波长"
                )
            # 为兼容性设置默认值
            num_segments = sample_count
            segment_length = body_length_ui
            backward_speed = 1.0
            turning_speed = 1.5

    # ==========================================================================
    # 噪声参数配置
    # ==========================================================================
    with st.expander("🌪️ 布朗噪声调节", expanded=False):
        st.markdown("##### 🔊 运动噪声参数")
        
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
        st.markdown("### 📊 标准设置")

        # 动作空间(方向)选择器 — 按身体模型过滤; 后续可扩展 16 方向
        if body_model_type == "worm2d":
            direction_options = {"4 方向 (离散)": "4", "8 方向 (离散)": "8"}
        else:
            direction_options = {"4 方向 (离散)": "4", "8 方向 (离散)": "8", "连续方向 (Actor-Critic)": "continuous"}
        direction_label = st.selectbox(
            "🧭 动作空间",
            list(direction_options.keys()),
            index=0,
            disabled=st.session_state.is_simulating,
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
            "🤖 学习算法",
            methods,
            index=default_method_idx,
            disabled=st.session_state.is_simulating,
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
            "🌡️ 环境类型", 
            list(field_map.keys()), 
            index=0, 
            disabled=st.session_state.is_simulating,
            key="field_selector_standard"
        )
        field_type = field_map[field_name]
        
        enable_step_tracking = st.toggle(
            "📈 详细分析",
            value=False,
            disabled=st.session_state.is_simulating,
            key="step_tracking_standard"
        )

    # ==========================================================================
    # 迁移学习实验配置
    # ==========================================================================
    elif mode_choice == "迁移学习实验":
        st.markdown("### 🔄 迁移设置")
        
        if not PYTORCH_AVAILABLE:
            st.error("⚠️ 需要 PyTorch")
            st.stop()
        
        method_choice = st.selectbox(
            "🧠 算法",
            ["🧠 DQN", "⚡ Dueling DQN"],
            index=1,
            disabled=st.session_state.is_simulating,
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
                "📚 源环境",
                transfer_options,
                index=0,
                disabled=st.session_state.is_simulating,
                key="source_field_selector"
            )
            # 提取英文名称
            source_field = source_field_display.split('(')[1].split(')')[0]
        
        with col2:
            target_field_display = st.selectbox(
                "🎯 目标环境",
                transfer_options,
                index=2,
                disabled=st.session_state.is_simulating,
                key="target_field_selector"
            )
            # 提取英文名称
            target_field = target_field_display.split('(')[1].split(')')[0]

    # ==========================================================================
    # 课程学习实验配置
    # ==========================================================================
    elif mode_choice == "课程学习实验":
        st.markdown("### 📚 课程设置")
        
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
            "🏆 测试阶段",
            curriculum_stages,
            index=len(curriculum_stages)-1,
            disabled=st.session_state.is_simulating,
            key="test_stage_selector"
        )
        test_stage_idx = curriculum_stages.index(test_stage_name)
        
        env_size = st.slider(
            "📐 环境尺寸",
            min_value=40, max_value=120, value=80, step=10,
            disabled=st.session_state.is_simulating,
            key="env_size_slider"
        )
        
        enable_step_tracking = st.toggle(
            "📊 详细分析",
            value=False,
            disabled=st.session_state.is_simulating,
            key="step_tracking_curriculum"
        )

    # ==========================================================================
    # 高级参数配置 - 紧凑版，减少默认训练量以避免卡死
    # ==========================================================================
    if mode_choice in ["标准训练模式", "迁移学习实验"]:
        with st.expander("⚙️ 高级参数", expanded=False):
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
                    "折扣因子", 0.8, 0.99, 0.95, 0.01,
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

            # Actor-Critic 专用超参数
            if "Actor-Critic" in method_choice:
                st.markdown("##### 🎯 Actor-Critic 超参数")
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
            st.markdown("##### 🎁 奖励函数")
            reward_variant = st.selectbox(
                "奖励变体",
                ["original", "energy"],
                index=0,
                format_func=lambda v: {"original": "不含能量（原版）", "energy": "含能量（单步化）"}[v],
                disabled=st.session_state.is_simulating,
                key="reward_variant_selectbox",
                help="original: Worm2D 原始温度阶梯（无能量项）；energy: 阶梯 + 单步能量惩罚 -w_e·ΔE/maxE"
            )
            reward_energy_weight = st.slider(
                "能量权重 w_e", 0.0, 0.5, 0.1, 0.01,
                format="%.2f",
                disabled=st.session_state.is_simulating or reward_variant != "energy",
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
    st.markdown("### 🎮 控制")
    
    if not st.session_state.is_simulating:
        start_button = st.button(
            "🚀 启动实验", 
            use_container_width=True, 
            type="primary",
            key="start_experiment_button"
        )
    else:
        start_button = False
        if st.button("⏹️ 停止", use_container_width=True, type="secondary", key="stop_experiment_button"):
            st.session_state.stop_requested = True

# ==============================================================================
# 主界面区域 - 紧凑设计
# ==============================================================================

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
        st.error(f"❌「{body_model_display}」身体模型目前仅支持标准训练 + Q-Learning 模式。\n\n请切换为「标准训练」模式或选择「Worm2D」身体模型。")
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
        # Actor-Critic 超参数
        "ac_hidden_size": ac_hidden_size,
        "ac_actor_lr": ac_actor_lr,
        "ac_critic_lr": ac_critic_lr,
        "ac_gamma": discount_factor,
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
    st.info(f"🔧 正在使用身体参数：节段数={num_segments}, 长度={segment_length}, 头径={head_radius}, 体宽={body_width}")
    st.info(f"🌪️ 正在使用噪声参数：位置噪声={position_noise}, 角度噪声={angle_noise}, 温度噪声={thermal_noise}")
    
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
    st.markdown("### 📊 实验监控")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 模式</h4>
            <p style="font-size: 1em; margin: 0;">{mode_choice}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🧠 算法</h4>
            <p style="font-size: 1em; margin: 0;">{method_choice}</p>
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
            <h4>🌡️ 环境</h4>
            <p style="font-size: 1em; margin: 0;">{display_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 身体参数展示
    st.markdown("#### 🐛 当前身体配置")
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
            card_text = f"<strong>阻尼:</strong> {damping}<br><strong>长度刚度:</strong> {length_stiffness}"
        else:
            card_text = f"<strong>后退:</strong> {backward_speed}px<br><strong>转向:</strong> {turning_speed}px"
        st.markdown(f"""
        <div class="compact-card">
            {card_text}
        </div>
        """, unsafe_allow_html=True)
    
    # 进度条和状态
    progress_bar = st.progress(0, text="🔄 初始化实验引擎...")
    status_placeholder = st.empty()
    
    # 结果展示区域 - 紧凑版
    st.markdown("### 📈 实验结果")
    
    result_tabs = st.tabs(["📊 图表", "🎥 动画", "📋 日志"])
    
    with result_tabs[0]:
        image_placeholder = st.empty()
        image_placeholder.markdown("""
        <div class="info-box">
            <h4>🖼️ 训练结果图表</h4>
            <p>训练完成后显示详细分析图表。</p>
        </div>
        """, unsafe_allow_html=True)
    
    with result_tabs[1]:
        video_placeholder = st.empty()
        video_placeholder.markdown("""
        <div class="info-box">
            <h4>🎥 行为动画</h4>
            <p>展示线虫训练过程中的行为轨迹。</p>
        </div>
        """, unsafe_allow_html=True)
    
    with result_tabs[2]:
        log_placeholder = st.empty()
        log_placeholder.code("⏳ 等待实验开始...", language="text")

    # 启动仿真引擎
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
        
        try:
            for current, total, message, stats in engine:
                if st.session_state.stop_requested:
                    break
                    
                if current == -1:
                    error_html = f"""
                    <div class="error-box">
                        <h4>💥 实验错误</h4>
                        <p>{message}</p>
                    </div>
                    """
                    st.markdown(error_html, unsafe_allow_html=True)
                    break
                
                # 更新进度条 - 直接更新，无需额外调用
                progress = current / total if total > 0 else 1.0
                progress_text = f"[{current}/{total}] {message}"
                progress_bar.progress(min(progress, 1.0), text=progress_text)
                
                # 更新状态信息
                if stats:
                    last_stats.update(stats)
                    if 'phase' in stats:
                        phase_emojis = {
                            "init": "🔄", "source_training": "🏋️", "target_testing": "🎯", 
                            "training": "📚", "testing": "🏆", "control": "🆚"
                        }
                        phase_emoji = phase_emojis.get(stats['phase'], "⚙️")
                        
                        status_html = f"""
                        <div class="status-box">
                            <h4>{phase_emoji} {stats.get('phase', '进行中').title()}</h4>
                        """
                        
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
            print(f"❌ 生成器处理错误: {gen_error}")
            error_html = f"""
            <div class="error-box">
                <h4>💥 处理错误</h4>
                <p>{gen_error}</p>
            </div>
            """
            st.markdown(error_html, unsafe_allow_html=True)

        # 实验完成处理
        print("🔧 调试：开始实验完成处理...")
        if not st.session_state.stop_requested:
            progress_bar.progress(1.0, text="✅ 实验完成！")
            
            success_html = """
            <div class="success-box">
                <h4>🎉 实验成功完成！</h4>
                <p>结果已保存到指定目录。</p>
            </div>
            """
            status_placeholder.markdown(success_html, unsafe_allow_html=True)
            
            # 显示最终日志
            log_placeholder.code("\n".join(log_content), language="text")
            
            # 显示结果图片
            try:
                results_image_path = None
                possible_paths = [
                    config.results_image,
                    os.path.join(config.output_dir, "training_results.png"),
                    os.path.join(config.output_dir, "training_results_simple.png")
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        results_image_path = path
                        break
                
                if results_image_path:
                    with result_tabs[0]:
                        image_placeholder.image(
                            results_image_path, 
                            caption="📊 训练结果分析",
                            use_column_width=True
                        )
                        print(f"✅ 成功显示结果图片: {results_image_path}")
                else:
                    print("⚠️ 未找到结果图片文件")
                    
            except Exception as e:
                with result_tabs[0]:
                    image_placeholder.markdown(f"""
                    <div class="error-box">
                        <h4>❌ 图表加载失败</h4>
                        <p>{e}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 🔧 添加：显示动画文件
            try:
                # 查找动画文件
                animation_paths = [
                    os.path.join(config.output_dir, "training_animation.gif"),
                    os.path.join(config.output_dir, "training_animation.mp4"),
                    os.path.join(config.output_dir, "worm_body_animation.gif"),
                    os.path.join(config.output_dir, "worm_body_animation.mp4"),
                    os.path.join(config.output_dir, "simple_training_animation.gif"),  # 回退动画
                ]
                
                found_animation = None
                animation_type = None
                
                for anim_path in animation_paths:
                    if os.path.exists(anim_path):
                        found_animation = anim_path
                        animation_type = "gif" if anim_path.endswith('.gif') else "mp4"
                        break
                
                if found_animation:
                    with result_tabs[1]:
                        if animation_type == "gif":
                            # 显示GIF动画
                            video_placeholder.image(
                                found_animation,
                                caption="🎥 线虫训练行为动画",
                                use_column_width=True
                            )
                            print(f"✅ 成功显示GIF动画: {found_animation}")
                        else:
                            # 显示MP4视频
                            try:
                                video_placeholder.video(
                                    found_animation,
                                    format="video/mp4",
                                    start_time=0
                                )
                                print(f"✅ 成功显示MP4视频: {found_animation}")
                            except Exception as video_error:
                                print(f"⚠️ MP4显示失败，尝试下载链接: {video_error}")
                                video_placeholder.markdown(f"""
                                <div class="info-box">
                                    <h4>🎥 训练动画</h4>
                                    <p>动画已生成，请前往结果目录查看：</p>
                                    <p><code>{found_animation}</code></p>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    with result_tabs[1]:
                        video_placeholder.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ 未找到动画文件</h4>
                            <p>动画可能生成失败或保存在其他位置。</p>
                        </div>
                        """, unsafe_allow_html=True)
                    print("⚠️ 未找到任何动画文件")
                    
            except Exception as anim_error:
                with result_tabs[1]:
                    video_placeholder.markdown(f"""
                    <div class="error-box">
                        <h4>❌ 动画加载失败</h4>
                        <p>{anim_error}</p>
                    </div>
                    """, unsafe_allow_html=True)
                print(f"❌ 动画显示错误: {anim_error}")
    except Exception as e:
        st.error(f"❌ 实验运行错误: {e}")
        st.code(traceback.format_exc(), language="python")
    
    finally:
        st.session_state.is_simulating = False
        st.session_state.stop_requested = False
        
        # 🔧 优化：使用按钮重启而不是自动rerun
        if st.button("🔄 新实验", use_container_width=True, type="primary", key="restart_button"):
            # 清理session state
            for key in list(st.session_state.keys()):
                if key.endswith('_slider') or key.endswith('_input') or key.endswith('_selector'):
                    continue  # 保留用户设置的参数
                if key in ['is_simulating', 'stop_requested', 'current_config']:
                    del st.session_state[key]
            st.rerun()

else:
    # 默认界面 - 紧凑版
    st.markdown("""
    <div class="info-box">
        <h3>👈 快速开始</h3>
        <p>在左侧配置参数，点击"启动实验"开始仿真。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 功能介绍 - 紧凑版
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎓 标准训练</h4>
            <p>单环境专精训练，适合基础研究。</p>
            <ul style="font-size: 0.9em;">
                <li>🎯 单环境训练</li>
                <li>📊 详细学习曲线</li>
                <li>🔧 多种算法</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🔄 迁移学习</h4>
            <p>验证知识迁移能力。</p>
            <ul style="font-size: 0.9em;">
                <li>🏋️ 源环境预训练</li>
                <li>🎯 目标环境测试</li>
                <li>📈 迁移效果评估</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>📚 课程学习</h4>
            <p>渐进式多阶段训练。</p>
            <ul style="font-size: 0.9em;">
                <li>📖 循序渐进</li>
                <li>🏆 留出法验证</li>
                <li>🆚 对照实验</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# 页脚 - 紧凑版
st.markdown("---")
st.markdown("""
<div class="footer-style">
    <h5>🐛 C. elegans 仿真系统</h5>
    <p style="font-size: 0.9em;">基于深度强化学习的多节段身体建模平台</p>
</div>
""", unsafe_allow_html=True)