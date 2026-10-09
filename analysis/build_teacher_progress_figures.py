"""Build two teacher-report figures from the current project evidence."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "程序"))
from core.plot_fonts import setup_chinese_font

OUT = ROOT / "进度汇报_2026-10-06" / "讲述稿配图"
RECORDS = ROOT / "docs/reports/assets/legacy-policy-bridge-2026-10-04/training-records.json"


def architecture():
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.set(xlim=(0, 13.5), ylim=(0, 6.6))
    ax.axis("off")
    ax.text(.25, 6.15, "项目阶段进展与后续衔接", fontsize=21, weight="bold")
    ax.text(.25, 5.7, "平台功能、独立研究核心与后续任务分开展示；截至2026年10月8日", fontsize=12, color="#505050")
    columns = [
        (.25, "现有平台：已扩展", "身体选择：Worm2D / CCB / ADB\n离散动作：4方向、8方向\n连续控制：朝向/幅度或波参数\nQ-learning、DQN与DDPG\n训练图表、身体动画与日志", "#276b78"),
        (4.8, "独立核心：已有实验验证", "统一单位、时钟与随机源\n局部含噪感知与快慢记忆\n固定材料弧长、完整形变速度\n阻力平衡、功率与累计耗散\n四方向执行与冻结策略接入", "#77526d"),
        (9.35, "后续工作：待接入与研究", "新身体接入现有可视化平台\n身体与局部趋温机制结合\n统一观测、动作和控制时钟\n保存模型并开展独立评估\n连续控制与复杂温度场比较", "#4d5b46"),
    ]
    for x, title, body, color in columns:
        ax.add_patch(Rectangle((x, 2.05), 3.9, 3.25, facecolor="white", edgecolor="#c8c8c8", linewidth=1))
        ax.plot([x, x + 3.9], [5.3, 5.3], color=color, linewidth=3)
        ax.text(x + .16, 4.87, title, fontsize=14, weight="bold", color=color)
        ax.text(x + .16, 4.35, body, fontsize=12, va="top", linespacing=1.9)
    ax.annotate("接入", xy=(9.2, 3.75), xytext=(8.8, 3.75), fontsize=10,
                ha="right", va="center", arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.text(.25, 1.38, "新身体的计算链路", fontsize=14, weight="bold")
    ax.text(.25, .9, "驱动参数 → 连续形变 → 局部阻力 → 合力/合力矩平衡 → 位移、转动与耗散", fontsize=13)
    ax.text(.25, .35, "已有成果：点模型机制对照、身体机械基准、策略接口测试；新身体趋温与泛化训练尚待开展。", fontsize=11, color="#505050")
    fig.savefig(OUT / "项目进展结构图.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def training():
    records = json.loads(RECORDS.read_text(encoding="utf-8"))
    assert len(records) == 60
    assert [r["episode"] for r in records] == list(range(60))
    rounds = np.arange(1, 61)
    rewards = np.array([r["total_reward"] for r in records], dtype=float)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.subplots_adjust(top=.79, bottom=.2, left=.09, right=.97)
    fig.text(.09, .94, "Q-learning接口样本的训练记录", fontsize=19, weight="bold")
    fig.text(.09, .865, "Worm2D · 17×17网格 · 固定单一种子 · 60轮，每轮120步", fontsize=12, color="#505050")
    ax.plot(rounds, rewards, color="#276b78", linewidth=1.5, marker="o", markersize=3, label="每轮累计奖励")
    ax.plot(rounds[4:], np.convolve(rewards, np.ones(5) / 5, mode="valid"),
            color="#77526d", linewidth=2, linestyle="--", label="最近5轮平均（仅辅助读图）")
    ax.set(xlabel="训练轮次", ylabel="累计奖励（原平台奖励尺度）", xlim=(1, 60), ylim=(0, 650))
    ax.grid(axis="y", color="#dedede", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.text(.09, .065, "用途：验证策略保存、冻结加载和身体接入。每轮起点变化；未进行多种子及独立测试，不据此认定收敛或泛化。", fontsize=10, color="#505050")
    fig.savefig(OUT / "Q学习接口样本训练曲线.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if not setup_chinese_font():
        raise RuntimeError("No installed Chinese font is available")
    plt.rcParams["font.size"] = 12
    architecture()
    training()
    print(OUT)
