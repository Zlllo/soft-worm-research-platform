"""Create the figures in the pathway-specificity research note."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("docs/reports/assets/pathway-specificity")
TEAL, RUST = "#126a71", "#b45332"


def main():
    plt.rcParams.update(
        {"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10}
    )
    data = json.loads((ROOT / "summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    with (ROOT / "amphid-outputs.csv").open() as f:
        counts = list(csv.DictReader(f))
    for cls, color in (("AFD", TEAL), ("ASE", RUST)):
        shares = []
        for d in range(1, 9):
            rows = [r for r in counts if r["source"][:-1] == cls and int(r["dataset"]) == d]
            shares.append(
                100 * sum(int(r["ipsi_aiy"]) for r in rows) / sum(int(r["total"]) for r in rows)
            )
        axes[0].plot(range(1, 9), shares, "o-", color=color, label=cls)
    axes[0].set(
        title="解剖重建中的感觉输出",
        xlabel="Witvliet 重建编号",
        ylabel="送到同侧 AIY 的输出占比（%）",
        xticks=range(1, 9),
        ylim=(-4, 105),
    )
    axes[0].legend(frameon=False)
    for j, interface in enumerate(("full", "restricted")):
        values = data["lesions"]["matched"][interface]
        keys = ("intact", "AFD-AIY-matched", "AIZ-RIA-matched")
        x = np.arange(3) + (j - 0.5) * 0.14
        color = (TEAL, RUST)[j]
        for seed in ("7", "17", "27"):
            axes[1].plot(
                x, [100 * values[k][seed]["complete"] for k in keys], color=color, alpha=0.28, lw=1
            )
        axes[1].plot(
            x,
            [100 * np.mean([m["complete"] for m in values[k].values()]) for k in keys],
            "o-",
            color=color,
            lw=2,
            label=("全接口", "受限接口")[j],
        )
    axes[1].set(
        title="同等权重削减后的导航",
        ylabel="完整成功率（%）",
        xticks=range(3),
        xticklabels=["完整", "减弱\nAFD→AIY", "减弱\nAIZ→RIA"],
        ylim=(-4, 105),
    )
    axes[1].legend(frameon=False, fontsize=9)
    for i, interface in enumerate(("full", "restricted")):
        delta = np.array(data["routing"][interface]["paired_by_seed"])[:, 0] * 100
        axes[2].scatter(i + np.linspace(-0.1, 0.1, len(delta)), delta, color=(TEAL, RUST)[i], s=23)
        axes[2].plot([i - 0.2, i + 0.2], [delta.mean()] * 2, color="black", lw=2.3)
    axes[2].axhline(0, color="#777777", ls="--", lw=0.8)
    axes[2].set(
        title="从头学习的配对布局差",
        ylabel="原图减交换图的成功率（百分点）",
        xticks=[0, 1],
        xticklabels=["全接口", "受限接口"],
        xlim=(-0.5, 1.5),
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.16)
    fig.savefig(ROOT / "pathway-comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
