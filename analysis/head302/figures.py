"""Summarize the paired design and render the preselected single-source example."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import animation, pyplot as plt
import numpy as np

from compare import CASES, FIELDS, GRAPH_SEEDS, OUT, TRAIN_SEEDS, write_csv
from core.plot_fonts import setup_chinese_font

MODELS = ("real", "rewired", "dense", "straight")
NAMES = dict(real="真实连接", rewired="保度重排", dense="全连接 RNN", straight="直行")
COLORS = dict(real="#007872", rewired="#995221", dense="#5454a0", straight="#777777")
LABELS = ("单热源", "中线梯度", "双热源", "环形热源")


def cubes(rows, model, metric):
    graphs = GRAPH_SEEDS if model == "rewired" else (0,)
    seeds = (0,) if model == "straight" else TRAIN_SEEDS
    values = np.full((len(graphs), len(seeds), CASES, len(FIELDS)), np.nan)
    for row in rows:
        if row["model"] == model:
            index = (graphs.index(int(row["graph_seed"])), seeds.index(int(row["fit_seed"])),
                     int(row["case"]), FIELDS.index(row["field"]))
            if np.isfinite(values[index]):
                raise ValueError("duplicate episode")
            values[index] = float(row[metric])
    if not np.isfinite(values).all():
        raise ValueError("incomplete evaluation grid")
    return values


def paired_differences(cube, other):
    # Resample the crossed factors independently, sharing case and fit draws
    # across architectures and fields. Never pool episodes as independent fits.
    diff = cube["real"] - cube[other]
    rng = np.random.default_rng(81103)
    g, s, n, _ = diff.shape
    samples = []
    for _ in range(4000):
        indices = np.ix_(rng.integers(g, size=g), rng.integers(s, size=s), rng.integers(n, size=n))
        samples.append(diff[indices].mean(axis=(0, 1, 2)))
    samples = np.asarray(samples)
    samples = np.column_stack((samples, samples.mean(axis=1)))
    means = diff.mean(axis=(0, 1, 2))
    return np.append(means, means.mean()), np.percentile(samples, [2.5, 97.5], axis=0)


def comparison(folder):
    with (folder / "episodes.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary, contrasts = [], []
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), gridspec_kw={"width_ratios": [1.14, 1]})
    for metric in ("mean_heat", "hot_residence"):
        cube = {model: cubes(rows, model, metric) * 100 for model in MODELS}
        for i, model in enumerate(MODELS):
            means = cube[model].mean(axis=(0, 1, 2))
            for field, mean in zip((*FIELDS, "equal_field_mean"), np.append(means, means.mean())):
                summary.append(dict(metric=metric, model=model, field=field, percent=round(mean, 5)))
            if metric == "mean_heat":
                y = np.arange(4) + (i-1.5)*.12
                axes[0].scatter(means, y, s=48, color=COLORS[model], label=NAMES[model], zorder=3)
        for i, other in enumerate(("rewired", "dense")):
            means, (lo, hi) = paired_differences(cube, other)
            for field, avg, low, high in zip((*FIELDS, "equal_field_mean"), means, lo, hi):
                contrasts.append(dict(metric=metric, comparator=other, field=field,
                                      real_minus_other_pp=round(avg, 5),
                                      low95=round(low, 5), high95=round(high, 5)))
            if metric == "mean_heat":
                means, lo, hi = means[:-1], lo[:-1], hi[:-1]
                axes[1].errorbar(means, np.arange(4)+(i-.5)*.18,
                                 xerr=np.array([means-lo, hi-means]), color=COLORS[other],
                                 fmt="o", capsize=3, lw=1.8, label="相对" + NAMES[other])
    write_csv(folder / "summary.csv", summary)
    write_csv(folder / "differences.csv", contrasts)
    axes[0].set(xlim=(20, 101), xlabel="轨迹平均归一化温度（%）", title="四种温场的闭环成绩")
    axes[1].axvline(0, color="#666666", lw=.8)
    axes[1].set(xlabel="真实连接减去对照（百分点）", title="配对差值与 95% 重采样区间")
    for ax in axes:
        ax.set_yticks(range(4), LABELS)
        ax.set_ylim(3.55, -.55)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=.17)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.09, right=.98, top=.88, bottom=.30, wspace=.30)
    fig.savefig(folder / "comparison.png", dpi=175)
    plt.close(fig)


def single_source_gif(folder):
    with np.load(folder / "example.npz") as data:
        field = data["single_center"]
        traces = [data["single_center_" + model] for model in MODELS]
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 3.05))
    lines, heads = [], []
    for ax, model, trace in zip(axes, MODELS, traces):
        ax.imshow(field, origin="lower", cmap="inferno", interpolation="bilinear")
        ax.plot(*trace[0], "o", mfc="none", mec="white", ms=6)
        ax.plot(20, 20, "+", color="white", ms=7)
        lines.append(ax.plot([], [], color="#b4ffee", lw=1.8)[0])
        heads.append(ax.plot([], [], "o", color="#b4ffee", ms=4)[0])
        ax.set(xticks=[], yticks=[], title=NAMES[model])
    title = fig.suptitle("")
    fig.text(.5, .04, "○ 共同起点      + 热源中心      亮色表示更热", ha="center")
    fig.subplots_adjust(left=.02, right=.98, bottom=.15, top=.78, wspace=.08)

    def draw(t):
        title.set_text(f"单热源  ·  头部轨迹     {t:02d} / {len(traces[0])-1} 步")
        for line, head, trace in zip(lines, heads, traces):
            line.set_data(*trace[:t+1].T)
            head.set_data(*trace[t:t+1].T)
        return [title, *lines, *heads]

    movie = animation.FuncAnimation(fig, draw, frames=range(0, len(traces[0]), 2))
    movie.save(folder / "single-source.gif", writer=animation.PillowWriter(fps=10), dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    folder = parser.parse_args().out
    setup_chinese_font()
    comparison(folder)
    single_source_gif(folder)
