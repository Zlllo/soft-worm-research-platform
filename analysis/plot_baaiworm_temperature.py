"""Reproduce the mirrored BAAIWorm temperature-field preview (no simulated paths)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from thermotaxis.baaiworm_temperature import BAAIWormTemperatureField


def main():
    chinese_fonts = ["PingFang SC", "Noto Sans CJK SC", "STHeiti", "Microsoft YaHei"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = [name for name in chinese_fonts if name in installed] + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(10, 4.7), layout="constrained")
    space = fig.add_subplot(121, projection="3d")
    profile = fig.add_subplot(122)
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 65), np.linspace(0, np.pi, 33))
    x, z = 4*np.cos(theta)*np.sin(phi), 4*np.sin(theta)*np.sin(phi)
    heights = np.linspace(-8, 8, 321)
    axis_points = np.column_stack([heights*0, heights, heights*0])
    for side, color, label in [(1, "#b55422", "温区在上方"), (-1, "#27689b", "温区在下方")]:
        field = BAAIWormTemperatureField((0, 0, 0), 1, (0, side, 0))
        y = 4*side + 2*np.cos(phi)
        surface_points = np.stack((x, y, z), axis=-1)
        np.testing.assert_allclose(field.temperature_c(surface_points), 20)
        space.plot_surface(x, z, y, color=color, alpha=.28, linewidth=0)
        space.plot_wireframe(x, z, y, rstride=6, cstride=8, color=color, linewidth=.5)
        profile.plot(heights, field.temperature_c(axis_points), color=color, lw=2.5, label=label)
    plane_x, plane_z = np.meshgrid([-5, 5], [-5, 5])
    space.plot_surface(plane_x, plane_z, np.zeros_like(plane_x), color="#777d87", alpha=.16)
    space.scatter(0, 0, 0, color="#222222", s=22)
    space.set(xlabel="X", ylabel="Z", zlabel="Y", zlim=(-7, 7), title="两个镜像场的 20 °C 等温面")
    space.set_box_aspect((10, 10, 14))
    space.view_init(elev=13, azim=-59)
    profile.axvline(0, color="#aeb5bb", lw=1)
    profile.axhline(20, color="#888888", lw=1, ls="--")
    profile.scatter([0], [19.125], color="#222222", s=23, zorder=3)
    profile.annotate("起点 19.125 °C", (0, 19.125), (1.1, 19.42), arrowprops={"arrowstyle":"-", "color":"#666666"})
    profile.set(xlabel="离开起始平面的有向距离（体长）", ylabel="头部所在位置的温度（°C）",
                title="经过起点、沿平面法线取样", xlim=(-8, 8), ylim=(18.92, 21.17))
    profile.spines[["top", "right"]].set_visible(False)
    profile.legend(loc="upper center", bbox_to_anchor=(.5,-.17), ncol=2, frameon=False)
    fig.suptitle("BAAIWorm 三维温度场设计", fontsize=17)
    output = Path(__file__).resolve().parents[1] / "docs/reports/assets/baaiworm-temperature/field.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
