"""RFT 物理离线验证: A-f 地形图 + 方向 / 平方缩放 / 各向同性 / 转向 检查。

验证项 (对应文献结论):
1. 方向: 行波向尾传播 → 身体向头朝向 (com_x 增加)
2. 缩放: 速度 ∝ A²·f (Taylor 1951: U = ½·ω·q·b²)
3. 各向同性极限: drag_ratio=1.0 时速度 → 0 (推进必要条件)
4. 转向: 曲率偏置 b 的符号决定旋转方向
5. 子步: 相位每宏步推进 2π·f (与子步数无关), history 每子步一帧

运行: py -3 verify_rft_physics.py
输出: rft_verify_output/ 目录 (PNG 图) + 终端验证表
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.worm_body import ActiveDeformationBody  # noqa: E402

OUT = Path(__file__).resolve().parent / "rft_verify_output"
OUT.mkdir(exist_ok=True)

WARMUP = 30    # 预热宏步 (进入稳态)
MEASURE = 120  # 测量宏步
ARENA = 400


def make_body(A=1.0, f=0.25, b=0.0, drag_ratio=None, width=ARENA, height=ARENA):
    body = ActiveDeformationBody(
        start_pos=(width / 2, height / 2),
        width=width,
        height=height,
        body_params={
            "sample_count": 9,
            "body_length": 12.0,
            "wave_amplitude": A,
            "wave_frequency": f,
            "steer_bias": b,
            "wave_length": 12.0,
            "sub_steps": 10,
        },
        noise_params={},
    )
    if drag_ratio is not None:
        body.drag_ratio = drag_ratio  # 各向同性对照实验用
    return body


def measure(body, steps=MEASURE):
    """跑 steps 个宏步, 返回 (每步位移向量, 每步能耗, 每步转角)。"""
    x0, y0 = body.com.copy()
    theta0 = body.body_theta
    e0 = body.energy
    for _ in range(steps):
        body.step_physics(env=None, dt=1.0)
    disp = (body.com - np.array([x0, y0])) / steps
    energy_per_step = (e0 - body.energy) / steps
    dtheta = (body.body_theta - theta0) / steps
    return disp, energy_per_step, dtheta


def main():
    print("=" * 64)
    print("RFT 物理验证")
    print("=" * 64)

    # ── 1. 方向: b=0 时应沿头朝向 (+x) 前进 ──
    #    有限长线虫 RFT 存在相位锁定的平均倾角 (端点力矩), 侧漂角约 5° (含包络),
    #    且镜像协变: 从相位 π 出发的漂移反向 (自洽性证据, 非 bug)。
    body = make_body(A=1.0, f=0.25, b=0.0)
    for _ in range(WARMUP):
        body.step_physics(env=None, dt=1.0)
    disp, energy, _ = measure(body)
    speed = float(np.hypot(disp[0], disp[1]))
    drift_angle = math.degrees(math.atan2(disp[1], disp[0]))
    print(f"\n[1] 方向 (A=1, f=0.25): 每步位移 = ({disp[0]:+.4f}, {disp[1]:+.4f}), 速率 = {speed:.4f} 格/步")
    print(f"    波向尾传播, 应沿 +x 前进 → {'PASS' if disp[0] > 1e-3 else 'FAIL'}")
    print(f"    侧漂角 = {drift_angle:+.2f}° (有限线虫相位锁定漂移, 应 <15°) → "
          f"{'PASS' if abs(drift_angle) < 15.0 else 'FAIL'}")
    # 镜像协变: 相位 0 与相位 π 各出发 1 宏步, 侧向位移应互为反号
    def one_macro(phase0):
        b2 = make_body(A=1.0, f=0.25, b=0.0)
        b2.wave_phase = phase0
        b2._rebuild_shape_points()
        x0, y0 = b2.com.copy()
        b2.step_physics(env=None, dt=1.0)
        return b2.com - np.array([x0, y0])

    d0 = one_macro(0.0)
    d1 = one_macro(math.pi)
    print(f"    镜像协变: 相位 0 出发 y 位移 = {d0[1]:+.5f}, 相位 π 出发 = {d1[1]:+.5f} "
          f"(应互为反号) → {'PASS' if abs(d1[1] + d0[1]) < 1e-3 else 'FAIL'}")
    print(f"    子步相位: 每宏步应推进 2π·f = {2 * math.pi * 0.25:.4f} rad")
    print(f"    实际推进: {body.wave_phase:.4f} rad ({body.wave_phase % (2 * math.pi):.4f} mod 2π)")
    print(f"    history 帧数/宏步 = {len(body.history) / (WARMUP + MEASURE):.1f} (应=子步数10, 平滑动画)")

    # ── 2. A²·f 平方缩放 ──
    print(f"\n[2] 平方缩放 (Taylor 1951): 速度 ∝ A²·f")
    A_list = [0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    f_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    speeds = np.zeros((len(A_list), len(f_list)))
    costs = np.zeros_like(speeds)
    for i, A in enumerate(A_list):
        for j, f in enumerate(f_list):
            b = make_body(A=A, f=f, b=0.0)
            for _ in range(WARMUP):
                b.step_physics(env=None, dt=1.0)
            d, e, _ = measure(b)
            speeds[i, j] = float(np.hypot(d[0], d[1]))
            costs[i, j] = e / max(speeds[i, j], 1e-9)
    xs = np.array([A * A * f for A in A_list for f in f_list])
    ys = speeds.ravel()
    fit = np.polyfit(xs, ys, 1)
    pred = np.polyval(fit, xs)
    r2 = 1.0 - np.sum((ys - pred) ** 2) / np.sum((ys - ys.mean()) ** 2)
    print(f"    A=0 速度 = {speeds[0].max():.6f} (应≈0)")
    print(f"    线性拟合 speed = {fit[0]:.4f}·(A²f) + {fit[1]:.4f}, R² = {r2:.4f} (应≈1)")
    print(f"    → {'PASS' if r2 > 0.98 and speeds[0].max() < 1e-3 else 'FAIL'}")

    # ── 3. 各向同性极限 ──
    body = make_body(A=1.0, f=0.25, b=0.0, drag_ratio=1.0)
    for _ in range(WARMUP):
        body.step_physics(env=None, dt=1.0)
    disp, _, _ = measure(body)
    iso_speed = float(np.hypot(disp[0], disp[1]))
    print(f"\n[3] 各向同性 drag_ratio=1.0: 速率 = {iso_speed:.6f} (应≈0)")
    print(f"    → {'PASS' if iso_speed < 1e-3 else 'FAIL'}")

    # ── 4. 曲率偏置转向 ──
    print(f"\n[4] 曲率偏置转向 (b 符号 → 转向方向):")
    for b_val in (0.03, -0.03):
        body = make_body(A=1.0, f=0.25, b=b_val)
        for _ in range(WARMUP):
            body.step_physics(env=None, dt=1.0)
        _, _, dtheta = measure(body)
        print(f"    b={b_val:+.2f}: 每步转角 = {math.degrees(dtheta):+.4f}°")
    b1 = None
    body = make_body(A=1.0, f=0.25, b=0.03)
    for _ in range(WARMUP):
        body.step_physics(env=None, dt=1.0)
    _, _, dt1 = measure(body)
    body = make_body(A=1.0, f=0.25, b=-0.03)
    for _ in range(WARMUP):
        body.step_physics(env=None, dt=1.0)
    _, _, dt2 = measure(body)
    print(f"    → {'PASS' if dt1 * dt2 < 0 and abs(dt1) > 1e-5 else 'FAIL'} (两向符号相反)")

    # ── 绘图 ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n⚠️ matplotlib 不可用, 跳过绘图")
        return

    F, A = np.meshgrid(f_list, A_list)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    c1 = axes[0].contourf(F, A, speeds, levels=20)
    fig.colorbar(c1, ax=axes[0])
    cs = axes[0].contour(F, A, speeds, levels=8, colors="white", linewidths=0.5)
    axes[0].clabel(cs, fontsize=7, fmt="%.2f")
    axes[0].set_title("速度 (格/宏步) — 等速线应为 A²f=常数 的双曲线")
    axes[0].set_xlabel("频率 f"); axes[0].set_ylabel("波幅 A")
    c2 = axes[1].contourf(F, A, costs, levels=20)
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_title("单位距离能耗 (能耗/步 ÷ 速度)")
    axes[1].set_xlabel("频率 f"); axes[1].set_ylabel("波幅 A")
    axes[2].scatter(xs, ys, s=12)
    axes[2].plot(xs, pred, "r-", lw=2, label=f"fit: {fit[0]:.3f}·A²f + {fit[1]:.3f} (R²={r2:.3f})")
    axes[2].set_title("速度 vs A²·f (Taylor 平方律)")
    axes[2].set_xlabel("A²·f"); axes[2].set_ylabel("速度"); axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "rft_landscape.png", dpi=110)
    print(f"\n图已保存: {OUT / 'rft_landscape.png'}")


if __name__ == "__main__":
    main()
