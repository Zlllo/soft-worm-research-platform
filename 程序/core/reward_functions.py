# --- File: core/reward_functions.py ---
"""
奖励函数模块 - 统一所有身体模型的奖励计算。

变体:
  - "original": Worm2D 原始温度阶梯奖励（逐位保留），不含能量项
  - "energy":   温度阶梯 + 单步能量惩罚 -w_e * ΔE / max_energy

设计说明 (2026-08-11):
  - 模块化: 原来散落在 worm_body.py 中的奖励逻辑（Worm2D `_calculate_reward`、
    CCB Q-learning / Actor-Critic 的手写奖励）统一收口到这里，后续只改本文件。
  - 单步化: 旧 CCB/ADB 用绝对能量 (maxE - E) * 0.001 作惩罚，混入"从开局到现在的
    总消耗"时间信号；新方案按步计算 ΔE = max(0, old_energy - energy)，只对当步
    消耗计费，不再依赖历史。
  - 改量级: 旧权重 0.001 相对温度奖励（0.1~5.0）可忽略不计；默认权重提升到 0.1。
  - Worm2D 能量从不衰减（ΔE ≡ 0），因此 energy 变体对其逐值等价于 original，
    前端在选择 Worm2D + 含能量时提示用户。
"""

DEFAULT_ENERGY_WEIGHT = 0.1


def compute_reward(env, worm, variant="original", old_energy=None,
                   energy_weight=DEFAULT_ENERGY_WEIGHT,
                   stuck_penalty=0.0, constraint_penalty=0.0,
                   target=None, movement=0.0):
    """
    统一奖励函数。

    Args:
        env: 温度环境（提供 get_temperature）
        worm: 身体模型（需有 recent_temperatures / energy / max_energy）
        variant: "original"（无能量项）或 "energy"（阶梯 + 单步能量惩罚）
        old_energy: 执行物理步之前的能量（energy 变体需要）
        energy_weight: 单步能量惩罚权重 w_e，默认 0.1
        stuck_penalty: 位移 < 1e-9 时的额外惩罚（AC 路径传 0.5，其余传 0）
        constraint_penalty: 曲率约束违例惩罚（AC 路径传 违例率*0.5，其余传 0）
        target: 温度采样位置 (x, y)，默认 worm 当前位置
        movement: 当步实际位移（stuck 判定用）

    Returns:
        float: 奖励值。出界位置直接返回 -10.0（与 Worm2D 原始行为一致）。
    """
    if target is None:
        target = (worm.x, worm.y)
    target_x, target_y = int(round(target[0])), int(round(target[1]))

    # ── 基础温度阶梯奖励 (Worm2D 原始奖励, 逐位保留) ──
    new_temp = env.get_temperature(target_x, target_y)

    # 出界或无效位置
    if new_temp == -float('inf'):
        return -10.0

    # 基础温度奖励系统 - 调整为120度最佳温度
    if new_temp >= 120:
        temp_reward = 5.0
    elif new_temp >= 115:
        temp_reward = 4.5
    elif new_temp >= 110:
        temp_reward = 4.0
    elif new_temp >= 100:
        temp_reward = 3.0
    elif new_temp >= 90:
        temp_reward = 2.0
    elif new_temp >= 80:
        temp_reward = 1.0
    elif new_temp >= 70:
        temp_reward = 0.5
    elif new_temp >= 60:
        temp_reward = 0.3
    elif new_temp >= 50:
        temp_reward = 0.1
    elif new_temp >= 40:
        temp_reward = 0.0
    elif new_temp >= 30:
        temp_reward = -0.3
    elif new_temp >= 20:
        temp_reward = -0.5
    elif new_temp >= 10:
        temp_reward = -0.8
    else:
        temp_reward = -1.0

    # 温度梯度奖励
    if len(worm.recent_temperatures) >= 2:
        recent_avg = sum(worm.recent_temperatures[-2:]) / 2
        if new_temp > recent_avg + 1.0:
            temp_reward += 1.0
        elif new_temp > recent_avg + 0.5:
            temp_reward += 0.3
        elif new_temp < recent_avg - 1.0:
            temp_reward -= 0.8
        elif new_temp < recent_avg - 0.5:
            temp_reward -= 0.2

    temp_reward += 0.1
    reward = temp_reward

    # ── 单步能量惩罚 (仅 energy 变体) ──
    if variant == "energy" and old_energy is not None:
        delta_energy = max(0.0, float(old_energy) - float(getattr(worm, 'energy', 0.0)))
        max_energy = max(float(getattr(worm, 'max_energy', 1.0)), 1.0)
        reward -= energy_weight * delta_energy / max_energy

    # ── 附加惩罚 (AC 路径安全正则项) ──
    if stuck_penalty and movement < 1e-9:
        reward -= stuck_penalty
    if constraint_penalty:
        reward -= constraint_penalty

    return reward


def apply_reward_config(worm, training_params):
    """
    根据训练参数把奖励变体装配到线虫对象上。

    training_params 支持键:
      - reward_variant: "original" / "energy"
      - reward_energy_weight: 单步能量惩罚权重（默认 DEFAULT_ENERGY_WEIGHT）

    在仿真引擎创建线虫后调用一次即可；重置线虫不会覆盖这两个属性。
    """
    variant = str(training_params.get("reward_variant", "original"))
    weight = float(training_params.get("reward_energy_weight", DEFAULT_ENERGY_WEIGHT))
    worm.reward_variant = variant
    worm.reward_energy_weight = weight
    return worm
