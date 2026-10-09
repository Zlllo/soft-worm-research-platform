"""Independent event-driven continuous telegraph first-passage validation.

This validates the continuous analytic limit, not the finite-control-period core.
Uses exponential event times, no censoring, and exact reflecting/absorbing events.
"""
import json
import math
import random
import statistics


def validate(replicates=20000, seed=180421):
    rng = random.Random(seed)
    x0, target, speed, rate = .003, .0125, .0002, .1
    durations = []
    for _ in range(replicates):
        x, direction, elapsed = x0, (1 if rng.random() < .5 else -1), 0.
        while True:
            event_time = rng.expovariate(rate)
            boundary_time = (target-x)/speed if direction == 1 else x/speed
            if event_time >= boundary_time:
                elapsed += boundary_time
                if direction == 1:
                    break
                x, direction = 0., 1
            else:
                x += direction*speed*event_time
                elapsed += event_time
                direction *= -1
        durations.append(elapsed)
    expected = target/speed + rate*(target**2-x0**2)/speed**2
    mean = statistics.mean(durations)
    se = statistics.stdev(durations)/math.sqrt(replicates)
    return {"replicates": replicates, "seed": seed, "initial_x_m": x0,
            "target_m": target, "speed_m_s": speed, "rate_per_s": rate,
            "analytic_mfpt_s": expected, "simulation_mean_s": mean,
            "mean_standard_error_s": se, "standardized_difference": (mean-expected)/se,
            "censoring": "none", "initial_direction": "random equiprobable"}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
