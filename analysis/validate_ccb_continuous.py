"""可复现的原平台 CCB 连续导航检验；不代表 SI 身体趋温结论。"""
import argparse
import contextlib
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '程序'))
with contextlib.redirect_stdout(io.StringIO()):
    from core.environment import Environment2D
    from core.utils import create_temperature_environment, reset_worm_for_new_round, reset_ccb_training_round
    from core.worm_body import ContinuousCenterlineBody
import torch

STARTS = [(5, 5), (30, 5), (30, 30), (5, 30), (20, 5), (30, 20), (20, 30), (5, 20)]


def make_body(noise=0.0, hidden=64, **params):
    config = dict(sample_count=9, body_length=12, forward_speed=1.2,
                  ac_hidden_size=hidden, ac_gamma=0.99)
    config.update(params)
    return ContinuousCenterlineBody((5, 5), 40, 40, config, {'position_noise': noise})


def evaluate(source, env, max_steps, hidden=64):
    from core.evaluation import evaluate_continuous_policy
    return evaluate_continuous_policy(source, env, max_steps, STARTS)['episodes']


def validate_geometry():
    rng = np.random.default_rng(20261008)
    body = make_body()
    worst_length = worst_angle = 0.0
    for _ in range(2000):
        body.step_physics(action={'heading': rng.uniform(-math.pi, math.pi),
                                  'step': rng.uniform(0, 5)})
        points = np.asarray(body.body_segments)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        worst_length = max(worst_length, float(np.max(abs(lengths - 1.5))))
        worst_angle = max(worst_angle, body.get_metrics()['curvature_max_deg'])
        assert np.all(points >= -1e-8) and np.all(points <= 39 + 1e-8)
    assert worst_length < 1e-8 and worst_angle <= 45 + 1e-6
    return {'steps': 2000, 'max_segment_length_error': worst_length, 'max_angle_deg': worst_angle}


def run(seed, env, args):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    body = make_body(args.noise, args.hidden, ac_random_starts=args.start_mode == "mixed")
    body.reward_variant = args.reward
    with contextlib.redirect_stdout(io.StringIO()):
        body.setup_actor_critic()
    agent = body.actor_critic_agent
    initial = evaluate(body, env, args.steps, args.hidden)
    rounds = []
    begin = time.monotonic()
    for round_num in range(args.rounds):
        # 起点计划与界面一致，不使用目标方向来替代策略。
        reset_ccb_training_round(body, env, (5, 5), 40, 40, 'single_center', round_num)
        for _ in range(args.steps):
            body.decide_move_actor_critic(env)
            if body.episode_done:
                break
        rounds.append(dict(round=round_num + 1, success=body.goal_reached,
                           steps=body.current_step, reward=body.total_reward,
                           final_distance=float(env.get_distance_to_best(body.x, body.y))))
        if (round_num + 1) % 25 == 0:
            print(json.dumps({'seed': seed, 'round': round_num + 1,
                              'recent_successes': sum(r['success'] for r in rounds[-25:]),
                              'elapsed_s': round(time.monotonic() - begin, 1)}), flush=True)
    final = evaluate(body, env, args.steps, args.hidden)
    checkpoint = args.output.parent / f'ccb-ddpg-seed-{seed}-{args.reward}.pt'
    torch.save(agent.state_dict(), checkpoint)
    return dict(seed=seed, reward=args.reward, rounds=rounds, initial_evaluation=initial,
                final_evaluation=final, elapsed_s=time.monotonic() - begin,
                checkpoint=str(checkpoint), replay_transitions=len(agent.replay_buffer))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--sampling', choices=['bilinear', 'grid'], default='bilinear')
    parser.add_argument('--start-mode', choices=['fixed', 'mixed'], default='mixed')
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--reward', choices=['original', 'continuous'], default='continuous')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ccb-continuous-2026-10-08/validation.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        array, best = create_temperature_environment(40, 40, None, 'single_center')
    env = Environment2D(array, best)
    if args.sampling == 'grid':
        def sample_grid(x, y):
            if not (0 <= x <= env.width - 1 and 0 <= y <= env.height - 1):
                return -float('inf')
            return env.get_temperature(x, y)
        env.get_temperature_continuous = sample_grid
    payload = dict(configuration=vars(args).copy(), geometry=validate_geometry(), runs=[])
    payload['configuration']['output'] = str(args.output)
    for seed in args.seeds:
        payload['runs'].append(run(seed, env, args))
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(args.output), 'successes': [
        sum(row['success'] for row in r['final_evaluation']) for r in payload['runs']]}), flush=True)


if __name__ == '__main__':
    main()
