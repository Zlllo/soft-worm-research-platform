"""CCB 冻结策略导航评估；不写回放、不更新网络、保留训练随机噪声状态。"""
import numpy as np

from .utils import reset_worm_for_new_round
from .worm_body import ContinuousCenterlineBody


def evaluate_continuous_policy(source, env, max_steps=500, starts=None):
    if source.model_name != 'continuous_centerline' or source.actor_critic_agent is None:
        raise ValueError('此评估仅用于 CCB 连续控制策略')
    if starts is None:
        starts = [(fx * (env.width - 1), fy * (env.height - 1)) for fx, fy in [
            (.125, .125), (.75, .125), (.75, .75), (.125, .75),
            (.5, .125), (.75, .5), (.5, .75), (.125, .5)]]
    params = {key: getattr(source, key) for key in [
        'body_length', 'forward_speed', 'max_turn_angle', 'damping',
        'angular_constraint', 'ac_gamma', 'goal_radius']}
    params['sample_count'] = source.num_segments
    params['curvature_limit_deg'] = source.angular_constraint
    body = ContinuousCenterlineBody(starts[0], env.width, env.height, params, {'position_noise': 0.0})
    agent = source.actor_critic_agent
    body.actor_critic_agent = agent
    body.use_actor_critic = body.continuous_sensor = True
    body.reward_variant = 'continuous'
    previous_mode, previous_actor_mode = agent.train_mode, agent.actor.training
    previous_noise = agent.noise.state.copy()
    records = []
    agent.eval()
    try:
        for start in starts:
            reset_worm_for_new_round(body, env, start, env.width, env.height, 'single_center')
            actual_start = [body.x, body.y]
            closest = float(env.get_distance_to_best(body.x, body.y))
            max_length_error, max_angle = 0.0, 0.0
            for _ in range(max_steps):
                body.decide_move_actor_critic(env)
                closest = min(closest, float(env.get_distance_to_best(body.x, body.y)))
                metrics = body.get_metrics()
                max_length_error = max(max_length_error, metrics['body_length_error_abs'])
                max_angle = max(max_angle, metrics['curvature_max_deg'])
                if body.episode_done:
                    break
            records.append({
                'requested_start': list(start), 'actual_start': actual_start,
                'success': body.goal_reached, 'steps': body.current_step,
                'closest_distance': closest,
                'final_distance': float(env.get_distance_to_best(body.x, body.y)),
                'max_length_error': max_length_error, 'max_angle_deg': max_angle,
                'trajectory': body.rl_step_records.copy(),
            })
    finally:
        agent.train_mode = previous_mode
        agent.actor.train(previous_actor_mode)
        agent.noise.state = previous_noise
    return {
        'mode': 'frozen_field_no_exploration_no_motion_noise',
        'information': 'legacy privileged observations include distance to known target',
        'goal_radius': source.goal_radius, 'max_steps': max_steps,
        'success_count': sum(row['success'] for row in records),
        'episode_count': len(records), 'episodes': records,
    }
