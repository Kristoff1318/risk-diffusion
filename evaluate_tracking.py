"""Independent native MuJoCo execution; no differentiable risk code is imported."""
import time
import numpy as np
from scipy.stats import beta
from environment import active_collision


def collision_components(env):
    wall, added = False, False
    for contact in env.sim.data.contact[:env.sim.data.ncon]:
        if contact.dist > .002:
            continue
        names = [env.model.geom_id2name(int(g)) or '' for g in (contact.geom1, contact.geom2)]
        wall = wall or any(name.startswith('wall_') for name in names)
        added = added or any(name.startswith('added_obstacle_') for name in names)
    return wall, added


def execute_reference(env, reference, start, goal, noise_std=0., noise_seed=0, save=False):
    """Same original controller and 800 steps, optionally with physical noise."""
    env.reset()
    env.set_state(np.asarray(start[:2]), np.asarray(start[2:]))
    env.set_target(np.asarray(goal))
    random = np.random.RandomState(noise_seed)
    noise = noise_std * random.randn(env.max_episode_steps, 2)
    wall, added = collision_components(env)
    path = [env._get_obs().copy()] if save else None
    effort, length = 0., 0.
    previous = np.asarray(start[:2]).copy()
    reached = False
    for step in range(env.max_episode_steps):
        state = env.state_vector().copy()
        if step < len(reference)-1:
            waypoint = reference[step+1]
        else:
            waypoint = reference[-1].copy()
            waypoint[2:] = 0.
        command = waypoint[:2] - state[:2] + (waypoint[2:] - state[2:])
        obs, _, _, _ = env.step(command)
        current_wall, current_added = collision_components(env)
        wall, added = wall or current_wall, added or current_added
        if noise_std:
            env.set_state(obs[:2].copy(), obs[2:] + noise[step])
            obs = env._get_obs()
        length += float(np.linalg.norm(obs[:2] - previous))
        previous = obs[:2].copy()
        effort += .01 * float(np.square(np.clip(command, -1., 1.)).sum())
        reached = reached or np.linalg.norm(obs[:2]-goal) < .5
        if save:
            path.append(obs.copy())
    distance = float(np.linalg.norm(obs[:2]-goal))
    return dict(wall=bool(wall), added_obstacle=bool(added), goal_failure=distance>=.5,
                failure=bool(wall or added or distance>=.5), reached_goal=bool(reached),
                final_distance=distance, effort=effort, length=length,
                trajectory=np.array(path) if save else None)


def summarize(flags):
    flags = np.asarray(flags, dtype=bool)
    failures = flags.any(1)
    count, n = int(failures.sum()), len(flags)
    lower = float(beta.ppf(.025, count, n-count+1)) if count else 0.
    upper = float(beta.ppf(.975, count+1, n-count)) if count<n else 1.
    upper_one_sided = float(beta.ppf(.95, count+1, n-count)) if count<n else 1.
    return dict(draws=n, failures=count, failure_rate=count/n, CI95=[lower,upper],
                upper95=upper_one_sided,
                counts=dict(zip(['wall','added_obstacle','goal_failure'], flags.sum(0).tolist())))


def evaluate(env, reference, start, goal, noise_std, draws, seed_base):
    flags, lengths, efforts = [], [], []
    example = None
    begun = time.perf_counter()
    for draw in range(draws):
        result = execute_reference(env, reference, start, goal, noise_std, seed_base+draw, save=draw==0)
        flags.append([result['wall'],result['added_obstacle'],result['goal_failure']])
        lengths.append(result['length']);efforts.append(result['effort'])
        if draw==0:example=result['trajectory']
    summary = summarize(flags)
    summary.update(seconds=time.perf_counter()-begun, mean_length=float(np.mean(lengths)),
                   mean_effort=float(np.mean(efforts)))
    return summary, np.asarray(flags,dtype=bool), example
