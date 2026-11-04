"""Reproduce legacy Diffuser sampling and waypoint tracking, without risk guidance."""
import argparse
import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

LEGACY = Path('/home/kristoff/Documents/rb-diffuser')
sys.path.insert(0, str(LEGACY))
import torch
from diffuser.datasets import load_environment
from diffuser.datasets.normalization import DatasetNormalizer, LimitsNormalizer
from diffuser.guides.policies import Policy

ROOT = Path(__file__).resolve().parent
CHECKPOINT = LEGACY / 'logs/maze2d-large-v1/diffusion/H384_T256'
EPOCH = 1960000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_policy():
    def config(name):
        with (CHECKPOINT / (name + '_config.pkl')).open('rb') as stream:
            return pickle.load(stream)
    model = config('model')()
    diffusion = config('diffusion')(model)
    diffusion.load_state_dict(torch.load(CHECKPOINT / f'state_{EPOCH}.pt')['ema'])
    diffusion.eval()
    # Restore the original normalizer classes from previously cached dataset limits.
    # LimitsNormalizer depends only on these extrema; no dataset load or download.
    normalizer = object.__new__(DatasetNormalizer)
    normalizer.action_dim, normalizer.observation_dim = 2, 4
    limits = np.load(ROOT / 'normalization.npz')
    normalizer.normalizers = {
        key: LimitsNormalizer(np.stack([limits[key + '_mins'], limits[key + '_maxs']]))
        for key in ('actions', 'observations')
    }
    return Policy(diffusion, normalizer)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample(policy, start, goal, seed):
    set_seed(seed)
    conditions = {0: start.copy(), policy.diffusion_model.horizon - 1: np.r_[goal, 0., 0.]}
    torch.cuda.synchronize()
    begun = time.perf_counter()
    with torch.no_grad():
        _, paths = policy(conditions, batch_size=1)
    torch.cuda.synchronize()
    return paths, time.perf_counter() - begun


def verify_sampling(policy, start, goal):
    paths, _ = sample(policy, start, goal, 0)
    cond = {0: start.copy(), policy.diffusion_model.horizon - 1: np.r_[goal, 0., 0.]}
    cond = policy._format_conditions(cond, 1)
    set_seed(0)
    with torch.no_grad():
        raw = policy.diffusion_model(cond).cpu().numpy()
    actions = policy.normalizer.unnormalize(raw[:, :, :2], 'actions')
    observations = policy.normalizer.unnormalize(raw[:, :, 2:], 'observations')
    assert np.array_equal(paths.actions, actions)
    assert np.array_equal(paths.observations, observations)
    return {'original_policy_vs_original_sampler_exact': True,
            'action_max_abs_error': float(np.max(np.abs(paths.actions - actions))),
            'reference_max_abs_error': float(np.max(np.abs(paths.observations - observations)))}


def execute(env, reference, start, goal):
    path = [start.copy()]
    commands, contacts, distances = [], [], []
    total_reward, first_success, max_penetration = 0., None, 0.
    for t in range(env.max_episode_steps):
        state = env.state_vector().copy()
        waypoint = reference[t + 1] if t < len(reference) - 1 else reference[-1].copy()
        if t >= len(reference) - 1:
            waypoint[2:] = 0.
        action = waypoint[:2] - state[:2] + (waypoint[2:] - state[2:])
        obs, reward, terminal, _ = env.step(action)
        # mujoco_py exposes the allocated contact buffer (100 slots), including
        # stale contacts. Only the first ncon entries belong to this step.
        wall_contacts = [c for c in env.sim.data.contact[:env.sim.data.ncon] if any(
            (env.model.geom_id2name(int(g)) or '').startswith('wall_')
            for g in (c.geom1, c.geom2))]
        contacts.append(any(c.dist <= .002 for c in wall_contacts))
        if wall_contacts:
            max_penetration = max(max_penetration, max(0., -min(float(c.dist) for c in wall_contacts)))
        distance = float(np.linalg.norm(obs[:2] - goal))
        if first_success is None and distance < .5:
            first_success = t
        path.append(obs.copy())
        commands.append(action.copy())
        distances.append(distance)
        total_reward += reward
        if terminal:
            break
    return dict(goal_reached=first_success is not None, first_success_step=first_success,
                final_goal_success=distances[-1] < .5, final_distance=distances[-1],
                min_distance=min(distances), wall_contact=any(contacts),
                wall_contact_steps=sum(contacts), max_penetration=max_penetration,
                collision_free_goal_reached=first_success is not None and not any(contacts),
                episode_steps=len(commands), total_reward=float(total_reward)), {
                    'execution': np.array(path), 'commands': np.array(commands),
                    'wall_contact_flags': np.array(contacts), 'goal_distances': np.array(distances)}


def report(out, env, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), squeeze=False)
    wall_indices = np.argwhere(env.maze_arr == 10)
    centers = wall_indices + 1. - env.model.body_pos[1, :2]
    for ax, row in zip(axes.flat, rows):
        data = np.load(out / f'episode_{row["seed"]}.npz')
        for p in centers:
            ax.add_patch(Rectangle((p[1] - .5, p[0] - .5), 1, 1, facecolor='.65'))
        for name, color, style in [('reference', 'tab:blue', '--'), ('execution', 'tab:orange', '-')]:
            p = data[name]
            ax.plot(p[:, 1], p[:, 0], style, color=color, lw=1.3, label=name)
        ax.plot(row['start'][1], row['start'][0], 'go', ms=4)
        ax.add_patch(Circle(row['goal'][::-1], .5, color='green', alpha=.15))
        ax.set_xlim(-.7, 11.3)
        ax.set_ylim(8.3, -.7)
        ax.set_aspect('equal')
        ax.set_title(f'seed {row["seed"]}: goal={row["goal_reached"]}, contact={row["wall_contact"]}', fontsize=9)
    axes.flat[0].legend(fontsize=7)
    for ax in list(axes.flat)[len(rows):]:
        ax.set_visible(False)
    fig.suptitle('Original Diffuser sampler + original state-tracking controller; no guidance or added noise')
    fig.tight_layout()
    fig.savefig(out / 'trajectories.png', dpi=140)
    plt.close(fig)
    summary = {key: sum(bool(r[key]) for r in rows) for key in (
        'goal_reached', 'final_goal_success', 'wall_contact', 'collision_free_goal_reached')}
    summary['episodes'] = len(rows)
    save_json(out / 'summary.json', summary)
    lines = ['# Original nominal Diffuser reproduction', '',
        'This run restores the original sampling and reference-tracking execution pipeline. '
        'There is no risk guidance, added obstacle, added execution noise, replanning, or retraining.', '',
        'The unchanged legacy Policy calls the unchanged GaussianDiffusion sampler. '
        'The checkpoint is EMA epoch 1960000. Original LimitsNormalizer classes use cached dataset extrema. '
        'The native environment is maze2d-large-v1. Each episode uses native reset seed 0–9 '
        'and the same-numbered Python/NumPy/Torch sampling seed. The default goal is retained. '
        'The reference has 384 states; execution runs all 800 native episode steps, tracking the final '
        'position with zero target velocity after the reference ends. No early stop at goal.', '',
        f'Reached goal at least once: **{summary["goal_reached"]}/{len(rows)}**. '
        f'Finished inside goal: **{summary["final_goal_success"]}/{len(rows)}**. '
        f'Any wall contact: **{summary["wall_contact"]}/{len(rows)}**. '
        f'Collision-free goal reaching: **{summary["collision_free_goal_reached"]}/{len(rows)}**.', '',
        'Goal reaching matches the old script: distance < 0.5 at any step. Wall contact is a separate '
        'measurement: any native wall contact with signed distance <= 0.002 during the full episode. '
        'These ten deterministic seed cases are a reproduction check, not a stochastic risk certificate.', '',
        '| Seed | Goal reached | First step (0-based) | Final distance | Wall-contact steps | Max penetration |',
        '|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f'| {r["seed"]} | {r["goal_reached"]} | {r["first_success_step"]} | '
                     f'{r["final_distance"]:.6f} | {r["wall_contact_steps"]} | {r["max_penetration"]:.6f} |')
    lines += ['', 'See protocol.json for pre-outcome choices and source/checkpoint hashes; '
        'verification.json for exact sampler parity; results.json and episode_*.npz retain every outcome.', '',
        'The earlier 97.39% experiment used a custom sampler, direct actions, added noise, '
        '383-step execution and different starts. It is an archived integration experiment, '
        'not a reproduction of this pipeline. These changes are bundled here; this run does not '
        'attribute differences to one change alone.', '', '![Trajectories](trajectories.png)']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/nominal_baseline_20261104')
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(environment='maze2d-large-v1', epoch=EPOCH, seeds=list(range(10)),
                    execution='original state-reference feedback, 800 steps', guidance=False,
                    added_noise=False, added_obstacles=False, goal_threshold=.5,
                    source_sha256={str(p): sha(p) for p in [
                        Path(__file__), ROOT / 'normalization.npz',
                        LEGACY / 'scripts/plan_maze2d.py', LEGACY / 'diffuser/guides/policies.py',
                        LEGACY / 'diffuser/models/diffusion.py',
                        CHECKPOINT / f'state_{EPOCH}.pt']})
    save_json(out / 'protocol.json', protocol)
    torch.set_num_threads(2)
    env = load_environment('maze2d-large-v1')
    assert env.max_episode_steps == 800
    policy = load_policy()
    env.seed(0)
    start = env.reset()
    save_json(out / 'verification.json', verify_sampling(policy, start, env.get_target()))
    rows = []
    for seed in protocol['seeds']:
        env.seed(seed)
        start = env.reset()
        goal = np.asarray(env.get_target(), dtype=float).copy()
        paths, seconds = sample(policy, start, goal, seed)
        metrics, arrays = execute(env, paths.observations[0], start, goal)
        row = dict(seed=seed, start=start.tolist(), goal=goal.tolist(), sampling_seconds=seconds, **metrics)
        rows.append(row)
        np.savez_compressed(out / f'episode_{seed}.npz', reference=paths.observations[0],
                            generated_actions=paths.actions[0], **arrays)
        save_json(out / 'results.json', rows)
        print(json.dumps(row), flush=True)
    report(out, env, rows)
    env.close()
    print('Saved', out, flush=True)


if __name__ == '__main__':
    main()
