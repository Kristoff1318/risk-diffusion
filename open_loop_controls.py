"""Slides 13-14 as written: diffusion CONTROLS u_k executed open loop (no tracking).

The existing checkpoint samples controls and states jointly. Here only the
controls are executed, blind: u_0..u_{N-1} with N = 383 (action k moves state
k to k+1; state 383 is the goal-conditioned end). Nothing looks at where the
ball actually is, so there is no feedback and A_k is the free point mass.

Same checkpoint, starts, goal, noise, gamma, 32-stage schedule and audit design
as compare_tracking.py. Runs on CPU with one thread, so samples differ from the
GPU samples in results/chance_comparison. For a paired comparison, each nominal
sample is also executed with the original tracking controller.

    python open_loop_controls.py pilot   # strength pilot on seed 10, no audits
    python open_loop_controls.py main    # seeds 0-2, budgets 5% / 10%, audits
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

import runtime  # puts the repository-local Diffuser first on sys.path
from diffuser.datasets import load_environment
from diffuser.datasets.normalization import LimitsNormalizer
from diffuser.models import GaussianDiffusion, TemporalUnet
from diffuser.models.helpers import apply_conditioning
from chance_guidance import wall_violation_probabilities, goal_violation_probability
from evaluate_tracking import collision_components, execute_reference, summarize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/open_loop_controls'
ASSETS = runtime.ASSETS
SETTINGS = dict(noise_std=.005, gamma=10., guidance_steps=32, steps=383, goal_radius=.5,
                validation_draws=256, audit_draws=256,
                validation_seed_base=2026110500, audit_seed_base=2026119500)


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model():
    config = json.loads((ASSETS / 'config.json').read_text())
    model_args = dict(config['model'])
    model_args['dim_mults'] = tuple(model_args['dim_mults'])
    diffusion = GaussianDiffusion(TemporalUnet(**model_args), **config['diffusion'])
    state = torch.load(ASSETS / f'state_{config["epoch"]}.pt', map_location='cpu')
    diffusion.load_state_dict(state['ema'])
    diffusion.eval()
    with np.load(ASSETS / 'normalization.npz') as limits:
        observations = LimitsNormalizer(np.stack([limits['observations_mins'], limits['observations_maxs']]))
        actions = LimitsNormalizer(np.stack([limits['actions_mins'], limits['actions_maxs']]))
    return diffusion, observations, actions


class OpenLoopChanceModel:
    """Slide 13 on the free point mass, driven directly by the controls."""

    def __init__(self, env, noise_std, steps):
        assert env.frame_skip == 1 and np.isclose(env.dt, .01)
        assert np.allclose(env.model.dof_damping, 1.) and np.allclose(env.model.actuator_gear[:, 0], 100.)
        mass = env.model.body_mass[1]
        self.a, self.b = float(mass / (mass + .01)), float(1. / (mass + .01))
        self.noise_std, self.steps = noise_std, steps
        self.wall_centers = torch.tensor(np.array([
            env.model.geom_pos[i, :2] - env.model.body_pos[1, :2] for i in range(env.model.ngeom)
            if (env.model.geom_id2name(i) or '').startswith('wall_')]), dtype=torch.float64)

    def propagate(self, controls, initial_state):
        """Slide 13, steps 1-3. controls: (B, >=N, 2) physical u in [-1, 1]."""
        position, velocity = initial_state[:, :2], initial_state[:, 2:]
        pp = torch.zeros_like(position)   # position variance, per axis
        pv = torch.zeros_like(position)   # position-velocity covariance
        vv = torch.zeros_like(position)   # velocity variance
        means, position_variances = [], []
        for k in range(self.steps):
            unclipped = (velocity.abs() < 5.).to(controls.dtype)
            # Step 1: mean state, x_bar_{k+1} = f(x_bar_k, u_k, 0). No feedback term.
            velocity = self.a * velocity.clamp(-5., 5.) + self.b * controls[:, k].clamp(-1., 1.)
            position = position + .01 * velocity
            # Step 2: A_k = [[1, .01*a], [0, a]] per axis (times the velocity-clip derivative).
            a_vv, a_pv = self.a * unclipped, .01 * self.a * unclipped
            # Step 3: Sigma_{k+1} = A Sigma A^T + G Sigma_w G^T (noise enters velocity).
            pp, pv, vv = (pp + 2. * a_pv * pv + a_pv.square() * vv,
                          a_vv * pv + a_pv * a_vv * vv,
                          a_vv.square() * vv + self.noise_std ** 2)
            means.append(torch.cat((position, velocity), -1))
            position_variances.append(pp)
        return torch.stack(means, 1), torch.stack(position_variances, 1)

    def risk(self, controls, initial_state, goal):
        means, variances = self.propagate(controls, initial_state)
        # Steps 4-6: wall events at x_1..x_{N-1}, goal at x_N (same functions as the repo).
        walls = wall_violation_probabilities(means[:, :-1, :2], variances[:, :-1], self.wall_centers)
        goal_risk = goal_violation_probability(means[:, -1, :2], variances[:, -1], goal)
        # Step 7: union-bound sum, not clipped at one.
        wall_risk = walls.sum((1, 2))
        return wall_risk + goal_risk, wall_risk, goal_risk, means, variances

    def energy(self, controls, initial_state, goal, budget, gamma):
        # Slide 14, steps 9-10: r = Delta_hat - Delta, E = softplus(gamma * r).
        return functional.softplus(gamma * (self.risk(controls, initial_state, goal)[0] - budget))


def sample(diffusion, observations, start, goal, seed, chance=None, budget=None, strength=0.):
    """Original DDPM loop; with strength > 0, slide 14 guidance on the CONTROL channels.

    As in chance_guidance.sample_guided: the energy is evaluated on the denoiser's
    clean prediction and its gradient follows that prediction back to the noisy
    sample. Guidance draws no random numbers, so nominal and guided share noise.
    """
    set_seed(seed)
    conditions = {0: start, diffusion.horizon - 1: np.r_[goal, 0., 0.]}
    conditions = {t: torch.tensor(observations.normalize(v)[None], dtype=torch.float32)
                  for t, v in conditions.items()}
    x = apply_conditioning(torch.randn((1, diffusion.horizon, diffusion.transition_dim)), conditions, 2)
    initial = torch.tensor(start[None], dtype=torch.float64)
    goal_tensor = torch.tensor(goal[None], dtype=torch.float64)
    trace, begun = [], time.perf_counter()
    for step in reversed(range(diffusion.n_timesteps)):
        timesteps = torch.full((1,), step, dtype=torch.long)
        if strength and step < SETTINGS['guidance_steps']:
            variable = x.detach().requires_grad_(True)
            clean = diffusion.predict_start_from_noise(variable, timesteps,
                                                       diffusion.model(variable, conditions, timesteps)).clamp(-1., 1.)
            clean = apply_conditioning(clean.clone(), conditions, 2)
            mean, variance, log_variance = diffusion.q_posterior(clean, variable, timesteps)
            # Action limits are exactly [-1, 1], so normalized controls are physical controls.
            energy = chance.energy(clean[..., :2].double(), initial, goal_tensor, budget, SETTINGS['gamma'])
            gradient = torch.autograd.grad(energy.sum(), variable)[0]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError('nonfinite chance-energy gradient')
            # Slide 14, steps 11-13: score_model - lambda*beta*grad(E), as a posterior-mean shift.
            mean, log_variance = mean.detach() - strength * variance.detach() * gradient, log_variance.detach()
            trace.append(dict(step=step, energy=float(energy.detach()), gradient_norm=float(gradient.norm())))
        else:
            with torch.no_grad():
                mean, _, log_variance = diffusion.p_mean_variance(x, conditions, timesteps)
        with torch.no_grad():
            x = mean + (step > 0) * (.5 * log_variance).exp() * torch.randn_like(x)
            x = apply_conditioning(x, conditions, 2)
    sample_np = x[0].numpy()
    controls = np.clip(sample_np[:, :2], -1., 1.)
    states = observations.unnormalize(sample_np[:, 2:])
    return controls, states, time.perf_counter() - begun, trace


def execute_controls(env, controls, start, goal, noise_std=0., noise_seed=0, save=False):
    """Native MuJoCo, controls played blindly for N steps; any contact or terminal miss fails."""
    steps = SETTINGS['steps']
    env.reset()
    env.set_state(np.asarray(start[:2]), np.asarray(start[2:]))
    env.set_target(np.asarray(goal))
    noise = noise_std * np.random.RandomState(noise_seed).randn(steps, 2)
    wall, _ = collision_components(env)
    first_contact = 0 if wall else None
    path = [env._get_obs().copy()] if save else None
    reached = False
    for k in range(steps):
        obs, _, _, _ = env.step(controls[k])
        contact, _ = collision_components(env)
        if contact and first_contact is None:
            first_contact = k + 1
        wall = wall or contact
        if noise_std:
            env.set_state(obs[:2].copy(), obs[2:] + noise[k])
            obs = env._get_obs()
        reached = reached or np.linalg.norm(obs[:2] - goal) < SETTINGS['goal_radius']
        if save:
            path.append(obs.copy())
    distance = float(np.linalg.norm(obs[:2] - goal))
    return dict(wall=bool(wall), goal_failure=distance >= SETTINGS['goal_radius'],
                failure=bool(wall or distance >= SETTINGS['goal_radius']), reached_goal=bool(reached),
                final_distance=distance, first_contact_step=first_contact,
                trajectory=np.array(path) if save else None)


def noisy_batch(env, run, draws, seed_base, keep=24):
    flags, examples = [], []
    for draw in range(draws):
        result = run(noise_seed=seed_base + draw, save=draw < keep)
        flags.append([result['wall'], False, result['goal_failure']])
        if draw < keep:
            examples.append(result['trajectory'])
    summary = summarize(flags)
    return summary, np.asarray(flags, dtype=bool), examples


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def task(env, seed):
    """Start and goal of the ORIGINAL experiment for this seed.

    The original ran gym 0.18; this environment has gym 0.23, whose reset
    draws a different start for the same seed. So starts are read from the
    saved original results instead of env.reset().
    """
    if seed == 10:
        start = np.load(ROOT / 'results/chance_comparison/pilot/walls_0.0.npz')['execution'][0]
    else:
        rows = json.loads((ROOT / 'results/chance_comparison/results.json').read_text())
        start = next(r['start'] for r in rows if r['seed'] == seed and r['name'].startswith('walls_'))
    return np.asarray(start, dtype=float).copy(), np.array([7., 9.])


def pilot():
    """Strength pilot on the original pilot start (seed 10); no noisy runs."""
    out = OUT / 'pilot'
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(SETTINGS, seed=10, budget=.1, strengths=[2e-5, 2e-4, 2e-3, 2e-2, .2, .6, 2., 6., 20.],
                    selection='fewest zero-noise failures, then latest first wall touch, '
                              'then smallest surrogate sum; tie -> smaller strength',
                    stochastic_executions=0)
    save_json(out / 'protocol.json', protocol)
    rows, todo = [], [0.] + protocol['strengths']
    torch.set_num_threads(1)
    env = load_environment('maze2d-large-v1')
    chance = OpenLoopChanceModel(env, SETTINGS['noise_std'], SETTINGS['steps'])
    diffusion, observations, _ = load_model()
    start, goal = task(env, 10)
    for strength in todo:
        try:
            controls, states, seconds, trace = sample(diffusion, observations, start, goal, 10,
                                                      chance, protocol['budget'], strength)
        except FloatingPointError as error:
            rows.append(dict(strength=strength, error=str(error)))
            log('PILOT', strength, 'ERROR', error)
            continue
        with torch.no_grad():
            risk = chance.risk(torch.tensor(controls[None], dtype=torch.float64),
                               torch.tensor(start[None]), torch.tensor(goal[None]))
        result = execute_controls(env, controls, start, goal, save=True)
        path = result.pop('trajectory')
        np.savez_compressed(out / f'strength_{strength}.npz', controls=controls, states=states, execution=path)
        rows.append(dict(strength=strength, surrogate_risk=float(risk[0]), generation_seconds=seconds,
                         zero_noise=result, max_gradient_norm=max([r['gradient_norm'] for r in trace] or [0.])))
        save_json(out / 'results.json', rows)
        log('PILOT', strength, 'risk sum %.4g' % rows[-1]['surrogate_risk'], 'zero-noise failure',
            result['failure'], 'first contact', result['first_contact_step'], '%.1fs' % seconds)
    scores = sorted((int(r['zero_noise']['failure']), -(r['zero_noise']['first_contact_step'] or 10 ** 6),
                     r['surrogate_risk'], r['strength'])
                    for r in rows if r['strength'] and 'error' not in r)
    decision = dict(selected_strength=scores[0][-1], scores=scores,
                    score_fields='(zero-noise failure, -first wall touch, risk sum, strength)',
                    scope='pilot seed 10 only; no audits')
    save_json(out / 'DECISION.json', decision)
    log('DECISION', decision)


def main():
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('Experiment already run; preserve existing outcomes.')
    decision = json.loads((OUT / 'pilot/DECISION.json').read_text())
    protocol = dict(SETTINGS, seeds=[0, 1, 2], budgets=[.05, .1], strength=decision['selected_strength'],
                    scene='walls only', execution='controls played open loop for 383 native steps',
                    risk_event='any wall contact at x_0..x_N OR goal distance >= 0.5 at x_N',
                    acceptance='one-sided exact 95% upper bound of validation failures <= budget',
                    comparison='each nominal sample also executed by the original tracking controller (800 steps)',
                    device='cpu, one thread',
                    source_sha256={p: runtime.sha(ROOT / p) for p in
                                   ['open_loop_controls.py', 'chance_guidance.py', 'evaluate_tracking.py', 'runtime.py']})
    save_json(OUT / 'protocol.json', protocol)
    torch.set_num_threads(1)
    env = load_environment('maze2d-large-v1')
    chance = OpenLoopChanceModel(env, SETTINGS['noise_std'], SETTINGS['steps'])
    diffusion, observations, _ = load_model()
    rows, began = [], time.perf_counter()
    for seed in protocol['seeds']:
        start, goal = task(env, seed)
        for budget in [None] + protocol['budgets']:
            name = f's{seed}_' + ('nominal' if budget is None else f'chance_d{budget}')
            try:
                controls, states, seconds, trace = sample(diffusion, observations, start, goal, seed, chance,
                                                          budget, 0. if budget is None else protocol['strength'])
            except FloatingPointError as error:
                rows.append(dict(name=name, seed=seed, budget=budget, error=str(error), accepted=False))
                save_json(OUT / 'results.json', rows)
                log('ERROR', name, error)
                continue
            with torch.no_grad():
                total, wall_risk, goal_risk, means, variances = chance.risk(
                    torch.tensor(controls[None], dtype=torch.float64), torch.tensor(start[None]), torch.tensor(goal[None]))
            zero = execute_controls(env, controls, start, goal, save=True)
            zero_path = zero.pop('trajectory')
            run = lambda noise_seed, save: execute_controls(env, controls, start, goal,
                                                            SETTINGS['noise_std'], noise_seed, save)
            base = 100 * seed
            validation, vflags, _ = noisy_batch(env, run, SETTINGS['validation_draws'],
                                                SETTINGS['validation_seed_base'] + base, keep=0)
            audit, aflags, examples = noisy_batch(env, run, SETTINGS['audit_draws'],
                                                  SETTINGS['audit_seed_base'] + base)
            row = dict(name=name, seed=seed, budget=budget, start=start.tolist(), goal=goal.tolist(),
                       generation_seconds=seconds, surrogate_risk=float(total), surrogate_wall=float(wall_risk),
                       surrogate_goal=float(goal_risk), final_position_std=float(variances[0, -1].sqrt().max()),
                       zero_noise=zero, validation=validation, audit=audit, trace=trace)
            arrays = dict(controls=controls, states=states, zero_noise=zero_path,
                          predicted_mean=means[0].numpy(), predicted_position_variance=variances[0].numpy(),
                          audit_examples=np.array(examples), validation_flags=vflags, audit_flags=aflags)
            if budget is not None:
                row['accepted'] = validation['upper95'] <= budget
            else:
                # Same sample, executed the current repo's way (state tracking, 800 steps).
                track = execute_reference(env, states, start, goal, save=True)
                arrays['tracking_zero_noise'] = track.pop('trajectory')
                flags, track_examples = [], []
                for draw in range(SETTINGS['audit_draws']):
                    result = execute_reference(env, states, start, goal, SETTINGS['noise_std'],
                                               SETTINGS['audit_seed_base'] + base + draw, save=draw < 24)
                    flags.append([result['wall'], result['added_obstacle'], result['goal_failure']])
                    if draw < 24:
                        track_examples.append(result['trajectory'])
                row['tracking'] = dict(zero_noise=track, audit=summarize(flags))
                arrays['tracking_examples'] = np.array(track_examples)
            np.savez_compressed(OUT / f'{name}.npz', **arrays)
            rows.append(row)
            save_json(OUT / 'results.json', rows)
            log('DONE', name, 'risk sum %.4g' % row['surrogate_risk'],
                'zero-noise fail', zero['failure'], 'audit %d/256' % audit['failures'],
                ('tracking audit %d/256' % row['tracking']['audit']['failures']) if budget is None
                else 'accepted %s' % row['accepted'], '%.1fs' % seconds)
    save_json(OUT / 'status.json', dict(complete=True, plans=len(rows), process_seconds=time.perf_counter() - began))
    log('COMPLETE')


def figure():
    """Plot saved outcomes only (no new runs). Red = a noisy run that touched a wall or missed the goal."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    env = load_environment('maze2d-large-v1')
    walls = OpenLoopChanceModel(env, SETTINGS['noise_std'], SETTINGS['steps']).wall_centers.numpy()
    rows = {r['name']: r for r in json.loads((OUT / 'results.json').read_text())}

    def clearance(points):
        offset = np.abs(points[:, None, :] - walls[None]) - .5
        outside = np.sqrt((np.clip(offset, 0, None) ** 2).sum(-1)) + np.minimum(offset.max(-1), 0)
        return outside.min(1) - .1

    def failed(path, goal):
        return bool((clearance(path[:, :2]) <= .002).any() or np.linalg.norm(path[-1, :2] - goal) >= .5)

    seeds = sorted({r['seed'] for r in rows.values()})
    fig, axes = plt.subplots(len(seeds), 4, figsize=(22, 5.2 * len(seeds)), squeeze=False)
    for i, seed in enumerate(seeds):
        nominal = rows[f's{seed}_nominal']
        goal = np.asarray(nominal['goal'])
        data = np.load(OUT / f's{seed}_nominal.npz')
        panels = [('Current repo: track the planned states', data['tracking_zero_noise'], data['tracking_examples'],
                   nominal['tracking']['audit'], None, data['states'])]
        for name, label in [('nominal', 'Slides: play controls blindly, no guidance'),
                            ('chance_d0.05', 'Slides: blind controls, guided, budget 5%'),
                            ('chance_d0.1', 'Slides: blind controls, guided, budget 10%')]:
            row = rows.get(f's{seed}_{name}')
            if row is None or 'error' in row:
                panels.append((label + ' (generation error)', None, None, None, None, None))
                continue
            arrays = np.load(OUT / f's{seed}_{name}.npz')
            panels.append((label, arrays['zero_noise'], arrays['audit_examples'], row['audit'], row, arrays['states']))
        for ax, (label, zero, examples, audit, row, states) in zip(axes[i], panels):
            for cx, cy in walls:
                ax.add_patch(Rectangle((cy - .5, cx - .5), 1, 1, color='.75', lw=0))
            ax.add_patch(Circle((goal[1], goal[0]), .5, color='green', alpha=.15))
            ax.set_xlim(-.9, 11.4); ax.set_ylim(8.4, -.9); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            if zero is None:
                ax.set_title(label, fontsize=10)
                continue
            ax.plot(states[:, 1], states[:, 0], '--', color='tab:blue', lw=1, alpha=.7)
            for path in examples:
                bad = failed(path, goal)
                ax.plot(path[:, 1], path[:, 0], color='red' if bad else 'tab:purple', lw=.6, alpha=.35)
            ax.plot(zero[:, 1], zero[:, 0], '-', color='black', lw=1.8)
            ax.plot(nominal['start'][1], nominal['start'][0], 'o', color='green', ms=7)
            text = f'{audit["failures"]}/256 noisy runs failed'
            if row is not None:
                text += f'; predicted risk sum {row["surrogate_risk"]:.3g}'
                if row['budget'] is not None:
                    text += '; ' + ('PASSES budget' if row['accepted'] else 'fails budget')
            ax.set_title(f'seed {seed} | {label}\n{text}', fontsize=10)
    handles = [plt.Line2D([], [], color='tab:blue', ls='--', label='planned states (from diffusion)'),
               plt.Line2D([], [], color='black', lw=1.8, label='executed with no noise'),
               plt.Line2D([], [], color='tab:purple', label='noisy run: safe'),
               plt.Line2D([], [], color='red', label='noisy run: touched a wall or missed goal')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=11)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(OUT / 'comparison.png', dpi=90)
    log('saved', OUT / 'comparison.png')


if __name__ == '__main__':
    if sys.argv[1] == 'figure':
        figure()
        sys.exit()
    {'pilot': pilot, 'main': main}[sys.argv[1]]()
