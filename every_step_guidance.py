"""Slide 13 guides "at each diffusion step": guidance in ALL 256 denoising steps.

Earlier experiments guided only the last 32 of 256 steps. This changes only
that, in both execution setups:
  tracking: the planned states are followed by the original feedback controller
            for 800 ticks; risk model = chance_guidance.TrackingChanceModel.
  blind:    the planned controls are played open loop for 383 ticks;
            risk model = open_loop_controls.OpenLoopChanceModel.
Everything else as in open_loop_controls.py: original starts, velocity noise
0.005, gamma 10, budgets 5% / 10%, 256 validation + 256 audit runs, CPU, one thread.

    python every_step_guidance.py pilot tracking|blind   # strength pilot, seed 10
    python every_step_guidance.py main tracking|blind    # seeds 0-2
    python every_step_guidance.py figure                 # plots from saved results
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

import runtime  # repository-local Diffuser first on sys.path
from diffuser.datasets import load_environment
from diffuser.models.helpers import apply_conditioning
from chance_guidance import TrackingChanceModel
from evaluate_tracking import collision_components, summarize
from open_loop_controls import OpenLoopChanceModel, load_model, task, set_seed, save_json, log

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/every_step_guidance'
SETTINGS = dict(noise_std=.005, gamma=10., guidance_steps=256, budgets=[.05, .1], seeds=[0, 1, 2],
                pilot_seed=10, pilot_budget=.1, pilot_strengths=[2e-6, 2e-5, 2e-4, 2e-3, 2e-2],
                pilot_selection='fewest zero-noise failures, then latest first wall touch, '
                                'then smallest risk sum; tie -> smaller strength',
                validation_draws=256, audit_draws=256,
                validation_seed_base=2026110600, audit_seed_base=2026119600, goal_radius=.5)
TICKS = dict(tracking=800, blind=383)


class Setup:
    """Risk model (slide 13) and execution for one setup."""

    def __init__(self, name, env):
        self.name, self.ticks = name, TICKS[name]
        if name == 'tracking':
            bounds = np.load(runtime.ASSETS / 'normalization.npz')
            self.model = TrackingChanceModel(env, bounds, (), SETTINGS['noise_std'], self.ticks)
        else:
            self.model = OpenLoopChanceModel(env, SETTINGS['noise_std'], self.ticks)

    def risk(self, sample, initial, goal):
        """sample: normalized diffusion sample (B, 384, 6) in float64. Returns the union-bound sum."""
        if self.name == 'tracking':
            return self.model.risk(sample, initial, goal).total_risk
        return self.model.risk(sample[..., :2], initial, goal)[0]  # action limits are [-1, 1]

    def energy(self, sample, initial, goal, budget):
        # Slide 14, steps 9-10: E = softplus(gamma * (Delta_hat - Delta)).
        return functional.softplus(SETTINGS['gamma'] * (self.risk(sample, initial, goal) - budget))

    def execute(self, env, plan, start, goal, noise_std=0., noise_seed=0, save=False):
        """Native MuJoCo. Any wall contact, or ending >= 0.5 from the goal, is a failure."""
        env.reset()
        env.set_state(np.asarray(start[:2]), np.asarray(start[2:]))
        env.set_target(np.asarray(goal))
        noise = noise_std * np.random.RandomState(noise_seed).randn(self.ticks, 2)
        wall, _ = collision_components(env)
        first = 0 if wall else None
        path = [env._get_obs().copy()] if save else None
        reference = plan['states']
        for k in range(self.ticks):
            if self.name == 'tracking':  # same controller as evaluate_tracking.execute_reference
                state = env.state_vector().copy()
                if k < len(reference) - 1:
                    waypoint = reference[k + 1]
                else:
                    waypoint = reference[-1].copy()
                    waypoint[2:] = 0.
                command = waypoint[:2] - state[:2] + (waypoint[2:] - state[2:])
            else:
                command = plan['controls'][k]
            obs, _, _, _ = env.step(command)
            contact, _ = collision_components(env)
            if contact and first is None:
                first = k + 1
            wall = wall or contact
            if noise_std:
                env.set_state(obs[:2].copy(), obs[2:] + noise[k])
                obs = env._get_obs()
            if save:
                path.append(obs.copy())
        distance = float(np.linalg.norm(obs[:2] - goal))
        return dict(wall=bool(wall), goal_failure=distance >= SETTINGS['goal_radius'],
                    failure=bool(wall or distance >= SETTINGS['goal_radius']), final_distance=distance,
                    first_contact_step=first, trajectory=np.array(path) if save else None)


def sample(diffusion, observations, setup, start, goal, seed, budget=None, strength=0.):
    """Original DDPM loop; with strength > 0, slide 14 guidance at every denoising step.

    As before, the energy is evaluated on the denoiser's clean prediction and its
    gradient follows that prediction back to the noisy sample. Guidance draws no
    random numbers, so nominal and guided plans share all noise.
    """
    set_seed(seed)
    conditions = {t: torch.tensor(observations.normalize(v)[None], dtype=torch.float32)
                  for t, v in {0: start, diffusion.horizon - 1: np.r_[goal, 0., 0.]}.items()}
    x = apply_conditioning(torch.randn((1, diffusion.horizon, diffusion.transition_dim)), conditions, 2)
    initial = torch.tensor(start[None], dtype=torch.float64)
    goal_tensor = torch.tensor(goal[None], dtype=torch.float64)
    trace, begun = [], time.perf_counter()
    for step in reversed(range(diffusion.n_timesteps)):
        timesteps = torch.full((1,), step, dtype=torch.long)
        if strength and step < SETTINGS['guidance_steps']:
            variable = x.detach().requires_grad_(True)
            clean = diffusion.predict_start_from_noise(
                variable, timesteps, diffusion.model(variable, conditions, timesteps)).clamp(-1., 1.)
            clean = apply_conditioning(clean.clone(), conditions, 2)
            mean, variance, log_variance = diffusion.q_posterior(clean, variable, timesteps)
            energy = setup.energy(clean.double(), initial, goal_tensor, budget)
            gradient = torch.autograd.grad(energy.sum(), variable)[0]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f'nonfinite chance-energy gradient at step {step}')
            # Slide 14, steps 11-13: score_model - lambda*beta*grad(E), as a posterior-mean shift.
            mean, log_variance = mean.detach() - strength * variance.detach() * gradient, log_variance.detach()
            if step % 16 == 0 or step < 4:
                trace.append(dict(step=step, energy=float(energy.detach()), gradient_norm=float(gradient.norm())))
        else:
            with torch.no_grad():
                mean, _, log_variance = diffusion.p_mean_variance(x, conditions, timesteps)
        with torch.no_grad():
            x = mean + (step > 0) * (.5 * log_variance).exp() * torch.randn_like(x)
            x = apply_conditioning(x, conditions, 2)
    normalized = x[0].numpy()
    plan = dict(normalized=normalized, controls=np.clip(normalized[:, :2], -1., 1.),
                states=observations.unnormalize(normalized[:, 2:]))
    return plan, time.perf_counter() - begun, trace


def prepare(setup_name):
    torch.set_num_threads(1)
    env = load_environment('maze2d-large-v1')
    diffusion, observations, _ = load_model()
    return env, Setup(setup_name, env), diffusion, observations


def plan_risk(setup, plan, start, goal):
    with torch.no_grad():
        return float(setup.risk(torch.tensor(plan['normalized'][None], dtype=torch.float64),
                                torch.tensor(start[None]), torch.tensor(goal[None])))


def pilot(setup_name):
    out = OUT / setup_name / 'pilot'
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'protocol.json', dict(SETTINGS, setup=setup_name, ticks=TICKS[setup_name]))
    env, setup, diffusion, observations = prepare(setup_name)
    start, goal = task(env, SETTINGS['pilot_seed'])
    rows = []
    for strength in [0.] + SETTINGS['pilot_strengths']:
        try:
            plan, seconds, trace = sample(diffusion, observations, setup, start, goal,
                                          SETTINGS['pilot_seed'], SETTINGS['pilot_budget'], strength)
        except FloatingPointError as error:
            rows.append(dict(strength=strength, error=str(error)))
            log(setup_name, 'PILOT', strength, 'ERROR', error)
            continue
        result = setup.execute(env, plan, start, goal, save=True)
        path = result.pop('trajectory')
        np.savez_compressed(out / f'strength_{strength}.npz', controls=plan['controls'],
                            states=plan['states'], execution=path)
        rows.append(dict(strength=strength, surrogate_risk=plan_risk(setup, plan, start, goal),
                         generation_seconds=seconds, zero_noise=result, trace=trace))
        save_json(out / 'results.json', rows)
        log(setup_name, 'PILOT', strength, 'risk sum %.4g' % rows[-1]['surrogate_risk'], 'zero-noise failure',
            result['failure'], 'first touch', result['first_contact_step'],
            'end distance %.2f' % result['final_distance'], '%.0fs' % seconds)
    scores = sorted((int(r['zero_noise']['failure']), -(r['zero_noise']['first_contact_step'] or 10 ** 6),
                     r['surrogate_risk'], r['strength']) for r in rows if r['strength'] and 'error' not in r)
    decision = dict(selected_strength=scores[0][-1], scores=scores,
                    score_fields='(zero-noise failure, -first wall touch, risk sum, strength)')
    save_json(out / 'DECISION.json', decision)
    log(setup_name, 'DECISION', decision)


def noisy(setup, env, plan, start, goal, draws, seed_base, keep):
    flags, examples = [], []
    for draw in range(draws):
        result = setup.execute(env, plan, start, goal, SETTINGS['noise_std'], seed_base + draw, save=draw < keep)
        flags.append([result['wall'], False, result['goal_failure']])
        if draw < keep:
            examples.append(result['trajectory'])
    return summarize(flags), np.asarray(flags, dtype=bool), examples


def main(setup_name):
    out = OUT / setup_name
    if (out / 'protocol.json').exists():
        raise FileExistsError('Already run; preserve existing outcomes.')
    strength = json.loads((out / 'pilot/DECISION.json').read_text())['selected_strength']
    save_json(out / 'protocol.json', dict(
        SETTINGS, setup=setup_name, ticks=TICKS[setup_name], strength=strength,
        risk_event='any wall contact OR final goal distance >= 0.5',
        acceptance='one-sided exact 95% upper bound of validation failures <= budget',
        source_sha256={p: runtime.sha(ROOT / p) for p in ['every_step_guidance.py', 'open_loop_controls.py',
                                                         'chance_guidance.py', 'tracking_dynamics.py',
                                                         'evaluate_tracking.py']}))
    env, setup, diffusion, observations = prepare(setup_name)
    rows = []
    for seed in SETTINGS['seeds']:
        start, goal = task(env, seed)
        for budget in [None] + SETTINGS['budgets']:
            name = f's{seed}_' + ('nominal' if budget is None else f'guided_d{budget}')
            try:
                plan, seconds, trace = sample(diffusion, observations, setup, start, goal, seed,
                                              budget, 0. if budget is None else strength)
            except FloatingPointError as error:
                rows.append(dict(name=name, seed=seed, budget=budget, error=str(error)))
                save_json(out / 'results.json', rows)
                log(setup_name, 'ERROR', name, error)
                continue
            zero = setup.execute(env, plan, start, goal, save=True)
            zero_path = zero.pop('trajectory')
            validation, vflags, _ = noisy(setup, env, plan, start, goal, SETTINGS['validation_draws'],
                                          SETTINGS['validation_seed_base'] + 100 * seed, 0)
            audit, aflags, examples = noisy(setup, env, plan, start, goal, SETTINGS['audit_draws'],
                                            SETTINGS['audit_seed_base'] + 100 * seed, 24)
            row = dict(name=name, seed=seed, budget=budget, start=start.tolist(), goal=goal.tolist(),
                       generation_seconds=seconds, surrogate_risk=plan_risk(setup, plan, start, goal),
                       zero_noise=zero, validation=validation, audit=audit, trace=trace)
            if budget is not None:
                row['accepted'] = validation['upper95'] <= budget
            np.savez_compressed(out / f'{name}.npz', controls=plan['controls'], states=plan['states'],
                                zero_noise=zero_path, audit_examples=np.array(examples),
                                validation_flags=vflags, audit_flags=aflags)
            rows.append(row)
            save_json(out / 'results.json', rows)
            log(setup_name, 'DONE', name, 'risk sum %.4g' % row['surrogate_risk'],
                'zero-noise fail', zero['failure'], 'first touch', zero['first_contact_step'],
                'audit %d/256' % audit['failures'], 'accepted %s' % row.get('accepted'), '%.0fs' % seconds)
    save_json(out / 'status.json', dict(complete=True, plans=len(rows)))
    log(setup_name, 'COMPLETE')


def figure():
    """One figure per setup from saved results only. Red = noisy run that touched a wall or missed the goal."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    env = load_environment('maze2d-large-v1')
    walls = OpenLoopChanceModel(env, SETTINGS['noise_std'], 1).wall_centers.numpy()

    def failed(path, goal):
        offset = np.abs(path[:, None, :2] - walls[None]) - .5
        clearance = (np.sqrt((np.clip(offset, 0, None) ** 2).sum(-1)) + np.minimum(offset.max(-1), 0)).min(1) - .1
        return bool((clearance <= .002).any() or np.linalg.norm(path[-1, :2] - goal) >= .5)

    titles = dict(tracking='Track the planned states', blind='Play the controls blindly')
    for setup_name in ['tracking', 'blind']:
        if not (OUT / setup_name / 'results.json').exists():
            continue
        rows = {r['name']: r for r in json.loads((OUT / setup_name / 'results.json').read_text())}
        fig, axes = plt.subplots(3, 3, figsize=(17, 15.5), squeeze=False)
        for i, seed in enumerate(SETTINGS['seeds']):
            for j, (key, label) in enumerate([('nominal', 'no guidance'), ('guided_d0.05', 'guided, budget 5%'),
                                              ('guided_d0.1', 'guided, budget 10%')]):
                ax = axes[i, j]
                for cx, cy in walls:
                    ax.add_patch(Rectangle((cy - .5, cx - .5), 1, 1, color='.78', lw=0))
                ax.add_patch(Circle((9, 7), .5, color='green', alpha=.2))
                ax.set_xlim(-.9, 11.4); ax.set_ylim(8.4, -.9); ax.set_aspect('equal')
                ax.set_xticks([]); ax.set_yticks([])
                row = rows.get(f's{seed}_{key}')
                if row is None or 'error' in row:
                    ax.set_title(f'seed {seed} | {label}: not generated', fontsize=10)
                    continue
                data = np.load(OUT / setup_name / f's{seed}_{key}.npz')
                goal = np.asarray(row['goal'])
                ax.plot(data['states'][:, 1], data['states'][:, 0], '--', color='tab:blue', lw=1, alpha=.7)
                for path in data['audit_examples']:
                    ax.plot(path[:, 1], path[:, 0], color='red' if failed(path, goal) else 'tab:purple',
                            lw=.6, alpha=.35)
                zero = data['zero_noise']
                ax.plot(zero[:, 1], zero[:, 0], '-', color='black', lw=1.8)
                touch = row['zero_noise']['first_contact_step']
                if touch is not None:
                    ax.plot(zero[touch, 1], zero[touch, 0], 'X', color='red', ms=10)
                ax.plot(row['start'][1], row['start'][0], 'o', color='green', ms=7)
                text = f'{row["audit"]["failures"]}/256 noisy runs failed; risk sum {row["surrogate_risk"]:.3g}'
                if row['budget'] is not None:
                    text += '; ' + ('PASSES budget' if row['accepted'] else 'fails budget')
                ax.set_title(f'seed {seed} | {label}\n{text}', fontsize=10)
        handles = [plt.Line2D([], [], color='tab:blue', ls='--', label='planned states (from diffusion)'),
                   plt.Line2D([], [], color='black', lw=1.8, label='executed with no noise'),
                   plt.Line2D([], [], color='red', marker='X', ls='', ms=9, label='first wall touch (no noise)'),
                   plt.Line2D([], [], color='tab:purple', label='noisy run: safe'),
                   plt.Line2D([], [], color='red', label='noisy run: touched a wall or missed goal')]
        fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=11)
        fig.suptitle(f'{titles[setup_name]}: guidance at every denoising step', fontsize=14)
        fig.tight_layout(rect=(0, .03, 1, .97))
        fig.savefig(OUT / f'{setup_name}.png', dpi=85)
        plt.close(fig)
        log('saved', OUT / f'{setup_name}.png')


if __name__ == '__main__':
    if sys.argv[1] == 'figure':
        figure()
    else:
        {'pilot': pilot, 'main': main}[sys.argv[1]](sys.argv[2])
