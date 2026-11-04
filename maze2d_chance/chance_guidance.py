"""Chance guidance, organized in the order of PowerPoint slides 13 and 14.

Read the implementation in this order:
    1. propagate_mean_and_covariance  — slide 13, steps 1–3
    2. wall_violation_probabilities   — slide 13, steps 4–6
    3. goal_violation_probability    — slide 13, steps 4–6
    4. DirectActionChanceModel.risk  — slide 13, steps 7–8
    5. DirectActionChanceModel.energy — slide 14, steps 9–10
    6. sample_direct_actions         — slide 14, steps 11–13 and autodiff

Execution model: generated ACTIONS are rolled out to obtain mean STATES. The
sampled observation channels are not used as the mean trajectory. This model
has NOT been connected to the nominal baseline's state-tracking controller;
nominal_baseline.py does not call this guide. This is the unfinished integration
previously described as "experimental", not a claim that state guidance is
inherently experimental.

Maze2D-specific assumptions: known initial state, independent Gaussian velocity
noise after each native step, and native free-space dynamics before contact.
Walls and the circular goal are locally linearized. The resulting union sum
is an approximate risk surrogate, not a proven bound on nonlinear execution.

The mathematical model and sampling update of the earlier direct-action guide
are preserved. This readability cleanup does not run a new guided experiment.
"""
import time
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as functional

# Sets up imports of the repository-local Diffuser source.
import runtime
from diffuser.models.helpers import apply_conditioning

DEFAULT_NOISE_STD = 0.005
DEFAULT_GAMMA = 10.0
DEFAULT_GUIDANCE_STRENGTH = 0.002  # Product lambda * beta in slide 14.


class RiskEstimate(NamedTuple):
    total_risk: torch.Tensor
    wall_risk: torch.Tensor
    goal_risk: torch.Tensor
    mean_states: torch.Tensor
    position_variances: torch.Tensor


def propagate_mean_and_covariance(actions, initial_state, noise_std,
                                  velocity_decay, action_gain, dt=0.01):
    """Slide 13, steps 1–3: mean rollout, dynamics Jacobian, covariance.

    State order is [position_x, position_y, velocity_x, velocity_y].
    Native pre-contact dynamics use semi-implicit Euler and implicit damping:
        v_next = a * clip(v, -5, 5) + b * clip(u, -1, 1)
        p_next = p + dt * v_next

    The local Jacobian A includes the velocity-clipping derivative. Covariance
    follows the slide's Sigma_next = A Sigma A.T + G Sigma_w G.T.
    Noise is injected into velocity AFTER the position update, matching the
    declared execution model. The initial covariance is zero (known state).
    """
    batch_size = len(actions)
    position = initial_state[:, :2]
    velocity = initial_state[:, 2:]
    covariance = actions.new_zeros((batch_size, 4, 4))
    process_covariance = actions.new_zeros((4, 4))
    process_covariance[2, 2] = noise_std ** 2
    process_covariance[3, 3] = noise_std ** 2
    mean_states, state_covariances = [], []

    for action in actions.unbind(dim=1):
        # Slide 13, step 2: A_k = df/dx at the current mean state.
        clip_derivative = (velocity.abs() < 5.0).to(actions.dtype)
        velocity_jacobian = velocity_decay * clip_derivative
        jacobian = actions.new_zeros((batch_size, 4, 4))
        jacobian[:, 0, 0] = 1.0
        jacobian[:, 1, 1] = 1.0
        jacobian[:, 0, 2] = dt * velocity_jacobian[:, 0]
        jacobian[:, 1, 3] = dt * velocity_jacobian[:, 1]
        jacobian[:, 2, 2] = velocity_jacobian[:, 0]
        jacobian[:, 3, 3] = velocity_jacobian[:, 1]

        # Slide 13, step 1: x_bar_next = f(x_bar, u, mean_noise).
        velocity = (velocity_decay * velocity.clamp(-5.0, 5.0)
                    + action_gain * action.clamp(-1.0, 1.0))
        position = position + dt * velocity

        # Slide 13, step 3: propagate the complete 4-by-4 covariance.
        covariance = (jacobian @ covariance @ jacobian.transpose(-1, -2)
                      + process_covariance)
        mean_states.append(torch.cat((position, velocity), dim=-1))
        state_covariances.append(covariance)

    return torch.stack(mean_states, dim=1), torch.stack(state_covariances, dim=1)


def gaussian_violation_probability(mean_margin, standard_deviation):
    """Slide 13, step 6: delta = 1 - Phi(m/s), with g >= 0 safe."""
    return 0.5 * torch.erfc(mean_margin / (standard_deviation * np.sqrt(2.0)))


def wall_violation_probabilities(mean_positions, position_variances, wall_centers,
                                 agent_radius=0.1, contact_margin=0.002):
    """Slide 13, steps 4–6: g, grad(g), s² = grad(g).T Sigma grad(g).

    Maze walls are unit boxes. g is signed distance from the agent center to a
    box, minus agent radius and contact margin. Negative g means collision.
    Per-axis noise is independent in this model, so position Sigma is diagonal.
    """
    # Slide 13, steps 4–5: evaluate and linearize each wall constraint.
    displacement = mean_positions[:, :, None, :] - wall_centers
    box_offset = displacement.abs() - 0.5
    outside_offset = box_offset.clamp(min=0.0)
    outside_distance = outside_offset.square().sum(-1).clamp(min=1.e-20).sqrt()
    signed_distance = outside_distance + box_offset.max(-1).values.clamp(max=0.0)
    mean_margin = signed_distance - (agent_radius + contact_margin)

    outside_gradient = outside_offset / outside_distance[..., None]
    inside_gradient = functional.one_hot(box_offset.argmax(-1), 2).to(box_offset.dtype)
    constraint_gradient = torch.where(
        (box_offset.max(-1).values > 0.0)[..., None],
        outside_gradient, inside_gradient,
    ) * displacement.sign()
    margin_variance = (constraint_gradient.square()
                       * position_variances[:, :, None, :]).sum(-1)
    standard_deviation = margin_variance.clamp(min=1.e-12).sqrt()

    # Slide 13, step 6: Gaussian probability of violating each wall at each time.
    return gaussian_violation_probability(mean_margin, standard_deviation)


def goal_violation_probability(final_mean_position, final_position_variance,
                                goal_position, goal_radius=0.5):
    """Slide 13, steps 4–6: linearize g_goal = radius - distance to goal."""
    displacement = final_mean_position - goal_position
    distance = displacement.square().sum(-1).clamp(min=1.e-20).sqrt()
    # The gradient sign cancels in grad(g).T Sigma grad(g).
    direction = displacement / distance[:, None]
    margin_variance = (direction.square() * final_position_variance).sum(-1)
    standard_deviation = margin_variance.clamp(min=1.e-12).sqrt()
    return gaussian_violation_probability(goal_radius - distance, standard_deviation)


class DirectActionChanceModel:
    """Slides 13–14 applied to Maze2D direct actions, not reference tracking."""

    def __init__(self, env, bounds, sigma=DEFAULT_NOISE_STD):
        assert env.frame_skip == 1 and np.isclose(env.dt, 0.01)
        assert np.allclose(env.model.dof_damping, 1.0)
        assert np.allclose(env.model.actuator_gear[:, 0], 100.0)
        mass = env.model.body_mass[1]
        self.velocity_decay = float(mass / (mass + 0.01))
        self.action_gain = float(1.0 / (mass + 0.01))
        self.noise_std = sigma
        # Convert world wall centers to the agent's qpos coordinate system.
        self.wall_centers = torch.tensor(np.array([
            env.model.geom_pos[i, :2] - env.model.body_pos[1, :2]
            for i in range(env.model.ngeom)
            if (env.model.geom_id2name(i) or '').startswith('wall_')
        ]), dtype=torch.float32)
        self.observation_scale = torch.tensor(
            (bounds['observations_maxs'] - bounds['observations_mins']) / 2)
        self.observation_midpoint = torch.tensor(
            (bounds['observations_maxs'] + bounds['observations_mins']) / 2)
        self.action_scale = torch.tensor((bounds['actions_maxs'] - bounds['actions_mins']) / 2)
        self.action_midpoint = torch.tensor((bounds['actions_maxs'] + bounds['actions_mins']) / 2)

    def physical_actions(self, diffusion_sample):
        """Adapter: convert the sample's first two channels to physical actions."""
        return diffusion_sample[..., :2] * self.action_scale + self.action_midpoint

    def risk(self, diffusion_sample, initial_state, goal_position):
        # Slide 13, steps 1–3: actions -> mean states and their covariance.
        # Drop the final sample row: H states correspond to H-1 executed actions.
        actions = self.physical_actions(diffusion_sample[:, :-1])
        mean_states, covariance = propagate_mean_and_covariance(
            actions, initial_state, self.noise_std, self.velocity_decay, self.action_gain)
        positions = mean_states[..., :2]
        position_variances = covariance.diagonal(dim1=-2, dim2=-1)[..., :2]

        # Slide 13, steps 4–6: local Gaussian wall and goal failure probabilities.
        wall_probabilities = wall_violation_probabilities(
            positions, position_variances, self.wall_centers)
        goal_probability = goal_violation_probability(
            positions[:, -1], position_variances[:, -1], goal_position)

        # Slide 13, steps 7–8: Delta_hat_UB = sum(delta_i,k) + delta_goal.
        # Retained indexing difference: walls include x_N; the slide sums
        # k=1..N-1. This adds terminal wall checking. Do not hide the difference.
        wall_risk = wall_probabilities.sum((1, 2))
        return RiskEstimate(wall_risk + goal_probability, wall_risk,
                            goal_probability, mean_states, position_variances)

    def energy(self, diffusion_sample, initial_state, goal_position, budget,
               gamma=DEFAULT_GAMMA):
        # Slide 14, steps 9–10: r(u) = Delta_hat_UB(u) - Delta,
        # E_risk(u) = softplus(gamma * r(u)). No extra normalization.
        residual = self.risk(diffusion_sample, initial_state, goal_position).total_risk - budget
        return functional.softplus(gamma * residual)


def sample_direct_actions(diffusion, chance_model, starts, goals, seeds, budgets,
                          strength=DEFAULT_GUIDANCE_STRENGTH):
    """Slide 14, steps 11–13: add the negative energy gradient to the score.

    This is the prior DIRECT-ACTION sampling adapter, not the nominal sampler.
    GaussianDiffusion's DDPM posterior-mean correction is
        mean_guided = mean_model - posterior_variance * lambda_beta * grad(E).
    The gradient is taken in normalized diffusion coordinates; the physical
    action conversion inside energy supplies the normalization chain rule.
    """
    batch_size = len(seeds)
    generators = [torch.Generator(device='cuda').manual_seed(seed) for seed in seeds]
    horizon, channels = diffusion.horizon, diffusion.transition_dim

    def sample_noise():
        return torch.stack([torch.randn((horizon, channels), device='cuda', generator=g)
                            for g in generators])

    def normalize_observations(observations):
        return ((observations - chance_model.observation_midpoint.numpy())
                / chance_model.observation_scale.numpy())

    conditions = {
        0: torch.tensor(normalize_observations(starts), device='cuda').float(),
        horizon - 1: torch.tensor(normalize_observations(
            np.c_[goals, np.zeros((batch_size, 2))]), device='cuda').float(),
    }
    sample = apply_conditioning(sample_noise(), conditions, diffusion.action_dim)
    initial_states = torch.tensor(starts).float()
    goal_positions = torch.tensor(goals).float()
    risk_budgets = torch.tensor(budgets).float()
    guided_candidates = torch.tensor([budget >= 0 for budget in budgets]).float()
    trace = []
    begun = time.perf_counter()

    for step in reversed(range(diffusion.n_timesteps)):
        timesteps = torch.full((batch_size,), step, device='cuda', dtype=torch.long)
        with torch.no_grad():
            mean, variance, log_variance = diffusion.p_mean_variance(sample, conditions, timesteps)

        if guided_candidates.any():
            # Slide 14 autodiff: E = compute_energy(u); E.backward(); grad_u = u.grad.
            # Here u is represented by the current noisy normalized sample,
            # not a separately optimized final path or sampled-state reference.
            differentiable_sample = sample.detach().cpu().requires_grad_(True)
            energy = chance_model.energy(
                differentiable_sample, initial_states, goal_positions, risk_budgets)
            gradient = torch.autograd.grad((energy * guided_candidates).sum(),
                                           differentiable_sample)[0]
            # Slide 14, steps 11–13: score_model - lambda * beta * grad(E_risk).
            mean = mean - strength * variance * gradient.to('cuda')
            if step % 32 == 0:
                trace.append(dict(step=step, energy=energy.detach().tolist(),
                                  gradient_norm=gradient.flatten(1).norm(dim=1).tolist()))

        with torch.no_grad():
            sample = mean + (step > 0) * (0.5 * log_variance).exp() * sample_noise()
            sample = apply_conditioning(sample, conditions, diffusion.action_dim)

    torch.cuda.synchronize()
    return sample.cpu().numpy(), time.perf_counter() - begun, trace
