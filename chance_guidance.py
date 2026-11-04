"""Slides 13–14: chance guidance for the actual nominal tracking controller.

Diffuser generates reference states [p_ref, v_ref]. The controller executes
u = p_ref - p + (v_ref - v), subject to native action/velocity clipping.
The reference therefore parameterizes the controller input; all risk is
computed on the executed mean/covariance, not on the reference positions.
Autodiff follows reference -> controller -> stochastic dynamics -> risk.
This is the explicit representation adapter from the slides' control-space
formulation to this existing state-reference diffusion checkpoint.

Slide 13, steps 1–3: tracking_dynamics.propagate_tracking.
Slide 13, steps 4–6: the Gaussian wall, circle and goal functions below.
Slide 13, steps 7–8: TrackingChanceModel.risk.
Slide 14, steps 9–10: TrackingChanceModel.energy.
Slide 14, steps 11–13: sample_guided, including the normalization chain rule.

The CDF sum is a local nonlinear-risk surrogate. Acceptance and reported risk
come from separate native MuJoCo executions, never from this number alone.
"""
import time
from typing import NamedTuple
import numpy as np
import torch
import torch.nn.functional as functional
import runtime
from diffuser.models.helpers import apply_conditioning
from tracking_dynamics import propagate_tracking


class RiskEstimate(NamedTuple):
    total_risk: torch.Tensor
    wall_risk: torch.Tensor
    goal_risk: torch.Tensor
    mean_states: torch.Tensor
    position_variances: torch.Tensor
    added_obstacle_risk: torch.Tensor


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


def circle_violation_probabilities(positions, variances, circles):
    """Slide 13 steps 4–6: added physical circles, inflated by agent radius."""
    if not circles:
        return positions.new_zeros((len(positions), positions.shape[1], 0))
    geometry = positions.new_tensor(circles)
    displacement = positions[:, :, None, :] - geometry[:, :2]
    distance = displacement.square().sum(-1).clamp(min=1.e-20).sqrt()
    gradient = displacement / distance[..., None]
    mean_margin = distance - geometry[:, 2] - .102
    variance = (gradient.square() * variances[:, :, None, :]).sum(-1)
    return gaussian_violation_probability(mean_margin, variance.clamp(min=1.e-12).sqrt())


class TrackingChanceModel:
    def __init__(self, env, bounds, obstacles=(), noise_std=.005, execution_steps=800):
        assert env.frame_skip == 1 and np.isclose(env.dt, .01)
        assert np.allclose(env.model.dof_damping, 1.)
        assert np.allclose(env.model.actuator_gear[:, 0], 100.)
        mass = env.model.body_mass[1]
        self.velocity_decay = float(mass / (mass + .01))
        self.action_gain = float(1. / (mass + .01))
        self.noise_std, self.execution_steps = noise_std, execution_steps
        self.obstacles = tuple(obstacles)
        self.wall_centers = torch.tensor(np.array([
            env.model.geom_pos[i, :2] - env.model.body_pos[1, :2]
            for i in range(env.model.ngeom)
            if (env.model.geom_id2name(i) or '').startswith('wall_')]), dtype=torch.float64)
        self.observation_minimum = torch.tensor(bounds['observations_mins'], dtype=torch.float64)
        self.observation_range = torch.tensor(
            bounds['observations_maxs'] - bounds['observations_mins'], dtype=torch.float64)

    def physical_reference(self, diffusion_sample):
        # Use the same [-1,1] -> physical transform as the checkpoint normalizer.
        return ((diffusion_sample[..., 2:].clamp(-1., 1.) + 1.) / 2.
                * self.observation_range + self.observation_minimum)

    def risk_reference(self, reference, initial_state, goal):
        # Slide 13 steps 1–3: propagation includes the actual feedback controller.
        means, position_variances, _, _ = propagate_tracking(
            reference, initial_state, self.noise_std, self.velocity_decay,
            self.action_gain, self.execution_steps)
        # Slide 13 steps 4–6: constraints evaluated on EXECUTED mean states.
        # Literal slide indexing: obstacle events x_1..x_(N-1), goal at x_N.
        # Independent execution checks also include x_0 and x_N collisions.
        wall = wall_violation_probabilities(
            means[:, :-1, :2], position_variances[:, :-1], self.wall_centers)
        added = circle_violation_probabilities(
            means[:, :-1, :2], position_variances[:, :-1], self.obstacles)
        goal_risk = goal_violation_probability(means[:, -1, :2], position_variances[:, -1], goal)
        # Slide 13 steps 7–8: sum event probabilities, without clipping at one.
        wall_risk, added_risk = wall.sum((1, 2)), added.sum((1, 2))
        return RiskEstimate(wall_risk + added_risk + goal_risk, wall_risk, goal_risk,
                            means, position_variances, added_risk)

    def risk(self, diffusion_sample, initial_state, goal):
        return self.risk_reference(self.physical_reference(diffusion_sample), initial_state, goal)

    def energy(self, diffusion_sample, initial_state, goal, budget, gamma=10.):
        # Slide 14 steps 9–10, exactly: r=Delta_hat-Delta; E=softplus(gamma*r).
        return functional.softplus(gamma * (self.risk(diffusion_sample, initial_state, goal).total_risk - budget))


def sample_guided(policy, chance_model, start, goal, seed, budget,
                  strength=.002, guidance_steps=32, gamma=10.):
    """Slide 14 score correction, paired with the ORIGINAL nominal RNG stream.

    The risk candidate is the denoiser's predicted clean reference, so the
    energy gradient follows its dependency on the current noisy sample.
    Guidance acts in the last 32 denoising stages; all 256 stages and random
    draws remain identical to nominal. No projection, endpoint optimizer,
    deterministic clearance loss, CVaR, or candidate resampling is used.
    """
    from nominal_baseline import set_seed
    diffusion = policy.diffusion_model
    set_seed(seed)
    conditions = policy._format_conditions(
        {0: start.copy(), diffusion.horizon - 1: np.r_[goal, 0., 0.]}, 1)
    sample = torch.randn((1, diffusion.horizon, diffusion.transition_dim), device='cuda')
    sample = apply_conditioning(sample, conditions, diffusion.action_dim)
    initial_state = torch.tensor(start[None], dtype=torch.float64)
    goal_tensor = torch.tensor(goal[None], dtype=torch.float64)
    trace = []
    torch.cuda.synchronize()
    begun = time.perf_counter()
    for step in reversed(range(diffusion.n_timesteps)):
        timesteps = torch.full((1,), step, device='cuda', dtype=torch.long)
        if strength and step < guidance_steps:
            variable = sample.detach().requires_grad_(True)
            prediction = diffusion.model(variable, conditions, timesteps)
            clean_reference = diffusion.predict_start_from_noise(variable, timesteps, prediction).clamp(-1., 1.)
            clean_reference = apply_conditioning(clean_reference.clone(), conditions, diffusion.action_dim)
            mean, variance, log_variance = diffusion.q_posterior(clean_reference, variable, timesteps)
            # Slide 14 autodiff through the model, unit conversion, controller,
            # mean/covariance, constraint linearization, CDF and union sum.
            energy = chance_model.energy(clean_reference.cpu().double(), initial_state,
                                         goal_tensor, budget, gamma)
            gradient = torch.autograd.grad(energy.sum(), variable)[0]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError('Nonfinite chance-energy gradient')
            # Slide 14 steps 11–13: s_guided=s_model-lambda*beta*grad(E).
            mean = mean.detach() - strength * variance.detach() * gradient
            log_variance = log_variance.detach()
            trace.append(dict(step=step, energy=float(energy.detach()),
                              gradient_norm=float(gradient.norm())))
        else:
            with torch.no_grad():
                mean, _, log_variance = diffusion.p_mean_variance(sample, conditions, timesteps)
        with torch.no_grad():
            # Exactly the same torch.randn_like call as the original sampler.
            noise = torch.randn_like(sample)
            sample = mean + (step > 0) * (.5 * log_variance).exp() * noise
            sample = apply_conditioning(sample, conditions, diffusion.action_dim)
    torch.cuda.synchronize()
    physical_reference = policy.normalizer.unnormalize(sample.cpu().numpy()[0, :, 2:], 'observations')
    return physical_reference, time.perf_counter() - begun, trace
