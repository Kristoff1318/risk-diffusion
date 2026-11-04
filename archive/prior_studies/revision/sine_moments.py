"""Exact one-step Gaussian-input sine moments; approximate multistep closure.

No Gaussian state assumption is exact after a nonlinear update. This module
changes propagation only, retaining the historical tangent/union risk events.
It is opt-in and does not modify the historical sampler or risk default.
"""
import math
import torch


def gaussian_sine_step(mean, covariance, controls, problem):
    """Moments of X + dt*u + dt*a*sin(2*swap(X)) for Gaussian X.

    Full state/sine and sine/sine cross-covariances are included. No eigenvalue
    clipping hides instability; callers inspect PSD and numerical diagnostics.
    """
    diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1)
    damp = torch.exp(-2 * diagonal)
    sine_mean = damp * torch.sin(2 * mean)
    difference = diagonal[..., :, None] + diagonal[..., None, :] - 2 * covariance
    total = diagonal[..., :, None] + diagonal[..., None, :] + 2 * covariance
    sine_second = .5 * (
        torch.cos(2 * (mean[..., :, None] - mean[..., None, :])) * torch.exp(-2 * difference)
        - torch.cos(2 * (mean[..., :, None] + mean[..., None, :])) * torch.exp(-2 * total)
    )
    sine_covariance = sine_second - sine_mean[..., :, None] * sine_mean[..., None, :]
    state_sine = 2 * covariance * (damp * torch.cos(2 * mean))[..., None, :]
    order = torch.tensor([1, 0], device=mean.device)
    cross = state_sine.index_select(-1, order)
    swapped_covariance = sine_covariance.index_select(-2, order).index_select(-1, order)
    coefficient = problem.dt * problem.strength
    next_mean = mean + problem.dt * controls + coefficient * sine_mean.index_select(-1, order)
    eye = torch.eye(2, dtype=mean.dtype, device=mean.device)
    next_covariance = (covariance + coefficient * (cross + cross.transpose(-1, -2))
                       + coefficient**2 * swapped_covariance + problem.process_sigma**2 * eye)
    next_covariance = .5 * (next_covariance + next_covariance.transpose(-1, -2))
    return next_mean, next_covariance


def gaussian_sine_moments(controls, start, problem):
    eye = torch.eye(2, dtype=controls.dtype, device=controls.device).expand(len(controls), -1, -1)
    mean = start
    covariance = problem.initial_sigma**2 * eye
    means, covariances = [mean], [covariance]
    for step in range(problem.horizon):
        mean, covariance = gaussian_sine_step(mean, covariance, controls[:, step], problem)
        means.append(mean); covariances.append(covariance)
    return torch.stack(means, 1), torch.stack(covariances, 1)


def chance_from_moments(mean, covariance, goal, problem):
    """Historical circle tangent and terminal face event sum, unchanged."""
    obstacles = torch.tensor(problem.obstacles, dtype=mean.dtype, device=mean.device)
    difference = mean[:, :-1, None, :] - obstacles[None, None, :, :2]
    distance = difference.norm(dim=-1).clamp_min(1e-7)
    normal = difference / distance[..., None]
    variance = torch.einsum('bhoi,bhij,bhoj->bho', normal, covariance[:, :-1], normal).clamp_min(1e-12)
    obstacle = .5 * torch.erfc((distance - obstacles[None, None, :, 2]) / torch.sqrt(2 * variance))
    sigma = torch.diagonal(covariance[:, -1], dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
    goal_parts = torch.stack([
        .5 * torch.erfc((problem.goal_half - sign * (mean[:, -1] - goal)) / (math.sqrt(2) * sigma))
        for sign in (-1, 1)
    ], -1)
    return {'risk': obstacle.sum((1, 2)) + goal_parts.sum((1, 2)),
            'obstacle': obstacle, 'goal': goal_parts, 'mean': mean, 'covariance': covariance}


def chance_sine(controls, start, goal, problem):
    mean, covariance = gaussian_sine_moments(controls, start, problem)
    return chance_from_moments(mean, covariance, goal, problem)
