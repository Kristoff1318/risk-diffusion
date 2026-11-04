"""Independent NumPy evaluator for stochastic control plans.

Each plan gets an independent generator seeded with ``seed + plan_index``.
This makes results reproducible while keeping plans' execution draws
independent of one another.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import beta as beta_distribution

from .problem import Problem


def _risk_interval(failures: int, draws: int, confidence: float) -> tuple[float, float, float]:
    alpha = 1.0 - confidence
    if failures == 0:
        lower = 0.0
    else:
        lower = beta_distribution.ppf(alpha / 2.0, failures, draws - failures + 1)
    if failures == draws:
        upper = 1.0
        one_sided = 1.0
    else:
        upper = beta_distribution.ppf(1.0 - alpha / 2.0, failures + 1, draws - failures)
        one_sided = beta_distribution.ppf(confidence, failures + 1, draws - failures)
    return lower, upper, one_sided


def _fractional_cvar(loss: np.ndarray, delta: float) -> float:
    ordered = np.sort(np.asarray(loss, dtype=float))[::-1]
    mass = delta * len(ordered)
    whole = int(math.floor(mass))
    frac = mass - whole
    if whole == 0:
        return float(ordered[0])
    total = float(ordered[:whole].sum())
    if frac > 0.0 and whole < len(ordered):
        total += frac * float(ordered[whole])
    return total / mass


def evaluate(controls: np.ndarray, starts: np.ndarray, goals: np.ndarray,
             problem: Problem = Problem(), draws: int = 4096, seed: int = 0,
             delta: float = .1, confidence: float = .95) -> list[dict[str, Any]]:
    """Evaluate open-loop plans under independent noisy executions."""
    controls = np.asarray(controls, dtype=float)
    starts = np.asarray(starts, dtype=float)
    goals = np.asarray(goals, dtype=float)
    if controls.ndim != 3 or controls.shape[-1] != 2:
        raise ValueError("controls must have shape (B, H, 2)")
    if controls.shape[1] != problem.horizon:
        raise ValueError("controls horizon does not match problem.horizon")
    if starts.shape != (len(controls), 2) or goals.shape != starts.shape:
        raise ValueError("starts and goals must have shape (B, 2)")
    if draws <= 0 or not (0.0 < delta <= 1.0) or not (0.0 < confidence < 1.0):
        raise ValueError("draws, delta, and confidence are out of range")

    if not all(np.isfinite(a).all() for a in (controls, starts, goals)):
        raise ValueError('Plan, start, and goal must contain finite values')

    obstacles = np.asarray(problem.obstacles, dtype=float).reshape(-1, 3)
    out: list[dict[str, Any]] = []
    for i, (u, start, goal) in enumerate(zip(controls, starts, goals)):
        rng = np.random.default_rng(seed + i)
        x = start[None, :] + problem.initial_sigma * rng.normal(size=(draws, 2))
        path = np.zeros(draws, dtype=float)
        collision_loss = np.full(draws, -np.inf)
        # The original event checks x0,...,x[H-1]; xH is reserved for goal.
        for k in range(problem.horizon):
            if obstacles.size:
                diff = x[:, None, :] - obstacles[None, :, :2]
                collision_loss = np.maximum(collision_loss,
                                             np.max(obstacles[None, :, 2] - np.linalg.norm(diff, axis=2), axis=1))
            drift = problem.strength * np.column_stack((np.sin(2.0 * x[:, 1]), np.sin(2.0 * x[:, 0])))
            previous = x
            x = x + problem.dt * (u[k] + drift) + problem.process_sigma * rng.normal(size=(draws, 2))
            path += np.linalg.norm(x - previous, axis=1)
        goal_loss = np.max(np.abs(x - goal[None, :]) - problem.goal_half, axis=1)
        loss = np.maximum(collision_loss, goal_loss)
        collided = collision_loss > 0.0
        reached = goal_loss <= 0.0
        safe = (~collided) & reached
        failures = int((~safe).sum())
        risk = failures / draws
        lower, upper, one_sided = _risk_interval(failures, draws, confidence)
        out.append({
            'risk': float(risk), 'joint_safety': float(safe.mean()),
            'collision': float(collided.mean()), 'goal': float(reached.mean()),
            'risk_upper': float(one_sided), 'risk_ci': [float(lower), float(upper)],
            'accepted': bool(one_sided <= delta and np.all(np.abs(u) <= problem.control_limit)),
            'cost': float(path.mean()), 'cvar': _fractional_cvar(loss, delta),
            'mean_positive_violation': float(np.maximum(loss, 0.0).mean()),
            'failures': failures, 'draws': int(draws),
        })
    return out

