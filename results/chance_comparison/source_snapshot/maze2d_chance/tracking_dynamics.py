"""Slide 13 mean/covariance propagation through the actual tracking controller."""
from typing import List
import torch


@torch.jit.script
def propagate_tracking(reference: torch.Tensor, initial_state: torch.Tensor,
                       noise_std: float, velocity_decay: float, action_gain: float,
                       execution_steps: int = 800):
    """Slide 13, steps 1–3, using the original nominal controller.

    Physical action: u = clip(p_ref - p + (v_ref - v), -1, 1).
    Native free transition: v_next = a*clip(v,-5,5) + b*u;
                            p_next = p + .01*v_next.
    Independent Gaussian velocity noise is added AFTER that position update.

    Feedback is INSIDE f. Thus A=df/dx includes derivatives of the controller,
    not just those of the uncontrolled point mass. Each axis is independent;
    its 2x2 (position, velocity) covariance obeys Sigma'=A Sigma A.T+Q.
    The scalar entries below are that matrix equation written explicitly.
    Gaussian closure remains local at clipping, and does not model contact.
    """
    position, velocity = initial_state[:, :2], initial_state[:, 2:]
    position_variance = torch.zeros_like(position)
    position_velocity_covariance = torch.zeros_like(position)
    velocity_variance = torch.zeros_like(position)
    means = torch.jit.annotate(List[torch.Tensor], [])
    position_variances = torch.jit.annotate(List[torch.Tensor], [])
    velocity_variances = torch.jit.annotate(List[torch.Tensor], [])
    cross_covariances = torch.jit.annotate(List[torch.Tensor], [])
    horizon = reference.shape[1]

    for step in range(execution_steps):
        if step < horizon - 1:
            desired_position = reference[:, step + 1, :2]
            desired_velocity = reference[:, step + 1, 2:]
        else:
            desired_position = reference[:, -1, :2]
            desired_velocity = torch.zeros_like(velocity)
        command = desired_position - position + (desired_velocity - velocity)
        command_derivative = (command.abs() < 1.).to(reference.dtype)
        velocity_clip_derivative = (velocity.abs() < 5.).to(reference.dtype)

        # Slide 13 step 2: local Jacobian, INCLUDING feedback and both clips.
        dv_dp = -action_gain * command_derivative
        dv_dv = velocity_decay * velocity_clip_derivative - action_gain * command_derivative
        dp_dp = 1. + .01 * dv_dp
        dp_dv = .01 * dv_dv

        # Slide 13 step 1: mean of the executed state, not the reference path.
        velocity = velocity_decay * velocity.clamp(-5., 5.) + action_gain * command.clamp(-1., 1.)
        position = position + .01 * velocity

        # Slide 13 step 3: Sigma_next = A Sigma A.T + G Sigma_w G.T.
        next_position_variance = (dp_dp.square() * position_variance
            + 2. * dp_dp * dp_dv * position_velocity_covariance
            + dp_dv.square() * velocity_variance)
        next_cross_covariance = (dp_dp * dv_dp * position_variance
            + (dp_dp * dv_dv + dp_dv * dv_dp) * position_velocity_covariance
            + dp_dv * dv_dv * velocity_variance)
        next_velocity_variance = (dv_dp.square() * position_variance
            + 2. * dv_dp * dv_dv * position_velocity_covariance
            + dv_dv.square() * velocity_variance + noise_std * noise_std)
        position_variance = next_position_variance
        position_velocity_covariance = next_cross_covariance
        velocity_variance = next_velocity_variance
        means.append(torch.cat((position, velocity), dim=-1))
        position_variances.append(position_variance)
        velocity_variances.append(velocity_variance)
        cross_covariances.append(position_velocity_covariance)
    return (torch.stack(means, dim=1), torch.stack(position_variances, dim=1),
            torch.stack(velocity_variances, dim=1), torch.stack(cross_covariances, dim=1))
