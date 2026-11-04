"""Qualify the feedback model and slide-energy derivative before experiments."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from runtime import ASSETS, load_policy
from environment import make_environment, active_collision, ADDED_OBSTACLES
from tracking_dynamics import propagate_tracking
from chance_guidance import TrackingChanceModel, sample_guided
from nominal_baseline import sample


def main():
    torch.set_num_threads(2)
    env = make_environment()
    env.seed(1)
    start = env.reset()
    bounds = np.load(ASSETS / 'normalization.npz')
    model = TrackingChanceModel(env, bounds)
    saved = np.load(Path(__file__).parent / 'results/nominal_baseline/episode_1.npz')
    reference = torch.tensor(saved['reference'][None], dtype=torch.float64)
    predicted, _, _, _ = propagate_tracking(reference, torch.tensor(start[None]), 0.,
                                             model.velocity_decay, model.action_gain, 800)
    error = float(np.max(np.abs(predicted.numpy()[0] - saved['execution'][1:])))
    assert error < 1.e-10, error

    # Independent augmented matrix covariance versus the scalar implementation.
    stable_reference = torch.tensor([[[1., 1., 0., 0.]] * 65], dtype=torch.float64)
    stable_start = torch.tensor([[1., 1., 0., 0.]], dtype=torch.float64)
    _, pp, vv, pv = propagate_tracking(stable_reference, stable_start, .005,
                                       model.velocity_decay, model.action_gain, 64)
    a, b = model.velocity_decay, model.action_gain
    matrix = np.array([[1-.01*b, .01*(a-b)], [-b, a-b]])
    covariance = np.zeros((2, 2)); max_covariance_error = 0.
    for k in range(64):
        covariance = matrix @ covariance @ matrix.T + np.diag([0., .005**2])
        max_covariance_error = max(max_covariance_error,
            abs(pp[0,k,0].item()-covariance[0,0]),
            abs(vv[0,k,0].item()-covariance[1,1]),
            abs(pv[0,k,0].item()-covariance[0,1]))
    assert max_covariance_error < 1.e-15
    rng = np.random.RandomState(71501)
    position, velocity = np.ones((30000, 2)), np.zeros((30000, 2))
    for k in range(64):
        command = 1.-position-velocity
        velocity = a*velocity + b*command
        position = position + .01*velocity
        velocity += .005*rng.randn(30000, 2)
    empirical = np.var(position, axis=0)
    relative_covariance_error = float(np.max(abs(empirical/pp[0,-1].numpy()-1)))
    assert relative_covariance_error < .04, relative_covariance_error

    # Finite difference of the complete wall/goal union + softplus objective.
    short = TrackingChanceModel(env, bounds, execution_steps=64)
    physical = torch.tensor([[[.402, 1., 0., 0.]]*65], dtype=torch.float64)
    normalized = 2.*(physical-short.observation_minimum)/short.observation_range-1.
    z = torch.cat([torch.zeros((1,65,2), dtype=torch.float64), normalized], -1).requires_grad_(True)
    initial = torch.tensor([[.402,1.,0.,0.]], dtype=torch.float64)
    goal = torch.tensor([[.902,1.]], dtype=torch.float64)
    energy = short.energy(z,initial,goal,.1)
    derivative = torch.autograd.grad(energy.sum(),z)[0][0,1,2].item()
    eps=1.e-7;plus=z.detach().clone();minus=z.detach().clone()
    plus[0,1,2]+=eps;minus[0,1,2]-=eps
    fd=((short.energy(plus,initial,goal,.1)-short.energy(minus,initial,goal,.1))/(2*eps)).item()
    derivative_error=abs(derivative-fd)/max(abs(derivative),abs(fd),1.e-12)
    assert abs(derivative)>1.e-8 and derivative_error<1.e-4,(derivative,fd,derivative_error)

    # Added obstacles are real MuJoCo collision geometry, not plot overlays.
    added=make_environment(ADDED_OBSTACLES);added.reset()
    added.set_state(np.array([3.,4.]),np.zeros(2));added.sim.forward()
    assert active_collision(added)
    added.set_state(np.array([1.,1.]),np.zeros(2));added.sim.forward()
    assert not active_collision(added)

    # Zero guidance is the original sampler bit for bit, under identical RNG.
    policy=load_policy();env.seed(0);start=env.reset();goal=np.array(env.get_target(),dtype=float)
    original,_=sample(policy,start,goal,0)
    zero,_,_=sample_guided(policy,model,start,goal,0,.1,strength=0.)
    assert np.array_equal(zero,original.observations[0])
    begun=time.perf_counter()
    probe,seconds,trace=sample_guided(policy,model,start,goal,10,.1,guidance_steps=2)
    assert np.isfinite(probe).all() and len(trace)==2
    result=dict(passed=True,mean_native_max_error=error,
                covariance_matrix_max_error=max_covariance_error,
                covariance_empirical_relative_error=relative_covariance_error,
                full_energy_gradient_relative_error=derivative_error,
                physical_added_obstacles_checked=True,zero_guidance_exact_sampler_parity=True,
                two_guidance_stage_seconds=seconds,
                two_guidance_stage_gradient_norms=[r['gradient_norm'] for r in trace])
    output=Path(__file__).parent/'results/tracking_qualification.json'
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    env.close();added.close()


if __name__=='__main__':main()
