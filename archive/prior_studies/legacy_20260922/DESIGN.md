# Control diffusion risk pilot

## Formulation and representation

The generator emits an open-loop control array `u[0:20]` with two components per step. This retains exact consistency between the generated variable, nonlinear dynamics, covariance propagation, and pathwise risk gradients. A state-only Diffuser sample would require an inverse controller and could change its risk after execution. Initial position and goal center condition the model; the obstacle is fixed in this pilot, so the model implicitly knows its map. General obstacle conditioning is outside this experiment.

The stochastic process is `x[k+1] = x[k] + 0.2 (u[k] + d(x[k])) + w[k]`, where `d(x) = (0.12 sin(2 y), 0.12 sin(2 x))` and independent `w[k] ~ N(0, 0.025² I)`. Initial state is fixed for each task. The obstacle is a circle of radius 0.43 at the origin. Success requires avoiding it at all 20 states and landing in a square with half-width 0.30 centered at the goal. There is no feedback policy. These choices make chance propagation nontrivial while retaining cheap independent rollouts.

## Training and sampling

Ordinary feasible training paths go above the obstacle with variable clearance; inverse dynamics converts them to controls. The single route side avoids mode averaging in the small MLP. A small MLP predicts DDPM noise at 40 training noise levels. Sampling uses 25 deterministic DDIM steps. Controls are normalized by training mean and standard deviation. The nominal model is conditioned on start and goal, not on risk. Conditional training uses an equal mixture of ordinary routes and lower-clearance routes labeled by 256 independent stochastic rollouts per route. It conditions on measured risk, not requested budget; at test time the requested budget is passed as a target. Training data, model, evaluator, and experiments have separate modules.

At every guided denoising step, infer clean controls `u_hat`, apply a common nominal terminal correction, evaluate an energy on the corresponding stochastic execution model, differentiate with PyTorch, modify `u_hat`, and take the next DDIM step. The same terminal correction is applied to all variants, including B0. It adds a constant control offset to bring the nominal endpoint to the goal center; it does not enforce obstacle safety. This is a heuristic guided sampler; the resulting distribution is not an exact posterior.

## Guidance energies

**Chance:** propagate mean `mu[k]` and covariance `P[k]` with local Jacobian `A[k]`, using `P[k+1]=A[k]P[k]A[k]^T + sigma_w² I`. For the circle, linearize signed distance in the radial direction at each mean. For the terminal goal square, evaluate four Gaussian halfspace tail probabilities. Sum the 24 event probabilities by Boole's bound. The energy is `softplus(gamma (risk_UB - Delta))/gamma`, with `gamma=15`. A sum above one remains above one during optimization and is reported as such; it is a bound surrogate, not a calibrated probability.

**Sample CVaR:** run 32 reparameterized stochastic paths during each guidance step. Each path's signed loss is a smooth maximum of 20 circular collision violations and two goal-box violations. The upper-tail empirical CVaR uses the detached empirical quantile and a softplus hinge. CVaR optimizes severity in the worst `Delta` fraction; it does not directly enforce a collision-probability bound.

**Projection:** 8 Adam steps reduce deterministic obstacle/goal violation plus a proximal control penalty. It is applied every fourth denoising step in the latter half. It makes no stochastic guarantee. The B5 combination uses chance guidance and deterministic projection.

## Independent evaluator

`evaluator.py` uses NumPy rather than differentiable Torch code, with fresh random draws. It records joint success, separate collision and goal probabilities, path length, signed violation CVaR, and an exact binomial interval for each plan. The same number of draws and seed schedule apply across ablations. Confidence intervals quantify execution Monte Carlo error only; variation across tasks and training seeds is separate.

The run also records discrete nominal feasibility and mean pairwise control-array distance across sampling seeds. A projection failure is a projected plan that still violates a discrete deterministic obstacle or goal condition. Diversity is an L2 distance in controls, not a count of distinct route homotopies.

## Reproduction and scope

Run `python -m risk_diffusion.run --epochs 80 --output risk_diffusion/outputs/final` using the `diffuser` Conda interpreter documented in the repository. `--quick` is a smoke test. The script saves model checkpoints, every plan-level row, calibration cases, strength sweeps, and four plots. The reference evaluation has 12 held-out start-goal pairs, three paired sampling seeds, three risk budgets, and 2,048 execution rollouts per reported plan. Use `--radius` and `--seed` for the extra configurations listed in `RESULTS.md`.

The conventional optimizer baseline and an actual Maze2D MuJoCo execution remain useful next benchmarks. This pilot isolates whether the proposed risk quantities and sampler can control a small nonlinear stochastic system.
