# Risk-bounded control diffusion: current design

The authoritative requirements are preserved verbatim in [ORIGINAL_PLAN.txt](ORIGINAL_PLAN.txt). [PLAN_STATUS.md](PLAN_STATUS.md) maps the work to the original stages. This revision follows the original sequence: validate nominal diffusion, independent stochastic execution evaluation, analytical chance guidance, sampled CVaR guidance, projection, then matched risk-conditioned training. Conventional optimization is an additional comparator. The initial pilot and its interpretation are preserved under `legacy_20260922/` and `inspection_20260926/`.

## Exact problem and representation

The generator outputs open-loop controls u[0:H], H=20, with two components and |u|<=2.5. States follow x[k+1]=x[k]+0.2(u[k]+a[sin(2y[k]),sin(2x[k])])+w[k]. The initial distribution is N(start, .015² I), and w[k] are independent N(0, sigma² I). The safe event follows the original indexing exactly: every x0,...,x[H-1] avoids all specified circles, and xH lies in the goal box. Boundary contact is nonviolating, so the hard signed trajectory loss L<=0 is precisely safe-and-goal. Control bounds are checked separately as deterministic feasibility conditions. Continuous-time safety between samples is not claimed.

`revision/problem.py` records geometry, process noise, initial uncertainty, goal width, and dynamics. Main configurations are a central circle with low/moderate noise, a staggered second obstacle, and stronger nonlinear drift. Models are trained separately for changed geometry/dynamics, so the map is implicit in each checkpoint; this is not a model claiming arbitrary map generalization.

Controls preserve dynamics consistency: every state used by guidance or evaluation comes from executing the same control sequence. No state-path checkpoint, temporal logic, dwell requirement, or FULL-versus-PREFIX DPCC machinery is introduced. Prior reuse decisions remain in [AUDIT.md](AUDIT.md).

## Nominal baseline gate

`revision/nominal.py` trains a risk-agnostic MLP control DDPM on ordinary feasible upper and lower waypoint routes, converted to controls through inverse nominal dynamics. The network uses sinusoidal time features, v prediction, a 128-step cosine noise schedule with terminal cumulative alpha 1.48e-7, per-time control normalization, and an EMA checkpoint. Sampling uses 64 DDIM steps. B0 has no endpoint correction, clipping, or obstacle repair.

The prespecified nominal gate is >=95% joint deterministic feasibility on 512 held-out samples, with generated route-side frequency within 0.10 of the held-out training-generator frequency. Report collision, goal, control, endpoint error, roughness, and mode coverage separately. Three training seeds pass at 99.0–99.2%; staggered and stronger nonlinear checkpoints pass at 98.4% and 98.0%. The simple waypoint generator is retained as a nominal sanity baseline.

## Original chance guidance during denoising

At each eligible denoising step, recover clean controls, propagate mean and covariance with the local dynamics Jacobian, and approximate circle collision events by their containing tangent halfspaces. Goal-box events are four halfspaces. Sum the Gaussian tail probabilities over all events without clipping the sum at one. The energy is softplus(gamma(R_UB-Delta))/gamma. Autodiff differentiates the energy through both stochastic-moment propagation and the clean-control prediction back to noisy controls.

For v prediction, eps=sqrt(1-alpha)z+sqrt(alpha)v. Implement the requested score modification s_guided=s_theta-lambda grad_z E via eps_guided=eps+lambda sqrt(1-alpha)grad_z E, then perform the corresponding DDIM step. Guidance starts when alpha>=0.5 to avoid relying on severely uncertain clean predictions. The working defaults are lambda=8 and gamma=30. Separate tuning tasks characterize sensitivity; the main experiment uses paired sampling seeds and a fixed configuration. These runs are an exploratory pilot, not a preregistered confirmatory study.

The tangent approximation and union sum can be conservative for Gaussian states, but local nonlinear propagation can underestimate risk. `revision/calibration.py` separately estimates joint failure, sums of true-event marginals, sums of tangent-event marginals, and the local Gaussian sum. It records mean/covariance errors and covariance eigenvalues. Surrogate risk is never reported as a validated probability bound.

## Original sampled CVaR alternative

For K=256 reparameterized rollouts, define L as the hard maximum of signed circle and terminal-goal violations. The empirical upper-tail CVaR is the exact fractional average of the worst Delta*K samples. This equals the minimized sampled Rockafellar–Uryasev formulation, including fractional tail mass. Sorting supplies piecewise gradients; no soft-max bias or detached approximate threshold is used. Consequently no beta smoothing parameter is present in the revised implementation. A separate RNG produces fixed rollout samples over one denoising process; evaluation samples are independent. Count both unique rollouts and all repeated rollout evaluations.

CVaR guidance changes the diffusion score through the same interface as chance guidance. A true CVaR<=0 constraint is sufficient for a chance bound under the modeled loss distribution; finite sampled guidance alone is not a guarantee. K and lambda sweeps and independent rejection are therefore retained.

## Projection and supplemental optimization

B4 solves a local constrained nearest-control problem using SLSQP: minimize normalized squared displacement from the proposed clean control sequence, subject to nominal obstacle/goal inequalities and control bounds. It is applied in the last three designated denoising stages and fed into subsequent denoising. The clean-control estimate is projected while the current epsilon estimate is retained in the next DDIM update; this is a heuristic inpainting rule, with no exact posterior claim. Every projection records constraint residuals, solver status, and failure. This is a local projection in a nonconvex set, not a claim of global closest-point optimality. B5 combines chance guidance with the same projection. A final-only projection comparator isolates the timing of correction.

A conventional SLSQP chance-constrained cost optimizer is tested from both a simple waypoint initialization and a diffusion initialization. Its explicit cost is path length + .02*dt*sum(u²) + .01*sum((u[k+1]-u[k])²). Cost-aware score guidance is also swept. These additions test the value of diffusion-time correction without replacing it. The main B0–B5 experiments first isolate risk corrections with zero explicit cost weight.

## Risk-conditioned training after the inference pilots

B6/B7 use exactly matched nominal/conditional models: same examples, 4,000 updates, batch size, architecture, initialization, optimizer, and training-noise streams. The missing-condition mask is explicit. Conditional dropout uses its own RNG. Optional classifier-free sampling combines conditional and unconditional v predictions; reported runs use scale=1.

Training examples are independently labeled using 512 stochastic executions. Each receives a requested budget from a fixed grid at least as large as its one-sided risk upper label. Thus the input means an upper budget, rather than silently substituting a requested budget for measured risk. Labels remain finite-sample estimates. Compare ordinary feasible data alone and an equal mixture of ordinary and marginal-clearance feasible data. Dedicated risk-planner data are not required for these pilot tests.

## Independent execution and acceptance

`revision/evaluate.py` implements the dynamics and hard events independently in NumPy. It reports joint safety, collision, goal success, executed length, fractional hard CVaR, violation magnitude, and exact binomial intervals. Acceptance uses a one-sided 95% upper failure bound, together with control feasibility. Experiments use 4,096 validation draws and 8,192 independent audit draws. Audit draws never influence selection.

`revision/planner.py` returns controls only when a candidate passes. For selection among a fixed batch, it adjusts confidence with Bonferroni to cover the batch simultaneously; if none passes, it returns `controls=None` and an explicit failure. Report proposal risk, acceptance rate, and accepted-plan audit risk separately. These statistical claims are conditional on the specified dynamics/noise model. Planning timing includes validation; the larger audit is research overhead.
