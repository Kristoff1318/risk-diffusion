# Reassessment after inspecting actual trajectories

This review supersedes the broad interpretations in the initial verdict. It does not change the planner implementation. Examples come from saved controls at Delta=0.10, with 8,192 fresh independent executions per displayed plan. The examples are deliberately selected to show an already-safe plan, a repair, and a remaining failure; they are not representative frequencies.

## What the trajectories show

| Case | Nominal failure | Chance failure | CVaR failure | Deterministic refinement failure |
|---|---:|---:|---:|---:|
| Already safe (seed 0, task 1) | 1.54% | 1.54% | 1.54% | 1.51% |
| Repair (seed 0, task 3) | 99.74% | 2.44% | 2.49% | 1.46% |
| Remaining failure (seed 2, task 1) | 100% | 26.43% | 74.21% | 1.92% |

In the remaining failure, the chance surrogate predicts 38.84% risk, already above the requested 10%. Its output is nevertheless returned as a plan. This example is an optimization/acceptance failure; it is not evidence that the surrogate falsely declared safety. The already-safe example fails only because of terminal goal misses. Across the original reference B4 plans, essentially all residual failure is terminal goal failure. The roughly 1.5% observed level is therefore a goal-limited level in these runs, not a proven minimum attainable risk.

All training routes take the upper passage. Several sampled nominal routes instead go through or below the obstacle, and some guided plans remain jagged. The prior has not learned a reliable nominal planning distribution. At Delta=.10, only 21/36 nominal plans even pass the discrete deterministic feasibility check. Calling this a sufficient nominal baseline was premature. The earlier claim that the original two-route failure was caused by mode averaging was a hypothesis, not an experimentally established diagnosis.

## Corrections to the earlier interpretation

1. **A risk budget is an inequality.** A plan with risk 2% satisfies a 10% bound. Realized risk need not equal Delta, and relaxing a constraint need not change an optimum if the constraint is inactive. The unresolved question is whether a cheaper acceptable plan exists and the method can exploit the relaxed budget. Our guidance has no explicit path-length or control-effort objective. Therefore the weak risk-versus-Delta response alone does not establish failure of risk-bounded planning. The actual returned unsafe plans and absence of a validated acceptance procedure are stronger concerns.
2. **Risk-conditioned training was not a fair test.** Nominal training used 2,500 routes and 80 epochs; conditional training used 1,600 routes, 40 epochs, different examples, and a different initialization. There are roughly 1,600 versus 520 optimizer steps. Dropping risk conditioning by replacing it with zero conflates an unconditional example with a legitimate zero-risk label; sampling also does not implement classifier-free guidance. Requested budget and measured risk have different meanings. The negative result does not establish that risk-aware training is unnecessary.
3. **B4 is iterative penalty refinement, not a constrained projection.** It takes Adam steps on a soft deterministic penalty with a proximity term. It neither solves a specified projection problem to tolerance nor enforces a stochastic constraint. Different optimizers, step sizes, and work allocations confound the comparison with gradient guidance. Its success is useful, but it does not establish superiority of projection as a general method.
4. **The chance surrogate is not a calibrated probability.** For a circular obstacle, the local tangent halfspace contains the collision disk, so its Gaussian probability is already geometrically conservative. Summing correlated events adds further conservatism. Nonlinear covariance propagation introduces a different approximation that can go either way. The previous report could not attribute the positive bias specifically to the union bound because it did not isolate these components. An upper bound need not lie on the identity line; its usefulness depends on soundness and tightness.

## Concrete implementation and design issues

### The diffusion prior needs to pass its own gate

The terminal diffusion cumulative alpha is 0.04214, leaving a data coefficient sqrt(alpha)=0.205 at the noisiest training level, whereas sampling starts from pure Gaussian noise. This is a measurable train/sample mismatch and a plausible contributor, not a demonstrated sole cause. The phenomenon is discussed in [Lin et al., Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891). Endpoint correction also alters every predicted clean control array using repeated detached control shifts. Its interaction with guidance and final clipping is unmeasured. Denoising loss alone does not validate a generative planner.

Before further risk experiments, compare generated and held-out training distributions for collision, goal error, roughness, control limits, clearance, and route side. Inspect samples before and after endpoint correction. Use a consistent nearly zero-terminal-SNR schedule with an appropriate parameterization, and train to held-out sample quality. A richer denoiser is an option only if this still fails. A simple fitted waypoint/control-basis distribution is an essential sanity baseline for this nearly low-dimensional data family.

### The chance energy keeps pushing safe plans

For E(R)=softplus(gamma(R-Delta))/gamma, dE/dR=sigmoid(gamma(R-Delta)). At R=.02, Delta=.10, gamma=15, this derivative is about .23. Thus the risk penalty remains appreciably active well inside the safe region. With no competing cost, excess clearance is expected. Moreover, every step applies the same correction strength to a detached predicted clean sample, regardless of denoising uncertainty. The sampler stops after a fixed number of corrections even if the final risk remains above budget. Final endpoint correction and clipping can modify a previously improved plan.

A better target is minimize J(u), subject to a risk surrogate R(u)<=Delta, with a proximity/trust-region term preserving the diffusion proposal. Use an adaptive dual weight or an explicit constrained refinement to enforce the budget. Recheck all constraints after every final modification. Rejection or an explicit failure result is required when no candidate passes. A smaller risk than Delta is acceptable; compare cost at the same bound.

### The CVaR experiment contains large smoothing and threshold effects

The code takes log-sum-exp over 22 signed violations at beta=18. Its upward difference from hard max can reach log(22)/18=0.172 in distance units. The softplus hinge adds at most log(2)/(18 Delta), which is 0.385 at Delta=.10 and 0.770 at Delta=.05. These are worst-case bounds, not measured offsets, but they are large relative to the obstacle radius .43 and goal half-width .30.

The threshold eta is a hard empirical quantile of smoothed losses, then detached. That is not generally the minimizing threshold of the softplus CVaR objective, so the usual envelope argument does not justify the computed gradient of that minimized objective. K=32 also supplies only 1.6 nominal tail observations at Delta=.05. Fresh noise at every update adds optimization variance.

The principled alternative is the Rockafellar–Uryasev objective eta + E[(L-eta)+]/Delta, using exact trajectory max/hinge subgradients initially, or jointly optimizing eta with carefully checked smooth approximations. Use fixed reparameterized samples over a local optimization block and independent validation samples. Compare K and smoothing convergence against the hard loss. True upper-tail CVaR_Delta(L)<=0 is a sufficient condition for P(L>0)<=Delta under the usual quantile definition; simply reducing a finite-sample smoothed loss provides no such guarantee. See [Rockafellar and Uryasev](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf).

### The experiment does not isolate the source of the gain

The obstacle sits in an unbounded plane; going farther around it is cheap and unobstructed. Small additive noise makes deterministic clearance highly effective. There is no conventional optimizer from the same proposal, no optimizer from a simple waypoint prior, and no matched-compute final-only refinement. Therefore we have not established that diffusion is necessary or that intervening during denoising is better than improving the final sample.

Circle-size variants do not supply qualitatively different planning geometry. Calibration routes are smooth waypoint routes, whereas actual generated routes can be jagged. Calibration needs to cover the planner's output distribution, near-boundary candidates, and guidance iterates. Discrete-time safety is legitimate for the stated model, but it should not be described as continuous-path collision safety.

## What I would do next, before enlarging the framework

1. **Repair and validate the nominal generator.** Set a held-out nominal feasibility gate and demonstrate reasonable control smoothness. Retain both route sides once the prior can model them. Compare against a direct waypoint sampler to see what diffusion contributes.
2. **Separate feasibility from cost.** Define J explicitly, including path length/control effort and smoothness as appropriate. Compare the same candidates under chance-constrained and CVaR-constrained refinement. Return failure if final independent validation cannot support the requested bound.
3. **Use a diagnostic baseline matrix.** Compare a simple proposal and diffusion proposal, each with final-only refinement and correction during denoising, at matched compute. Add a deterministic clearance optimizer and an ordinary stochastic optimizer. This distinguishes prior quality, risk estimation, and optimization timing.
4. **Decompose risk error.** On identical trajectories, compare exact event Monte Carlo, Monte Carlo tangent-halfspace events, sums of Monte Carlo marginals, and locally propagated Gaussian probabilities. This separates geometric conservatism, temporal correlation, and moment propagation errors.
5. **Test a real risk/cost decision.** Use a short narrow route and a longer wide route, with enough noise that deterministic feasibility is insufficient. With paired seeds and several Delta values, evaluate cost, per-plan bound compliance, rejection rate, and diversity of accepted plans. Do not require risk to equal Delta.
6. **Revisit conditional training only after those comparisons.** Match data, optimizer steps, normalization, and seeds, use an explicit missing-condition mask, and distinguish conditioning on measured risk from conditioning on an upper budget. The current B6/B7 result is inconclusive.

Use fresh execution draws for final validation, report an upper confidence bound rather than only a point estimate, and avoid repeatedly selecting against the same validation set. For an accepted batch, state whether confidence is per-plan or simultaneous. For the reference Delta=.10 results, applying the already stored 95% interval changes CVaR passing plans from 28/36 to 27/36; B2 remains 35/36 and B4 36/36. These intervals still assume the specified dynamics/noise model is correct.

The most defensible next design is a validated nominal proposal followed by a genuine stochastic constrained optimizer, with diffusion-time corrections retained only if they improve feasibility or runtime against an equally funded final-refinement baseline. The present data do not yet select a winning risk surrogate or rule out risk-conditioned learning.
