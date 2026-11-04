# Literature audit for risk bounded diffusion planning

This audit is scoped to the supplied deck, the pasted project brief, and the
current `risk_diffusion/revision/` implementation. The goal is to identify what is
already established, what can be transferred directly, and where an
experiment can make a useful comparison. The project should claim an empirical
comparison of risk surrogates for diffusion sampling, not a new general theory
of chance constrained planning.

## Diffusion trajectory priors and guidance

* **Diffuser.** Janner, Du, Tenenbaum, and Levine, *Planning with Diffusion
  for Flexible Behavior Synthesis*, ICML 2022,
  [PMLR page](https://proceedings.mlr.press/v162/janner22a.html),
  [paper PDF](https://proceedings.mlr.press/v162/janner22a/janner22a.pdf).
  Diffuser establishes the trajectory level denoising formulation, state action
  sequence representation, test time objective guidance, and inpainting. It is
  the direct source for the nominal prior and guidance interface. It does not
  provide stochastic execution risk calibration.

* **SafeDiffuser.** Xiao et al., *SafeDiffuser: Safe Planning with Diffusion
  Probabilistic Models*, ICLR 2025,
  [OpenReview paper](https://openreview.net/pdf?id=ig2wk7kK9J),
  [project page](https://safediffuser.github.io/safediffuser/),
  [arXiv record](https://arxiv.org/abs/2306.00148). It inserts finite time
  control barrier constraints into denoising and evaluates Maze2D and other
  control tasks. This overlaps with the proposed projection and safety
  guidance theme. Its safety constraints are barrier based and its guarantees
  are different from an empirical probability of collision under stochastic
  execution. It should be a related baseline or cited boundary, not described
  as an unoccupied problem.

* **Risk Guided Diffusion.** Thakker et al., *Risk-Guided Diffusion: Toward
  Deploying Robot Foundation Models in Space, Where Failure Is Not an Option*,
  2025 preprint, [arXiv record](https://arxiv.org/abs/2506.17601). The paper
  combines a learned proposal with a physics based risk module at inference
  time. This is close in system architecture to nominal diffusion plus a
  model based risk correction. The present project differs in its controlled
  comparison of local Gaussian chance risk and sampled CVaR, but should avoid
  claiming inference time risk guidance itself as novel.

* **Constrained Diffusers.** *Constrained Diffusers for Safe Planning and
  Control*, 2025 preprint, [arXiv record](https://arxiv.org/abs/2506.12544),
  [OpenReview PDF](https://openreview.net/pdf?id=tahkGZjjWA). It applies
  projected, primal dual, and augmented Lagrangian updates during reverse
  diffusion. This directly motivates an ablation of projection, while leaving
  stochastic execution risk and calibration as separate questions.

* **Motion Planning Diffusion.** *Motion Planning Diffusion: Learning and
  Adapting Robot Motion Planning with Diffusion Models*, 2024 preprint,
  [arXiv record](https://arxiv.org/abs/2412.19948). It is relevant for learned
  trajectory priors and cost guidance in robot motion planning. It does not
  replace the independent stochastic evaluator required here.

## Chance constraints and nonlinear stochastic planning

* **Linear Gaussian chance constraints.** Blackmore, Ono, and Williams,
  *Chance-Constrained Optimal Path Planning with Obstacles*, IEEE Transactions
  on Robotics 27(6), 2011, [MIT open manuscript](https://dspace.mit.edu/server/api/core/bitstreams/1bcd431a-3fdd-4735-a689-2ea5d1f2b5d0/content),
  [DOI](https://doi.org/10.1109/TRO.2011.2161160). The paper propagates
  linear Gaussian state uncertainty and converts obstacle chance constraints
  into tractable approximations. It is the correct ancestor for the deck's
  local linearization, Gaussian CDF, and risk allocation ideas. Its assumptions
  do not justify treating the proposed nonlinear stochastic rollout as
  calibrated without Monte Carlo checks.

* **Non Gaussian nonlinear risk bounded planning.** Han, Jasour, and Williams,
  *Non-Gaussian Risk Bounded Trajectory Optimization for Stochastic Nonlinear
  Systems in Uncertain Environments*, ICRA 2022,
  [arXiv record](https://arxiv.org/abs/2203.03038),
  [author manuscript](https://cap.csail.mit.edu/sites/default/files/research-pdfs/Non-Gaussian%20Risk%20Bounded%20Trajectory%20Optimization%20for%20Stochastic%20Nonlinear%20Systems%20in%20Uncertain%20Environments.pdf).
  It converts stochastic dynamics and chance constraints into deterministic
  constraints on moments and optimizes controls. This is the closest prior
  work to the proposed nonlinear setting. The present framework uses a
  diffusion proposal and compares its local risk surrogate with sampled CVaR;
  that combination is an experimental integration, not a new chance
  constraint derivation.

* **Probabilistic planning under bounded risk.** Ono, Williams, and Blackmore,
  *Probabilistic Planning for Continuous Dynamic Systems under Bounded Risk*,
  [arXiv record](https://arxiv.org/abs/1402.0579). This work gives the broader
  risk bounded planning context and risk allocation language. The current
  project does not need its temporal qualitative state plan machinery.

## CVaR and tail risk

Rockafellar and Uryasev, *Optimization of Conditional Value-at-Risk*, The
Journal of Risk 2(3), 2000, [author PDF](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf),
[DOI](https://doi.org/10.21314/JOR.2000.038). Their auxiliary threshold
formulation is the reference for the sampled CVaR objective. In this project,
the loss must be defined at the trajectory level and the empirical quantile
must be reported with the number of stochastic rollouts. A smooth max or hinge
changes the loss being optimized, so the experiment should report both the
surrogate and the independent hard violation rate.

## Consequences for this project

1. Use Diffuser style trajectory denoising and guidance as the prior and
   sampler interface.
2. Treat local Gaussian chance risk as a deliberately limited approximation
   inherited from linear Gaussian chance constrained planning. Compare its
   calibration and false safe decisions against a held out Monte Carlo
   evaluator.
3. Treat sampled CVaR as a tail loss, not as a probability constraint. Population upper-tail CVaR at tail mass delta <= 0 is sufficient for the chance condition with this signed loss; a finite sampled guidance value alone does not establish it.
4. Describe projection as a constrained sampling update and evaluate whether
   it improves actual stochastic safety. Deterministic post projection
   feasibility is an intermediate metric.
5. Position the contribution as a reproducible ablation and calibration study
   on a controlled stochastic 2-D system. Avoid “first risk bounded diffusion
   planner” language because SafeDiffuser, Risk-Guided Diffusion, and
   Constrained Diffusers already occupy adjacent or overlapping territory.


## Additional original-slide references checked in the revision

- [Diffusion Policies for Generative Modeling of Spacecraft Trajectories](https://arxiv.org/abs/2501.00915) is the spacecraft diffusion paper referenced by the supplied slides. It supplies relevant precedent for generative trajectory modeling; it should not be silently substituted with the different Diffuser paper. This revision checked the abstract and bibliographic record, not a full independent reproduction.
- [Convex Risk-Bounded Continuous-Time Trajectory Planning and Tube Design in Uncertain Nonlinear Systems](https://arxiv.org/abs/2305.17291), Jasour, Han, and Williams (2023), addresses continuous-time risk bounds using risk contours and polynomial optimization. Its guarantee machinery differs from our discrete-time local Gaussian approximation. The record/abstract was checked; we do not claim to have reproduced its algorithms.

## Current empirical position

The revised comparison uses ordinary diffusion, Gaussian chance guidance, exact empirical CVaR guidance, constrained projection, matched conditioning, and a conventional chance-constrained optimizer. Each component has precedent. The nonlinear calibration counterexample is useful diagnostic evidence, but by itself is not a new robotics algorithm or a conference-level contribution. Prior DPCC execution-prefix questions remain out of scope.
