# Phase one: preservation, baseline audit, and staged improvement

Date: 2026-10-02. Scope: risk-bounded diffusion for stochastic nonlinear systems. This phase adds a bounded CPU reproduction harness and this experiment plan; it changes no sampler, risk model, evaluator, training code, or historical results.

## Preservation and boundary

Project-only Git repository: `/home/kristoff/Documents/experiments/research/risk_diffusion`.
Baseline commit: `4785062735660ef04a2e79b89e476057c197aed2`.
Baseline tag: `baseline-2026-10-02`.
Improvement branch: `improve/nonlinear-risk-calibration`.
The pre-Git archive and read-only file manifest are under `/home/kristoff/Documents/Codex/2026-10-02/task/risk_diffusion_preservation/`.
The archive includes all 225 original files, including historical results/checkpoints and bytecode. The tracked preservation manifest records every original file's SHA-256 and byte count and the archive hash. The archive is 23,354,166 bytes. Files remain in their original locations. The snapshot and manifest have read-only mode bits; these are tamper-evident local preservation, not externally enforced write-once storage.

There was no enclosing Git repository or prior Git change history. No project symlinks, credentials filenames, `.env` files, private keys, AWS access-key patterns, GitHub-token patterns, or OpenAI-key patterns were found by the scoped inspection (not a proof that arbitrary secrets cannot exist). Git uses the existing configured author, Kristoff Misquitta; no identity was invented. Git ignores historical output trees, generated binaries, environments, bytecode, and new phase-one output folders. Sibling projects and their environments are outside this repository. No applicable AGENTS.md was found in this project's ancestors or subtree. No specialized skill is needed for this local Python/Git scientific audit.

## Completed reproduction and evidence

Run `phase1_smoke.py` with the existing diffuser Python environment, `PYTHONDONTWRITEBYTECODE=1`, CUDA hidden, scientific thread counts capped at two, and the workspace parent on PYTHONPATH. The harness requires a new output directory and refuses to overwrite one. It uses no training, downloads, or installs. New phase-one status/log/results are under `phase1_outputs/smoke_20261002`; a copy of the small evidence JSON is tracked beside this document.

- Existing mathematical checks passed: linear covariance, exact fractional empirical CVaR and gradients, chance-energy finite differences, and independent evaluator/binomial bounds.
- Twelve saved-plan replay cases: B0/B2/B3, training seed 0, moderate noise, Delta=.10, task seed 2901, task indices 0/3/11/23. The historical 4,096 validation draws (seed 75100+task) and 8,192 audit draws (seed 198100+task) reproduce failure counts, audit risks, and acceptance exactly.
- Existing checkpoint sampled on CPU: four tasks, B0/B2/B3, 64 denoising steps, lambda=8, K=256, matched initial diffusion seed 40101. Validation/audit use distinct new seed bases 202610020/202610120. CPU random streams are not expected to reproduce CUDA generation bitwise. This is an implementation smoke, not evidence for a changed method or reliable method ranking.
- All three fresh variants accepted 4/4. Mean independent audit failure risk: B0 3.033%, B2 3.021%, B3 2.969%; nominal J: 3.939/3.940/3.947. These easy four cases do not measure improvement. Sampling times for a batch of four: .019/.116/.138 seconds on this CPU; not comparable to historical GPU batch timings.
- Raw B0 equivalence, public planner's accepted/rejected result semantics, and nominal projection feasibility/idempotence passed on CPU.
- The preserved nonlinear counterexample reproduced 7,589 failures in 65,536 draws (11.5799%), CI [11.3359%,11.8273%], predicted Gaussian sum 9.9403%, and rejection. Decomposition at its separate 32,768 draws reproduced joint risk 11.4471%, exact marginal sum 11.8744%, tangent marginal sum 12.5580%.
- After execution, all 225 original file hashes still matched. Existing 91-entry September 26 manifest also matched before this phase.

The harness body took 1.282 seconds; output was approximately 35 KB. The exact snapshot contributes approximately 23.4 MB of new artifacts. No GPU was used, and neither the 600 aggregate command-second ceiling nor 500 MB artifact ceiling was approached. This phase records actual smoke elapsed time; historical training timings are not new compute. No substantial training or sweep was launched.

## Audit: what the current energy means

Current dynamics: x[k+1] = x[k] + dt*(u[k] + a*[sin(2y[k]), sin(2x[k])]) + w[k]. Controls, propagated moments, nominal states, and independent rollouts use the same model. Initial and per-step Gaussian execution noise are physical model assumptions; diffusion-proposal spread is not execution uncertainty.

`revision/risk.py` propagates a local deterministic mean and first-order covariance A P A^T + Q. It replaces circle events with containing tangent halfspaces, sums per-time/per-obstacle Gaussian event marginals and four terminal goal-face marginals, then uses softplus(gamma*(sum-Delta))/gamma as guidance energy. Full mean/covariance derivatives reach noisy controls through the clean-control estimate. All-event summation can exceed one: it is an energy surrogate, not a calibrated probability. Cross-time/event correlation is discarded by union aggregation, although within-state covariance is retained. Near-coincident obstacle center and mean can degenerate the tangent direction; document and probe this case rather than infer geometric conservatism unconditionally.

The counterexample's union aggregation adds .427 percentage points and tangent geometry adds .684 points; approximate Gaussian propagation removes 2.618 points relative to empirical tangent marginals. Thus tightening the union sum alone could worsen false-safe predictions. Mean error reaches .0369 and terminal covariance error .00357. Improved nonlinear moments must be examined before reducing conservative event accounting.

Softplus still supplies some gradient inside the feasible region. The default isolates risk with zero explicit cost weight, so lowering risk does not demonstrate a cost-optimal use of Delta. Current CVaR is exact for its finite empirical tail, not for the unknown true execution tail. Independent validation remains necessary for both energies.

Original event indexing checks obstacle avoidance at x0,...,x[H-1] and the goal at xH. Keep this exactly for baseline comparisons; do not silently claim final-state collision or continuous-time safety. Any stronger endpoint or swept-volume semantics require a named separate benchmark and reevaluation of all baselines.

## First scientific experiment: Gaussian nonlinear moment closure

Hypothesis: replacing local Jacobian propagation by exact one-step Gaussian moments of the sine dynamics reduces marginal-risk underestimation at nonlinear boundaries without the rollout cost of sampled CVaR. This is a Gaussian assumed-density approximation, not a nonlinear chance guarantee.

Implement as an opt-in separate propagation function; preserve the original default and checkpoint. For a Gaussian state, E[sin(2Y)] = exp(-2 Pyy)*sin(2 my). Cov(X,sin(2Y)) = 2 Pxy*exp(-2 Pyy)*cos(2 my). Use the full analytic sine cross moments:

E[sin(2X)sin(2Y)] = .5*[cos(2(mx-my))*exp(-2(Pxx+Pyy-2Pxy)) - cos(2(mx+my))*exp(-2(Pxx+Pyy+2Pxy))].

Apply these to both mean and covariance of the complete discrete update, including cross terms and process Q; symmetrize only for roundoff and report any PSD correction. The Gaussian closure is reapplied each step, so multimodal/non-Gaussian trajectory effects remain. No gradient should be detached.

First evaluate frozen control banks only: current local Gaussian, analytic sine closure, full-rollout empirical marginal reference, and fixed-sample CVaR reference. Reproduce the old failure as a diagnostic (never a held-out proof). Include linear/noise-zero special cases and finite-difference gradient checks. Match geometry and event semantics. Measure state mean/covariance error versus time, marginal event-probability error, signed underestimation, false-safe surrogate decisions and calibration at .05/.10/.20, CPU runtime and peak memory. Retain risk >1 outputs as sums, with clearly named fields. Finite MC confidence intervals accompany event estimates.

Before sampling or optimizing new controls, freeze distinct task sets: 24 tuning tasks from dataset seed 2026100301, 32 evaluation tasks from seed 2026100401; matched noise/sampling seed banks are fixed in the protocol JSON. The old counterexample and existing September 26 tasks are diagnostic sets, not independent held-out evaluation. Do not use evaluation audit draws to tune energy, select proposals, or set thresholds.

Initial estimated resource envelope: 30-120 CPU seconds, <50 MB new artifacts, no new model training and no GPU needed. This estimate must be checked by a small timer before broadening. Success means reduced held-out boundary/marginal error with measured cost; one repaired example is insufficient. If closure does not help, retain and report the negative result.

## Subsequent stages and gates

1. **Closed-loop decision experiment on the same controls-to-state model.** Compare nominal B0, original chance B2, new closure guidance, sampled CVaR B3, and conventional SLSQP initialized from waypoint and diffusion proposals. Use the same checkpoints, task conditions, proposal count, validation/audit draw counts, and correction budgets. Add an explicit cost sweep chosen on tuning tasks. Report failures and rejected candidates, total wall time including validation, surrogate calls, differentiable rollout counts, accepted-plan J, and fresh audit risk. Match operating points or show Pareto frontiers rather than equate risk budgets with equal realized risk. Keep final-only refinement as a comparator to isolate denoising-time correction. Estimated small stage: 2-10 minutes GPU/CPU combined, <100 MB; start only after a resource slot and concrete budget are assigned.
2. **Event dependence/scaling.** Once nonlinear moments are audited, study horizon 10/20/40 and one/two/four obstacles with fixed physical noise. Measure exact joint events, empirical marginals and overlaps, Gaussian/tangent sums, and gradient/runtime scaling. Event grouping or second-order union corrections require empirical/dependence validation; do not treat approximate correlated Gaussians as guarantees. Compare total rollout-based decision cost, not just energy latency.
3. **Conventional risk-cost frontier.** Strengthen cost optimization and convergence accounting; use the same independent acceptance, report initialization/optimizer failure, and validate at common budgets. Diffusion must demonstrate a measurable benefit in solution quality, total compute, diversity, or nonconvex robustness before larger claims.
4. **Manipulator specification gate, then software-only demo.** Define a Panda physical noise model (initial joint-state estimation error and per-step actuation/process perturbation, covariance and temporal correlation); known joint/velocity or acceleration dynamics; actuator and joint bounds; full-arm links/capsules, self-collision and environment geometry; final joint/end-effector pose and tolerances; trajectory horizon and discretization/swept-volume policy. Define joint failure as any arm collision, limit violation, or terminal miss under physical execution draws. State map/calibration/model assumptions and discretization gaps. Reuse qualified Panda/MPD geometry/proposals read-only without altering their original projects. Benchmark nominal diffusion, guided/CVaR variants, projection, conventional stochastic optimization, and feasible sampling or trajectory optimization with identical execution validators. Proposal randomness must never replace physical noise. No hardware actuation. This stage requires a reviewed concrete model/specification, baseline qualification, resource allocation, and a separate budget; no implementation or training started in phase one.

## Current blockers and decisions

No preservation, identity, software, or CPU reproduction blocker remains. Sustained GPU work awaits the parent resource slot because a QSP project shares this machine. Manipulator assets have not yet been qualified for physical uncertainty or full-arm execution risk. No nonlinear probability certificate, conference-level novelty, or diffusion advantage over conventional planning is claimed. The first next experiment is frozen-control moment calibration, not a larger training sweep.
