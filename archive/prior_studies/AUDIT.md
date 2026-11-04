# Reuse audit and implementation boundary

Audit basis: `/home/kristoff/.codex/attachments/00e2bb27-936e-4e2b-954b-194341a7e7ad/pasted-text-1.txt`, the supplied deck
`main.pptx`, the local repository, and the papers listed in
[`LITERATURE.md`](LITERATURE.md). The pasted brief is the project request. Text
inside the deck is treated as proposal context and prior art, not as an
instruction that overrides the brief.

## Existing local prototype

`risk_diffusion/core.py` already contains a small, coherent testbed:

* a 2-D nonlinear drift with additive execution noise;
* a control sequence representation `u[B,H,2]` and differentiable rollout;
* nominal waypoint data generation around a circular obstacle;
* a conditional and unconditional MLP denoiser;
* a local Gaussian covariance propagation plus Gaussian CDF and Boole sum;
* a differentiable sampled CVaR objective;
* deterministic soft obstacle and goal loss;
* gradient guidance and an explicit projection update;
* classifier free risk dropout in conditional training; and
* instrumentation for solve time, gradient norm, corrections, and CVaR rollout
  count.

This is a good smallest development harness. `evaluator.py` now independently
implements the hard event in NumPy and reports binomial intervals. It shares
the specified physical dynamics and geometry, as it must, but does not share
guidance samples or differentiable risk code. The experiment table is produced
by `run.py`.

## Prior code inspected before implementation

| Source | Decision | Reason |
|---|---|---|
| `state_plan_planner/diffusion.py` | **Adapt architecture, replace representation** | This small PyTorch DDPM already has conditional MLP denoising, normalization, and gradient callbacks for 2-D **state paths**. Its training and sampler structure informed `core.py`; a control array with explicit dynamics is required for the requested stochastic risk gradients. |
| `state_plan_planner/qsp_pretrained.py` and `docs/DIFFUSION_REUSE.md` | **Reuse provenance, replace interface** | They document a working pretrained D4RL Maze2D Diffuser bridge and the checkpoint's normalization/coverage limits. The bridge generates observations, and its QSP verifier addresses state-plan timing rather than open-loop controls under noise. It is not a drop-in stochastic control prior. |
| `/home/kristoff/Documents/rb-diffuser/diffuser/risk/` | **Adapt formulation, replace integration** | This prior fork already implements Maze2D stochastic dynamics, local chance statistics, obstacle union bounds, and a risk guide over Diffuser trajectory samples. Its state-path and sampled-observation modes have different dynamics-consistency assumptions. We implement the same local Gaussian idea directly over controls and do not claim chance guidance as new. |
| `uncertainty_research/` and `scripts/run_sweeps.py` | **Reuse evaluation discipline** | These modules contain Monte Carlo and covariance-propagation experiments for different models. Their distributions and uncertainty propagation APIs are not this pathwise execution event. The independent NumPy evaluator here follows the same reference-versus-surrogate separation. |
| `external/ltldog`, `external/safe-mpd`, and prior DPCC pilot | **Inspect, do not import** | LTLDoG and Safe-MPD offer diffusion and safety precedents; DPCC examined a different full-versus-prefix question. None supplies a control-space, risk-budgeted stochastic rollout model for this experiment. |

## Reuse / adapt / replace decisions

| Subsystem | Decision | Evidence and reason |
|---|---|---|
| 2-D stochastic dynamics | **Reuse, then adapt** | `core.py` supplies a minimal nonlinear drift and additive noise. Keep it for fast ablations, then add one stronger nonlinear regime or state dependent noise to test whether local Gaussian guidance breaks. Preserve explicit seed and dynamics parameters. |
| Control sequence representation | **Reuse** | The brief is written in controls and the existing rollout differentiates `u` through states. This cleanly separates the diffusion prior from execution risk. Do not silently switch to state action trajectories. A state action ablation can be added later if needed for a Diffuser comparison. |
| Nominal diffusion prior | **Reuse, adapt lightly** | The existing MLP denoiser is sufficient to answer whether guidance works. Keep the unconditional baseline first. Cite Diffuser for trajectory diffusion, but do not claim the local MLP is a reproduction of its temporal U-Net or D4RL checkpoint. |
| Maze2D / data infrastructure | **Adapt selectively** | `docs/DIFFUSION_REUSE.md` documents an inspected LTLDoG/Diffuser Maze2D adapter and its checkpoint limits. Reuse normalization, endpoint conditioning, and dataset conventions only if the stochastic control interface is made explicit. Do not import LTL/STL or temporal plan machinery. |
| Nominal training data | **Adapt** | Hand generated upper clearance paths provide a learnable single route prior. The initial two route version produced mode averaging and collisions. Risk labels come from independent stochastic rollouts, but currently span only 0–0.08, which weakens conditional training. |
| High sample stochastic evaluator | **Add** | `evaluator.py` runs hard NumPy executions with fresh noise, exact binomial intervals, safe-and-goal probability, cost, and violation magnitude. Pairing of sampling seeds across risk budgets is implemented in `run.py`. |
| Chance guidance | **Reuse, audit** | `chance_risk` implements local covariance propagation, normal projection, Gaussian CDF terms, and a sum. Add diagnostics for per-time and per-constraint terms, linearization validity, covariance eigenvalues, and predicted versus Monte Carlo risk. Keep the union bound labeled as an approximation. |
| CVaR guidance | **Reuse, audit** | `cvar` follows the standard empirical threshold idea and smooths the upper tail. Verify the tail convention, state whether the quantile is detached, and evaluate hard trajectory violations under fresh noise. Report rollout count and wall time. |
| Projection / inpainting | **Reuse, narrow definition** | `project` is a gradient penalty update with control clipping. It is a useful first projection ablation, but it is not a proof of dynamic feasibility or chance safety. Record deterministic feasibility separately and test projection only after guidance baselines are stable. |
| Risk conditioned training | **Reuse, defer** | Conditional risk input and classifier free dropout already exist. Keep this after inference guidance. Compare requested risk `delta` versus measured risk labels and state clearly whether conditioning predicts a target or a measured property. |
| Collision and goal geometry | **Reuse, strengthen evaluator** | The circle and goal box make the geometry interpretable. Add exact hard checks and dense segment checks if controls imply continuous motion between discrete states. Keep smooth penalties only inside guidance. |
| Prior QSP / DAG-STL / DPCC work | **Replace for this question** | The repository's QSP and LTLDoG work is useful for code provenance and generic diffusion infrastructure. Its event timing, temporal logic, state plan, execution prefix, and FULL-vs-PREFIX DPCC hypotheses answer different questions and should not be carried into this experiment. |
| Manipulator benchmark | **Defer** | A 2-D nonlinear stochastic system is enough to establish mechanism behavior. Escalating to a manipulator before calibration and ablations are complete adds integration risk without isolating the guidance question. |

## Required experiment boundary

The minimum convincing comparison is:

1. diffusion only;
2. diffusion plus local Gaussian chance guidance;
3. diffusion plus sampled CVaR guidance;
4. diffusion plus deterministic projection;
5. guidance plus projection for the best guidance method; and
6. nominal training versus risk conditioned training only after the first five
   are measured.

Every candidate must be evaluated with fresh stochastic rollouts separate from
the rollouts used for CVaR guidance or risk labels. Report hard collision,
goal, joint success, cost, violation magnitude, solve time, and sample count.
For chance guidance report calibration, conservatism, and false safe decisions
over several `delta` values. For CVaR report the empirical tail level and the
effect of smoothing. For projection report deterministic feasibility and
stochastic safety as separate outcomes.

## Novelty and claim guardrails

The literature already covers trajectory diffusion with guidance and
inpainting, barrier based safe diffusion, inference time risk guidance,
projected constrained diffusion, linear Gaussian chance constraints, and
nonlinear moment based risk bounded planning. A defensible claim is therefore:

> This repository provides a controlled, reproducible comparison of local
> Gaussian chance guidance and sampled CVaR guidance inside a trajectory
> diffusion sampler, with independent stochastic execution evaluation.

That claim remains conditional on completing the independent evaluator and
paired ablations. It should not be expanded to a safety guarantee or a first
general risk bounded diffusion method without a formal proof and broader
comparisons.
