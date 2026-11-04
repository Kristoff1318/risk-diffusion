# Opt-in closure guidance: matched CPU planning comparison

Completed 2026-10-02. The protocol was committed as **42a1847 before evaluation**. Baseline tag/defaults and all historical artifacts remain untouched. Added an opt-in sampler in revision/guidance_closure.py, a bounded comparator derived from the existing conventional optimizer, and a frozen experiment harness. Six parity checks reproduced the historical nominal/chance/CVaR CPU sampler exactly (maximum difference zero). Recorded energy/gradient traces were finite; every independently reconstructed covariance was PSD. No training, new assets, installs, downloads, GPU, or manipulator adaptation.

## Protocol and limits

Sixteen new tasks per dynamics regime (32 total); known model, a=.12 or .35, physical initial/process Gaussian noise unchanged; Delta=.05/.10; one proposal per task/variant/budget (320 total including conventional). Each guided method uses 64 denoising steps, 32 eligible correction stages, lambda=8, gamma=30, and the same .1*J cost pressure. J is nominal path length + .02*dt*control effort + .01*control roughness. CVaR uses 256 fixed noise paths (8,192 internal rollout evaluations per proposal). Diffusion initial noise, task conditions, physical validation/audit noise, and schedules are paired across variants/budgets. Cost weight .1 comes from the already implemented tuning grid and was fixed before final outcomes; no final-task tuning occurred. Equal lambda need not equalize energy gradient scales.

The conventional comparator uses the existing buffered SLSQP formulation: minimize the same J, internal Gaussian risk threshold .8*Delta, maximum 300 iterations, ftol=1e-5. It starts from the identical frozen nominal diffusion controls and has a .6-second wall guard per solve. Timeout retains and reports the last evaluated candidate, which may still pass independent validation; failed solves are never silently removed. This is a limited single-initialization comparator, not a broad conventional-planner benchmark.

Admission requires the independent 4,096-draw one-sided 95% combined-failure upper bound <= Delta and control bounds. The separate 8,192-draw audit is never used to select, tune, or accept. Per-task audit two-sided 95% intervals for failure, collision, and goal success accompany the CSV. Intervals concern fixed tasks and modeled execution noise; they are not familywise or population-level safety statements. Obstacles at x0..x[H-1] and goal at xH preserve the original discrete-time event semantics. No continuous-time or terminal-obstacle guarantee is added.

## Acceptance, accepted-plan risk, cost, and time

Risks below are fresh **accepted-plan** mean audit failure risks. All-proposal risks are reported separately below. Query time includes generation/refinement and validation, but excludes research audit/decomposition. Diffusion times are amortized batch-of-16 throughput, not isolated single-request latency; SLSQP is per-plan CPU optimization. Fixed steps and samples are not equal compute budgets.

| Dynamics / budget | Method | Accepted / 16 | Accepted audit failure | Accepted nominal J | Query ms per plan |
|---|---|---:|---:|---:|---:|
| moderate_noise/delta_0.05 | nominal | 14 | 2.646% | 3.9312 | 5.26 |
| moderate_noise/delta_0.05 | chance | 15 | 2.789% | 3.8979 | 12.82 |
| moderate_noise/delta_0.05 | closure_chance | 15 | 2.793% | 3.8977 | 17.58 |
| moderate_noise/delta_0.05 | cvar | 14 | 2.632% | 3.9161 | 26.81 |
| moderate_noise/delta_0.05 | conventional | 16 | 3.880% | 3.3483 | 351.21 |
| moderate_noise/delta_0.1 | nominal | 16 | 3.053% | 3.8931 | 5.02 |
| moderate_noise/delta_0.1 | chance | 16 | 2.998% | 3.8785 | 13.34 |
| moderate_noise/delta_0.1 | closure_chance | 16 | 3.001% | 3.8783 | 17.77 |
| moderate_noise/delta_0.1 | cvar | 16 | 2.927% | 3.8853 | 25.94 |
| moderate_noise/delta_0.1 | conventional | 16 | 7.640% | 3.2368 | 306.92 |
| nonlinear/delta_0.05 | nominal | 13 | 2.280% | 3.8786 | 5.06 |
| nonlinear/delta_0.05 | chance | 14 | 2.279% | 3.8638 | 12.56 |
| nonlinear/delta_0.05 | closure_chance | 14 | 2.246% | 3.8642 | 17.28 |
| nonlinear/delta_0.05 | cvar | 15 | 1.878% | 3.8772 | 26.33 |
| nonlinear/delta_0.05 | conventional | 3 | 4.451% | 3.2844 | 539.77 |
| nonlinear/delta_0.1 | nominal | 15 | 2.869% | 3.8525 | 5.05 |
| nonlinear/delta_0.1 | chance | 15 | 2.738% | 3.8449 | 12.59 |
| nonlinear/delta_0.1 | closure_chance | 15 | 2.691% | 3.8452 | 17.22 |
| nonlinear/delta_0.1 | cvar | 15 | 1.965% | 3.8762 | 25.84 |
| nonlinear/delta_0.1 | conventional | 7 | 8.718% | 3.1904 | 542.22 |

Closure and original chance accept exactly the same tasks in every regime/budget. The closure cost differences on common accepted tasks are -.000288/-.000250 (moderate .05/.10) and +.000364/+.000326 (nonlinear .05/.10). These changes are practically tiny. Closure reduces mean all-proposal nonlinear audit failure by .109/.112 percentage points, and accepted-only failure by .033/.046 points. Moderate-noise failure changes are tiny in the opposite direction. No acceptance gain or convincing cost-risk frontier improvement is established.

Closure query time rises roughly 33-38% over original chance (about 13 -> 17 ms); energy/backward time rises from roughly 7.3-8.1 -> 12.0-12.6 ms per amortized plan. Equal steps are explicitly not an equal-compute win. CVaR costs about 26 ms and at nonlinear/.05 accepts 15 versus chance/closure's 14; it has higher accepted nominal J and a different accepted set. Do not compare accepted risks without that selection context.

The conventional method exposes a different frontier point in moderate noise: all 16 accepted at both budgets, J ~3.35/.05 and ~3.24/.10 versus chance's ~3.90/~3.88, but audit risk ~3.88%/~7.64% versus ~2.79%/~3.00%, and 0.31-0.35 seconds per query. It uses more risk and compute; this is not a matched-realized-risk superiority claim. It shows why diffusion advantage cannot be inferred from guidance ablations alone.

## All proposals, failure types, and optimizer failures

| Dynamics / budget | Method | All-proposal failure | All-proposal collision | All-proposal goal success | Rejected | Optimizer timeouts |
|---|---|---:|---:|---:|---:|---:|
| moderate_noise/delta_0.05 | nominal | 3.053% | 0.545% | 97.334% | 2 | 0 |
| moderate_noise/delta_0.05 | chance | 2.901% | 0.381% | 97.344% | 1 | 0 |
| moderate_noise/delta_0.05 | closure_chance | 2.905% | 0.385% | 97.343% | 1 | 0 |
| moderate_noise/delta_0.05 | cvar | 2.917% | 0.401% | 97.350% | 2 | 0 |
| moderate_noise/delta_0.05 | conventional | 3.880% | 0.387% | 96.436% | 0 | 0 |
| moderate_noise/delta_0.1 | nominal | 3.053% | 0.545% | 97.334% | 0 | 0 |
| moderate_noise/delta_0.1 | chance | 2.998% | 0.478% | 97.330% | 0 | 0 |
| moderate_noise/delta_0.1 | closure_chance | 3.001% | 0.481% | 97.330% | 0 | 0 |
| moderate_noise/delta_0.1 | cvar | 2.927% | 0.410% | 97.349% | 0 | 0 |
| moderate_noise/delta_0.1 | conventional | 7.640% | 1.035% | 93.198% | 0 | 0 |
| nonlinear/delta_0.05 | nominal | 6.590% | 4.656% | 95.816% | 3 | 0 |
| nonlinear/delta_0.05 | chance | 3.050% | 1.015% | 97.526% | 2 | 0 |
| nonlinear/delta_0.05 | closure_chance | 2.941% | 0.935% | 97.584% | 2 | 0 |
| nonlinear/delta_0.05 | cvar | 3.427% | 1.682% | 97.497% | 1 | 0 |
| nonlinear/delta_0.05 | conventional | 10.970% | 0.455% | 89.399% | 13 | 7 |
| nonlinear/delta_0.1 | nominal | 6.590% | 4.656% | 95.816% | 1 | 0 |
| nonlinear/delta_0.1 | chance | 3.289% | 1.218% | 97.398% | 1 | 0 |
| nonlinear/delta_0.1 | closure_chance | 3.177% | 1.131% | 97.461% | 1 | 0 |
| nonlinear/delta_0.1 | cvar | 3.681% | 1.903% | 97.356% | 1 | 0 |
| nonlinear/delta_0.1 | conventional | 20.552% | 0.931% | 80.157% | 9 | 10 |

Rejected cases were retained. At moderate/.05, chance/closure reject task 1; nominal/CVaR reject 1 and 13. At nonlinear/.05, chance/closure reject 10 and 14; CVaR rejects 14; nominal rejects 4,10,14. At nonlinear/.10, every diffusion variant rejects 14. Full conventional rejection/timeout IDs and all measurements are in the per-case CSV/raw JSON.

All 32 moderate conventional solves succeeded. In stronger nonlinear dynamics, 7/16 at .05 and 10/16 at .10 timed out; acceptance was only 3/16 and 7/16. The strict-budget conventional proposals include four Gaussian-surrogate false-safe cases confirmed by fresh audit; validation rejected them. Retaining and independently evaluating last iterates exposes, rather than conceals, limited solve time and surrogate mismatch. These outcomes cannot establish an intrinsic diffusion advantage over a fully converged or better-calibrated conventional optimizer.

No accepted plan had an audit point estimate or 95% lower interval above its budget in this small run. This is not a zero failure risk or universal acceptance guarantee. Failure includes collision or terminal-goal miss; the events overlap, so their probabilities should not simply be added. Goal misses dominate several optimizer failures.

## Does closure help beyond conservative risk inflation?

Closure slightly changes controls and improves measured nonlinear failure, so its effect is not solely changing a diagnostic number. However, the planning benefit is small and accompanied by more computation. At nonlinear/.10, mean active surrogate sum rises .04101 -> .04244 while all-proposal audit failure falls .03289 -> .03177; acceptance and cost barely change. Thus surrogate calibration changes more than the planning frontier.

The 2,048-draw decomposition stream is diagnostic and noisier than the audit. At nonlinear/.10, Gaussian-minus-empirical-tangent error improves -.01545 -> -.01139, tangent marginal MAE .001135 -> .000722; tangent geometry adds .01090 -> .00995 and event union aggregation adds .01276 -> .01196. Better moments do not remove non-Gaussian tail mismatch or dependence/aggregation error. The residual contributions partly cancel. Exact one-step Gaussian-input moments still imply only approximate multistep closure and do not supply an exact chance bound.

At .05 the same pattern holds: nonlinear Gaussian error -.01448 -> -.01064 and tangent marginal MAE .001079 -> .000686, but no closure acceptance gain. There was no projection, conditioning, retraining, or event-definition change to explain the differences.

## Decision and next bounded step

**Keep closure opt-in; do not replace the default on this evidence.** It improves physical-model calibration, but this fixed-strength guidance experiment does not demonstrate a better cost-risk-acceptance frontier. No equal-compute or diffusion-over-conventional advantage is claimed.

Proposed next step, not launched: a small matched conventional-refinement study with original versus closure risk constraints, same explicit J, fixed CPU time cap, and identical diffusion/waypoint initializations. Use eight new nonlinear tasks, .05/.10 budgets, fresh paired validation/audit, and retain every timeout/failure. This directly tests whether better propagation helps when optimization spends the risk budget near a cost frontier, where the prior diagnostic effect was largest. Pre-freeze all configurations and do not tune on final tasks. Estimated 30-90 CPU seconds, <10 MB, no training or GPU. A larger guidance-strength sweep or manipulator adaptation is not justified yet.

## Deliverables and resources

PROTOCOL.json and FROZEN_TASKS.npz preserve the pre-evaluation specification and nominal controls. PER_CASE_RESULTS.csv includes all 320 proposals, acceptance, collision/goal/combined risk intervals, costs, runtime, and solver failure. PAIRED_TASKS.csv compares each task against original chance, separating common acceptance. SUMMARY.json reports proposal and accepted-only metrics independently. Raw plans, gradient traces, decomposition and results are retained in phase3_outputs/guidance_20261002, with hashes in RESOURCE_LEDGER.json.

Freeze process .50 seconds; experiment 37.75 wall seconds (body 37.23), 69.26 user + .24 system CPU seconds across threads, peak RSS 1,120,184 KB (~1.07 GiB). Raw artifacts ~1.97 MB; final deliverables/Git growth remain well below the 25 MB cap. Counted validation/audit/decomposition trajectory evaluations: 1,310,720 / 2,621,440 / 655,360, plus 524,288 repeated internal CVaR rollout evaluations. Paired methods/budgets share underlying noise draws; these are evaluation counts, not millions of independent task samples. No GPU, training, installs, downloads, extra assets, or hardware use.
