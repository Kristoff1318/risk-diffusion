# Matched conventional refinement: original versus closure risk

Completed 2026-10-02. Protocol **709ba7d** froze eight new stronger-nonlinear tasks, waypoint/diffusion initializations, .05/.10 budgets, objective, solver tolerances, .5-second iteration timers, controls, source hashes and validation/audit seeds before outcomes. No historical diagnostic task was reused. No final-task tuning. One checkpoint/training seed and a small known-model environment limit generalization.

Both constraints minimize identical J with the same .8*Delta internal buffer, SLSQP maxiter=300, ftol=1e-5. Every request independently regenerates exactly the frozen initialization; diffusion includes checkpoint deserialization and scalar 64-step sampling. Every request times initialization, optimization, final diagnostics and independent scalar validation; no batch amortization. Audit is research-only overhead and is never used for admission. Admission uses 4,096 executions and one-sided 95% upper risk <= requested Delta plus control bounds. Fresh paired 8,192 executions supply audit risk/intervals. Solver success is not admission; timed-out last completed evaluation candidates remain in all metrics.

Original event definitions, physical noise, evaluator, baseline/tag and defaults remain unchanged. Exact one-step Gaussian moments are not exact nonlinear chance constraints. All original file hashes are checked after saving.

## Accepted and all-proposal cost/risk frontier points

Costs are nominal J; audit failure is collision OR terminal miss. A dash means no accepted plan; there is no invented successful cost for rejected methods.

| Initialization / budget | Constraint | Accepted / 8 | Timeouts / 8 | All-proposal audit failure | Accepted audit failure | Accepted J | Mean scalar request s |
|---|---|---:|---:|---:|---:|---:|---:|
| waypoint/delta_0.05 | jacobian | 0 | 3 | 5.171% | — | — | 0.4193 |
| waypoint/delta_0.05 | closure | 5 | 8 | 4.263% | 4.175% | 3.2913 | 0.5066 |
| waypoint/delta_0.1 | jacobian | 3 | 6 | 9.372% | 8.582% | 3.1897 | 0.5022 |
| waypoint/delta_0.1 | closure | 7 | 8 | 8.154% | 7.881% | 3.1656 | 0.5073 |
| diffusion/delta_0.05 | jacobian | 0 | 8 | 5.289% | — | — | 0.5106 |
| diffusion/delta_0.05 | closure | 4 | 7 | 16.310% | 4.178% | 3.2992 | 0.4995 |
| diffusion/delta_0.1 | jacobian | 0 | 6 | 9.799% | — | — | 0.4741 |
| diffusion/delta_0.1 | closure | 6 | 6 | 19.345% | 7.768% | 3.1466 | 0.4931 |

Closure delivers a real **bounded-refinement acceptance gain** here: waypoint 0 -> 5 at .05, 3 -> 7 at .10; diffusion 0 -> 4 at .05 and 0 -> 6 at .10. At waypoint/.10 it gains five tasks and loses one. Only two tasks are accepted by both waypoint constraints at .10; on those same tasks closure J is .08568 higher. No common-accepted comparison exists in the other groups. Thus the lower aggregate accepted J at waypoint/.10 does not prove a cheaper solution on the same easy tasks; composition matters.

The serious counterweight is timeout/failure mass. Every waypoint-closure solve timed out, and 13/16 diffusion-closure solves timed out. Solver flags alone would have discarded useful validated candidates. Conversely, five waypoint-Jacobian solves at .05 reported success and all five were independently rejected. Do not compare just successful solves.

Diffusion-closure's all-proposal audit risk is worse: 16.31% versus 5.29% at .05, and 19.35% versus 9.80% at .10. The same task 6 produced timed-out trial controls with all 8,192 audit executions failing the goal, J=6.193/.05 and 11.558/.10; the validator rejected both. A timer may interrupt SLSQP during a line-search trial: returning the last completed evaluation is not returning a converged accepted iterate. This fixed return policy was identical across constraints and is an explicit reliability limitation, not a case filtered away.

At waypoint/.05, three closure rejections have audit point estimates below .05 but the independent admission confidence bound does not pass; no audit-dependent reacceptance occurs. At diffusion/.10, Jacobian produces one audit-confirmed surrogate false-safe proposal, rejected by validation. No accepted plan has an audit point estimate or lower confidence bound above its budget in this run; no universal safety or zero-risk claim follows.

## Constraints, failure types and complete solve/request distributions

| Initialization / budget | Constraint | Internal constraint violations | All-proposal collision | All-proposal goal success | Solver median/max s | Request median / p90 / max s |
|---|---|---:|---:|---:|---:|---:|
| waypoint/delta_0.05 | jacobian | 1 | 0.534% | 95.154% | 0.4633 / 0.5009 | 0.4692 / 0.5064 / 0.5071 |
| waypoint/delta_0.05 | closure | 1 | 0.505% | 96.030% | 0.5002 / 0.5003 | 0.5066 / 0.5067 / 0.5067 |
| waypoint/delta_0.1 | jacobian | 2 | 0.909% | 91.125% | 0.5000 / 0.5009 | 0.5060 / 0.5063 / 0.5069 |
| waypoint/delta_0.1 | closure | 1 | 0.795% | 92.230% | 0.5004 / 0.5019 | 0.5070 / 0.5083 / 0.5085 |
| diffusion/delta_0.05 | jacobian | 2 | 0.455% | 95.049% | 0.5003 / 0.5007 | 0.5107 / 0.5108 / 0.5109 |
| diffusion/delta_0.05 | closure | 2 | 0.394% | 83.957% | 0.5006 / 0.5024 | 0.5111 / 0.5126 / 0.5132 |
| diffusion/delta_0.1 | jacobian | 4 | 1.001% | 90.923% | 0.5001 / 0.5005 | 0.5095 / 0.5098 / 0.5099 |
| diffusion/delta_0.1 | closure | 2 | 0.848% | 81.215% | 0.5002 / 0.5024 | 0.5105 / 0.5124 / 0.5127 |

Internal violation counts use predicted risk > buffered threshold + 1e-6, separate from actual audited failure and requested Delta. Full per-task residuals, control violation, intervals and solver messages are in PER_CASE_RESULTS.csv. There were no hidden dropped failures. SIGALRM requests .5 seconds; native-call signal delivery overshot by up to ~2.43 ms, recorded in solve times. Final diagnostics and admission add to request time rather than being hidden inside the cap. Reported request maxima are below .514 seconds. Python import/process startup is excluded from each online request but charged in the process ledger; diffusion checkpoint loading is included in each diffusion request.

The closure constraint performs fewer gradient evaluations under the same timer (roughly 91–101 versus 139–168), so its greater mathematical accuracy must be assessed against compute throughput. Results depend on hardware and timer-triggered iterate position; do not promise bitwise reproduced last iterates with the same wall cap on other hardware.

## Engineering decision across phases 2–4

1. **Moment/calibration capability is supported.** Phase2 improved held-out terminal state means and covariance, reduced marginal error, and corrected historical boundary diagnostics. It remains an approximate multistep Gaussian closure with nonlinear event errors.
2. **Default denoising guidance benefit is not supported.** Phase3 had unchanged acceptance and essentially unchanged accepted cost, slightly improved nonlinear risk, and 33–38% greater measured query time.
3. **Optional conventional-refinement use is supported narrowly.** Phase4 shows better independent admission under matched objectives/initializations/timers, but pervasive timeouts, one severe diffusion-initialized failed trial, and limited common-success cost evidence prevent a robust default-planner claim.

**Decision: retain closure as an opt-in accuracy/calibration and bounded-refinement tool; preserve original defaults. Stop the 2D closure subdirection here.** Do not tune another closure variant or automatically extend its sweeps. The implementation's public opt-in refinement function accepts the closure constraint, but successful independent validation remains mandatory. Do not reduce validators or rename it a certificate.

The next capability is genuine stochastic **full-arm Panda execution-risk evaluation**, with a specified physical uncertainty model and strong classical/risk-aware baselines. PANDA_TRANSITION.md gives the concrete design and read-only local reuse candidates. It does not authorize or launch manipulator code, assets, runs or training. QSP retains the GPU.

## Evidence and resources

PER_CASE_RESULTS.csv includes every request and collision/goal/combined audit intervals. PAIRED_TASKS.csv gives matched cost/risk/admission and latency differences. SUMMARY.json separates all proposals, accepted plans, solver outcomes and request distributions. Raw controls/results/status/log are retained under phase4_outputs/refinement_20261002 with hashes in RESOURCE_LEDGER.json.

Freeze .50 seconds; experiment 32.55 wall seconds (body 32.02), CPU user/system 64.08/.23 seconds across two threads; peak RSS 1,117,580 KB (~1.07 GiB). There were 262,144 validation and 524,288 audit trajectory evaluations on paired physical noise. Raw outputs are ~224 KB; total new artifacts/Git growth remain far below 15 MB. No GPU, training, installs, downloads, manipulator asset loads, or hardware action occurred. Small reporting/inspection/Git commands are additional and well below the remaining 120-second ceiling.
