# Traceability to the original plan

The authoritative user request is preserved verbatim in [ORIGINAL_PLAN.txt](ORIGINAL_PLAN.txt). This status describes the September 26 revision. Improvements repair the implementation and strengthen the evidence within that plan. Earlier pilot conclusions are preserved under `legacy_20260922/`; they are superseded by the current RESULTS and VERDICT.

| Original stage | Implemented evidence | Scope / remaining limitation |
|---|---|---|
| 1. Audit and reuse | AUDIT.md, repository inspection; reusable diffusion concepts and environments reviewed | No revival of FULL/PREFIX or temporal logic machinery |
| 2. Minimal stochastic nonlinear 2D testbed | revision/problem.py; two obstacle layouts, two noise levels, stronger nonlinear drift | Discrete-time, known-model, open-loop pilot |
| 3. Explicit representation | Control sequences, dynamics-derived states; DESIGN.md | Map/dynamics implicit in separate checkpoints |
| 4. Healthy nominal baseline first | revision/nominal.py, run_nominal.py; three seed gates and both extension gates passed | Simple feasible waypoint training distribution |
| 5. Independent execution evaluator | NumPy evaluate.py; exact hard events, confidence intervals, separate validation and audit RNG | Guarantees conditional on declared execution distribution |
| 6. Chance guidance during denoising | risk.py and guidance.py; full covariance/mean gradients, Gaussian CDF sum; calibration.py | Approximate nonlinear moments can be false-safe |
| 7. Genuine sampled CVaR guidance | Exact fractional empirical upper tail; fixed reparameterized samples; K sweep | Finite sampled optimization does not certify chance feasibility |
| 8. Projection after guidance | Local nearest-control constrained SLSQP at three denoising stages; B4/B5/final-only comparison | Local nonconvex solver; deterministic feasibility only |
| 9. Risk-aware training afterward | Matched architecture/data/updates/RNG; explicit condition mask; ordinary and marginal-data experiments | Reported CFG scale 1; no CFG amplification study |
| 10. Q1–Q8 | Explicit answers in RESULTS.md | Budget/cost calibration remains weak; broad generalization unresolved |
| 11. B0–B7 | B0–B3 three seeds; B4/B5 extension pilot; B6/B7 matched three-seed comparisons | Geometry/dynamics extensions use one trained seed each |
| 12. Metrics | Per-plan JSONs include risk, goal, cost, CVaR, rollout/step counts; summary plots, diversity.json, timing.json | Small task set; no claim of independent 96-task population statistics |
| 13. Explicit stochastic risk question | Surrogate decomposition, independent acceptance, fresh audit, nonlinear false-safe counterexample | Mean risk below budget alone is never treated as sufficient |
| 14. Literature | LITERATURE.md includes original slide references, nearby diffusion and conventional methods | No conference-level novelty established |
| 15. Deliverables | DESIGN, LITERATURE, RESULTS, VERDICT; revision/reproduce.sh; scripts and saved checkpoints/results/figures | Reproduction requires the documented scientific Python/CUDA environment |
| 16. Autonomy and simplification | Staged pilots; conventional optimizer retained as comparator | No escalation to a manipulator |

## Enhancements attached to the plan

1. Repair the nominal prior and remove hidden endpoint correction before interpreting safety ablations.
2. Correct empirical CVaR and match conditional training budgets.
3. Separate proposed risk, accepted-plan risk, and fresh audit risk; return failure when no plan passes.
4. Decompose surrogate error and actively probe near the risk boundary.
5. Compare explicit cost optimization and final-only correction to establish what diffusion contributes.

These implement and test the original formulation. They do not replace the diffusion-time chance/CVaR experiments.
