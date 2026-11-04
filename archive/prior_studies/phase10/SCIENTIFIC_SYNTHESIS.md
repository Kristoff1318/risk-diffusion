# Scientific synthesis and decision

The strongest current result is that learned initialization becomes useful after ordinary nominal preparation. Phase7's raw MPD comparison was confounded: classical had IK and nominal optimization, while MPD had neither. Phase8 applied the identical 80-step nominal optimizer to the same 16 MPD seeds without IK or endpoint projection, restoring 15/16 nominally valid paths across all eight tasks. Phase9 then compared both sources after equal nominal preparation, with fresh independent risk validation.

| Source / budget | Prepared only | Eight dense deterministic steps | Eight sampled-CVaR steps |
|---|---:|---:|---:|
| Classical / 5% | 3/8 | 5/8 | 5/8 |
| Classical / 10% | 4/8 | 5/8 | 6/8 |
| MPD / 5% | 7/8 | 7/8 | 7/8 |
| MPD / 10% | 7/8 | 7/8 | 7/8 |

These establish three different capabilities, which should not be conflated:

1. **Improving initialization:** MPD's prepared proposals supplied useful nominal geometry that classical's two IK-based starts sometimes missed. That is an initialization result in this cohort, not yet a novel uncertainty-propagation contribution.
2. **Optimizing cost under risk:** CVaR reduced paired MPD cost by about 0.22 J and audited risk by 0.70/0.35 percentage points versus dense correction at 5%/10%. It added no coverage; prepared-only selected MPD paths already had zero observed audit failures. Extra optimization spends some safety margin to reduce nominal effort.
3. **Satisfying risk when nominal planning fails:** CVaR added one classical task over dense correction at 10%. At 5% it exchanged a gained task for a lost task. This is limited, mixed evidence, not a universal advantage.

The runtime cost is substantial: sampled correction averaged about 8.3 s per two-candidate request, versus 0.26 s for dense deterministic correction. Equal eight-step counts are not equal compute. Independent admission adds about 2.55 s to either; initial generation/IK and nominal preparation are separate recorded historical costs. Phase9 reused proposals and is diagnostic, not a fresh held-out or cold end-to-end trial.

Why is there little risk-repair headroom? Under the frozen ideal-observation controller, e[k+1]=0.9 e[k]+0.05(b+eta[k]). Joint error standard deviation contracts from 4 mrad at time zero to about 1.52 mrad at the goal. The bias is persistent but small, observation is perfect, and no actuator saturation or observation delay is modeled. Nominally repaired MPD paths are already robust enough for seven selected tasks. Dense cost optimization can move them closer to pose/collision boundaries; CVaR mostly moderates that cost-risk tradeoff. Neither solver can repair an initial violation already counted as failure.

## Initial-risk floor: exact scope

For fixed nominal start q0 and independent e0[j] ~ N(0,0.004^2), the probability of any initial joint violation is

p_initial = 1 - product_j [Phi((upper[j]-q0[j])/sigma0) - Phi((lower[j]-q0[j])/sigma0)].

Task 42576 has joint-4 upper clearance 0.002527613831 rad (0.631903 standard deviations). Its joint-4 failure probability is 0.263724981932. Joint 1 contributes 0.002852919781; their independent union is 0.265825515495. Later joint commands cannot undo the t=0 event, so the unchanged full-trajectory union probability is at least 26.5826%. Its 5% and 10% targets are impossible **for this fixed initial distribution and failure definition**.

This is an analytic conditional-model probability, not a Monte Carlo estimate: there is no sampling confidence interval to attach. Noise parameters, independence and calibration are assumed, not inferred. Hardware uncertainty, parameter uncertainty, correlated initial errors, safe-start conditioning, an abort-before-execution policy, or repositioning would change the model/event. This does not imply global geometric unreachability or infeasibility under those other choices. A low initial-joint floor does not certify attainable full-arm risk; initial collision and later events remain.

The new standalone pre-planning diagnostic, initial_joint_floor.py, reports that event, every joint's contribution and whether a requested budget is ruled out **only by this immutable event**. It uses stable Gaussian tails and a log-product; tiny values may still underflow and must not be called exact zero risk. It performs no sampling, FK, planning or training. Existing-task verification reproduces the recorded 26.5826% value. Initial-risk exclusions must remain visible in full benchmark reporting.

## Recommendation

Prioritize a cheaper covariance-based risk energy rather than expanding sampled-CVaR sweeps. Exact feedback covariance offers structure that the 32-execution gradient loop ignores. A local distance-risk approximation could reduce repeated geometry work; it could also be inaccurate or expensive once Jacobian derivatives and millions of correlated event constraints are included. Treat it as an optimization surrogate, with independent full nonlinear admission/audits.

Keep sampled CVaR as the preserved reference and a fallback research option. Current evidence does not justify making it the default: its small/mixed gains require much more correction time. The next question is whether a cheaper covariance-aware objective can improve independently measured cost-risk behavior on fresh, fixed tasks beyond dense deterministic repair. Do not increase noise, weaken tracking or move obstacles to manufacture that opportunity. If fresh healthy prepared paths already satisfy the budget, report limited headroom and stop.

See DISTANCE_RISK_DESIGN.md and FRESH_CONFIRMATION_PLAN.md. No new planning, inference, optimization or training has run in this documentation phase.

