# Results: original-plan revision, September 26

## What was repaired

The original plan remains authoritative in ORIGINAL_PLAN.txt. The earlier pilot confounded risk guidance with an unhealthy nominal prior, a hidden endpoint correction, an approximate CVaR threshold/loss, a soft clearance optimizer called projection, and unmatched conditional training. Its apparent projection advantage and negative conditioning verdict were too strong. Those reports are archived under legacy_20260922; the current implementation is revision/.

The revised order is nominal gate → independent evaluator → chance guidance → exact empirical CVaR → constrained projection → matched conditioning. The extra final optimizer is a comparator. All primary risk corrections act during denoising.

## Protocol and nominal gate

- Nominal prior: 6,000 ordinary feasible control sequences; 4,000 training updates; cosine v-prediction diffusion; 64 DDIM sampling steps; no B0 endpoint repair. Three training seeds achieve 99.22%, 99.22%, and 99.02% deterministic feasibility on 512 held-out conditions. Staggered and stronger nonlinear checkpoints achieve 98.44% and 98.05%.
- Main risk experiment: 32 shared start/goal conditions × three training seeds = 96 proposals per method/budget; two noise/goal configurations; Delta = .05, .10, .20. The task conditions repeat across training seeds and methods, so 96 is not 96 independent environments.
- Extension experiment: 32 shared tasks, one trained seed per geometry/dynamics configuration. B4/B5 are compared on this subset, not conflated with the three-seed main table.
- Independent validation: 4,096 executions per proposal, one-sided exact 95% failure upper bound; accepted only if upper bound <= Delta and control bounds hold. A separate 8,192-draw audit evaluates every proposal. Mean all-proposal risk, acceptance count, and accepted-plan audit risk are kept separate. Shared random numbers pair method comparisons; per-plan confidence statements are not simultaneous across the entire research table.
- The public planner corrects for selection within a fixed candidate batch, returns the lowest-J accepted candidate, and returns explicit failure otherwise. The experiment tables measure one proposal per condition/seed, not the public multi-candidate success rate.
- J = nominal path length + .02 dt sum(u²) + .01 sum((u[k+1]-u[k])²). Main guidance isolates risk with zero explicit J weight. Executed path length and hard tail loss are measured independently below.

## Primary B0–B3 comparison, moderate noise, Delta = .10

| Method / setting | Plans accepted | Mean audit risk, all proposals | Mean audit risk, accepted | Mean J, accepted |
|---|---:|---:|---:|---:|
| B0 / moderate_noise | 91/96 | 4.78% | 3.16% | 3.894 |
| B1 / moderate_noise | 91/96 | 4.62% | 3.16% | 3.894 |
| B2 / moderate_noise | 96/96 | 3.18% | 3.18% | 3.881 |
| B3 / moderate_noise | 93/96 | 3.33% | 3.01% | 3.896 |

Joint safe-and-goal probability is one minus the reported failure risk. Among accepted plans here, none had a fresh audit point estimate above its budget. This observation does not prove universal safety or a zero false-accept probability.

### Additional execution metrics

The following are means over all proposals using the independent 4,096-draw evaluation, including rejected attempts. CVaR is in signed spatial-violation units, not probability units.

| Method | Goal success | Expected executed length | Hard CVaR at 10% | Mean positive violation |
|---|---:|---:|---:|---:|
| B0 | 97.12% | 3.878 | -0.0094 | 0.0051 |
| B1 | 97.10% | 3.879 | -0.0103 | 0.0034 |
| B2 | 97.26% | 3.893 | -0.0176 | 0.0018 |
| B3 | 97.21% | 3.893 | -0.0171 | 0.0019 |

## B4/B5 and harder settings, Delta = .10

| Method / setting | Plans accepted | Mean audit risk, all proposals | Mean audit risk, accepted | Mean J, accepted |
|---|---:|---:|---:|---:|
| B0 / nonlinear | 29/32 | 5.52% | 1.94% | 3.930 |
| B0 / staggered | 30/32 | 4.70% | 3.07% | 3.897 |
| B2 / nonlinear | 32/32 | 2.07% | 2.07% | 3.904 |
| B2 / staggered | 31/32 | 3.95% | 3.15% | 3.888 |
| B3 / nonlinear | 31/32 | 4.57% | 1.57% | 3.940 |
| B3 / staggered | 31/32 | 4.45% | 3.12% | 3.892 |
| B4 / nonlinear | 29/32 | 4.68% | 1.94% | 3.930 |
| B4 / staggered | 30/32 | 4.70% | 3.07% | 3.897 |
| B5 / nonlinear | 32/32 | 2.07% | 2.07% | 3.904 |
| B5 / staggered | 31/32 | 3.95% | 3.15% | 3.888 |

B4 is the actual constrained nominal nearest-control projection. It is nearly inactive on healthy nominal plans. B5 is indistinguishable from B2 in these small pilots; projection did not add acceptance. All recorded projection calls passed the residual check. A final-only projection gave the same accepted count as B4, with stronger-nonlinear mean proposal risk 4.87% versus B4's 4.68%. Nominal projection can improve deterministic geometry without adequately controlling stochastic goal/collision failure.

The selected trajectory figure deliberately shows the largest nominal-risk task in each environment. It illustrates repair and failure modes; it is not a representative average or a separate performance estimate.

![Selected difficult trajectories](revision/outputs/summary/trajectories.png)

## Matched B6/B7 training experiment

Models share 4,000 examples, architecture, initialization, optimization steps, and training-noise streams. A separate dropout RNG and explicit missing-condition mask support proper conditional/unconditional training. Each independently labeled example receives a requested upper budget compatible with its finite-sample upper label. Reported sampling uses conditional scale 1, without CFG amplification. These nominal controls differ from the main 6,000-example baseline, so comparisons belong within each matched table.

### Ordinary plus marginal-clearance data, Delta = .10

| Method / setting | Plans accepted | Mean audit risk, all proposals | Mean audit risk, accepted | Mean J, accepted |
|---|---:|---:|---:|---:|
| B6 / ordinary + marginal | 81/96 | 7.19% | 3.24% | 3.825 |
| B7 / ordinary + marginal | 96/96 | 3.58% | 3.58% | 3.785 |
| MatchedNominal / ordinary + marginal | 79/96 | 9.15% | 3.60% | 3.770 |
| MatchedNominalChance / ordinary + marginal | 96/96 | 3.80% | 3.80% | 3.739 |

At Delta=.05, matched nominal + chance accepts 81/96 and B7 accepts 85/96. Their mean all-proposal costs are 3.747 and 3.797, respectively. This small acceptance gain is accompanied by higher cost and has not been established as statistically reliable across a larger independent task population. At Delta=.10 both accept 96/96.

### Ordinary data alone, Delta = .10

| Method / setting | Plans accepted | Mean audit risk, all proposals | Mean audit risk, accepted | Mean J, accepted |
|---|---:|---:|---:|---:|
| B6 / ordinary only | 91/96 | 5.56% | 3.19% | 3.912 |
| B7 / ordinary only | 96/96 | 3.25% | 3.25% | 3.895 |
| MatchedNominal / ordinary only | 90/96 | 5.70% | 3.22% | 3.900 |
| MatchedNominalChance / ordinary only | 96/96 | 3.22% | 3.22% | 3.882 |

At Delta=.05, both guided variants accept 88/96; at .10 both accept 96/96. Ordinary labeling supplies a useful but narrow measured risk range (~.006–.170); adding marginal examples broadens it to ~.010–.496 and also makes the matched unconditional proposal distribution riskier. A dedicated risk-aware planner is unnecessary to construct this pilot dataset. The experiment does not establish that its extra data pipeline improves the guided planner enough to justify using it by default.

## Answers to Q1–Q8

### Q1. What safety energy should be used?

The original local Gaussian union sum with softplus threshold is a useful inference energy, provided its output is treated as a surrogate. Its gradient reduces the approximate sum of marginal event probabilities, including changes to propagated covariance. Exact empirical upper-tail CVaR is the genuine alternative: its gradient reduces the magnitude of the worst signed trajectory losses. It can reduce tail severity without crossing the threshold necessary to meet the chance bound. Both use the same diffusion-score correction interface and independent validation.

### Q2. Is local Gaussian guidance sufficiently accurate for nonlinear execution?

It is useful for many tested proposals, but it is not a reliable probability upper bound. Calibration on ordinary, conservative routes alone hid an important failure. A conventional cost optimizer pushed a stronger-nonlinear trajectory near the requested boundary: predicted Gaussian sum **9.940%**, measured joint failure **11.580%**, two-sided 95% interval **[11.336%, 11.827%]**, using 65,536 fresh executions. It is rejected by independent validation.

A second 32,768-draw diagnostic decomposes the same trajectory:

| Quantity | Value |
|---|---:|
| Actual joint failure | 11.447% |
| Sum of exact event marginals | 11.874% |
| Sum of tangent-halfspace event marginals | 12.558% |
| Local Gaussian sum | 9.940% |

Union aggregation adds 0.427 percentage points and tangent geometry adds 0.684 points, but Gaussian propagation subtracts 2.618 points relative to empirical tangent marginals. Nonlinear moment/distribution error overwhelms those conservative terms. This case is a deliberately found counterexample, not an estimate of how frequently false-safe predictions occur. Its source, draw seeds, and reproduction script are saved.

![Calibration and error decomposition](revision/outputs/summary/calibration.png)

### Q3. Does CVaR beat chance guidance?

There is no general win. At moderate noise/.10, chance accepts 96/96 versus CVaR's 93/96, at essentially equal all-proposal J. On the extension seed, CVaR slightly lowers moderate-noise proposal risk, while chance handles the strongest nonlinear failed nominal case much better. Chance accepts 32/32 versus CVaR's 31/32 there. Conditional accepted-risk averages alone would misleadingly reward the method that rejects more difficult plans.

In this implementation K=256 CVaR is faster than full differentiable Gaussian covariance guidance on the measured GPU; analytical does not automatically mean faster. Thirty-two guidance stages evaluate 8,192 rollout-trajectories per plan from 256 unique fixed noise paths. Chance guidance uses no sampled execution paths internally. Both require independent validation.

### Q4. Does projection help?

It provides an explicitly checked deterministic repair, but adds no accepted plans beyond chance guidance in these runs. Once the nominal gate passes, the earlier large projection advantage disappears. Keep projection as an optional feasibility repair and comparator; current evidence does not justify paying for it on every sample. No global nearest-point or stochastic safety guarantee is claimed.

### Q5. Is risk-aware training necessary?

It is not necessary for the successful nominal + chance pilot. Matched conditioning yields a small strict-budget acceptance gain on mixed-clearance data and no gain after guidance on ordinary data. Correction counts remain fixed by the sampler; this experiment does not demonstrate reduced denoising iterations or a planning-speed advantage. Broader conditions and adaptive stopping would be needed to establish those benefits.

### Q6. How should labeled data be generated?

Start with ordinary feasible controls and independent rollout labels. Include marginal-clearance feasible routes if broader risk support is needed. Use an explicit requested-budget semantics and missing-condition mask; avoid labeling a path's measured risk as though it were the requested budget. Here each data regime uses 4,000 × 512 = 2,048,000 label executions, reused across matched models and seeds. Labels are estimates, not certificates.

### Q7. How should guidance strength be chosen?

The separate 24-task tuning set sweeps lambda={0,.5,2,8,32}, Delta={.05,.10,.20}, CVaR K={64,256,1024}, and chance gamma={10,30,100}; exact hard CVaR requires no beta smoothing parameter. At Delta=.10, chance mean risk falls 4.91% → 4.16% → 3.52% → 3.23% → 3.01% as lambda increases; acceptance is 23,23,23,24,24 out of 24. CVaR risks are 4.91%, 4.78%, 4.45%, 3.49%, 2.93%, with acceptance 23,23,23,23,24. Lambda=8 is the working default, not a universal optimum.

The K=64 and 1024 CVaR probes at .05 each accept 23/24, with risks 3.41% and 3.53%; larger K was not automatically better. Chance gamma=10 and 100 each accept 24/24 at .10, with risks 3.20% and 3.28%. A cost-weight sweep (.03,.1,.3) lowers J but can reduce acceptance at the strictest budget. These small sweeps characterize sensitivity rather than establish a tuned optimum.

Repeated identical-condition sampling (five new conditions, 32 draws each, three checkpoints) gives upper-route fractions B0=.548, B2=.546, B3=.548 and mean pairwise nominal path RMS .692, .693, .694. Both route families survive at lambda=8. This coarse diversity check does not establish stability at arbitrary strengths or rare modes.

![Safety and cost tradeoff](revision/outputs/summary/safety_cost.png)

### Q8. Does Delta control realized risk?

It controls acceptance and changes guidance, but currently gives weak cost/risk adaptation. For moderate noise and Delta=.05/.10/.20, B2 accepts 91/96, 96/96, 96/96; its mean proposal risks are 3.08%, 3.18%, 3.41%. B0 accepts 83/96, 91/96, 93/96; B3 accepts 87/96, 93/96, 96/96. All are conditional empirical results.

Realized risk need not equal Delta: it is an upper constraint. The unresolved issue is whether a looser budget buys appreciably lower cost. The conservative nominal prior and risk-only energy exploit that tradeoff weakly. At present this is a useful validated proposal mechanism, not a demonstrated cost-optimal risk-budget controller.

![Requested risk versus measured risk and acceptance](revision/outputs/summary/requested_risk.png)

## Conventional cost optimizer: essential competing explanation

An additional SLSQP optimizer minimizes explicit J subject to the same chance surrogate, from waypoint or diffusion initializations. The initial 80-iteration/.10 run contains many solver failures and the confirmed Gaussian false-safe case. It is retained, not filtered away. A separate 300-iteration run with internal surrogate threshold .08 improves convergence.

On 12 moderate-noise tasks, the buffered waypoint and diffusion-final optimizers both converge and independently accept 12/12, with mean audit risk ~7.66% and J ~3.221/3.216. These are shorter than the diffusion-guided routes, though they use more of the allowed risk budget; this is not a matched-risk claim. They require roughly .31/.28 seconds of optimization per plan, excluding proposal and validation. The small nonlinear extension remains problematic: waypoint accepts 4/12 and diffusion-final 5/12; two diffusion-initialized solves fail. A fixed 20% surrogate buffer is insufficient for reliable nonlinear calibration.

The conventional comparator exposes both the cost opportunity and the surrogate weakness. These results do not establish a performance advantage for diffusion over conventional stochastic planning.

## Runtime, rollout accounting, and verification

Times below average three measured 32-plan batches on the documented RTX 4090 environment. They include the 4,096-draw independent validator. Research-only 8,192-draw audit cost is excluded; batch throughput is not single-query latency. Projection timing is workload dependent and often skips feasible plans.

| Method | Sample / guidance / projection | Validation | Total per 32-plan batch |
|---|---:|---:|---:|
| B0 | 0.009 s | 0.156 s | 0.165 s |
| B2 | 0.281 s | 0.156 s | 0.437 s |
| B3 | 0.184 s | 0.156 s | 0.340 s |
| B4 | 0.064 s | 0.159 s | 0.223 s |
| B5 | 0.334 s | 0.158 s | 0.492 s |

All variants use 64 denoising steps; chance/CVaR corrections apply at 32 stages. CVaR uses 256 unique noise paths, 8,192 repeated rollout evaluations, plus validation and audit. Risk conditioning costs an additional labeling pass; it does not remove validation. Per-plan JSONs preserve execution confidence intervals, predicted risks, goal success, costs, CVaR, projection diagnostics, and timing.

Mathematical/end-to-end checks pass: covariance propagation, exact fractional CVaR/gradients, chance finite differences, NumPy evaluation and exact binomial bounds, raw B0 sampler equivalence, accepted/rejected public planner behavior, and constrained projection feasibility/idempotence. Parent review corrected evaluator interval/test errors and public-planner/projection edge cases from delegated work.

## Limits and reproducibility

Run `bash risk_diffusion/revision/reproduce.sh` from the workspace root. Saved checkpoints, per-plan JSON/NPZ, separate tuning outputs, boundary audit, repeated-condition diversity, and plotting scripts accompany these results. This is a small known-model 2D investigation: two layouts, limited dynamics/noise regimes, three main training seeds but only one per extension, discrete-time collision events, open-loop controls, and a narrow training generator. Model mismatch, continuous collision safety, large-map generalization, higher-dimensional robotics, adaptive stopping, and a broad conventional-planner comparison remain unresolved. The study has evidence for a simpler implementation and clear failure mechanisms; it has not established a conference-level robotics contribution.
