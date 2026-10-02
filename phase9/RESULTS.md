# Equal-nominal diagnostic replay: terminal results

This is diagnostic reuse of the same eight tasks and saved proposals, not fresh held-out evidence or a cold end-to-end speed comparison. Phase7 raw MPD lacked nominal repair; phase8 restored nominal feasibility. Here BOTH families start from their exactly matching 80-step nominal outputs. All original invalid slots remain.

Terminal: 96/96 completed requests; complete=True.

## Frozen comparison

Classical uses saved phase7 nominal controls; MPD uses saved phase8 repaired controls. Exact source/inputs/preparation settings are hash checked. Classical historical initialization includes two 80-step full-pose DLS IK seeds, MPD historical initialization uses two DDIM samples. Both nominal preparers use 80 gradients/slot. We reuse these paths only as a diagnostic and report prior timing rather than pretending to regenerate them. Goals are free-terminal-joint panda_hand poses, never fixed joint endpoints or projections.

Each family has no extra correction, eight dense deterministic steps, or eight sampled-CVaR steps. Corrected branches use identical 5 ms full-event signed normalized maximum loss, cost J, Adam .003, gradient cap20, lambda500, buffer.05 and best-completed-iterate selection. Deterministic uses one zero-error execution (degenerate tail, delta=1); CVaR uses the exact fractional .05/.10 tail over32 common physical executions. The sole declared change is the stochastic tail/physical scenarios. Steps are matched; scenario-gradient compute is not. Actual incremental timings are reported.

Physics, ideal-observed K2 tracking, 10s/200steps, doubled synthetic noise, both native world/self geometry, joint/command/executedvelocity and 1cm/2deg pose events are unchanged. All validation uses 5ms states. Finite simulation and sphere/discrete geometry do not imply hardware safety.

## Summary

All-proposal mean uses all16 paths/arm/budget, including every failure, on a common1024 audit prefix. Accepted means are conditional on different task subsets.

|Budget|Arm|Accepted/8|Nominal valid/16|All-proposal risk|Mean J all|Accepted risk|Correction s|Incremental request s|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|5%|classical_none|3|7|65.393%|6.6694|0.000%|0.000|2.565|
|5%|classical_dense|5|8|53.455%|6.3758|0.664%|0.281|2.836|
|5%|classical_cvar|5|8|52.014%|6.3950|0.127%|8.313|10.869|
|5%|mpd_none|7|15|10.449%|5.0858|0.000%|0.000|2.556|
|5%|mpd_dense|7|15|15.674%|4.9324|0.837%|0.256|2.813|
|5%|mpd_cvar|7|15|9.674%|4.9462|0.140%|8.266|10.824|
|10%|classical_none|4|7|65.393%|6.6694|0.867%|0.000|2.556|
|10%|classical_dense|5|8|53.455%|6.3758|0.664%|0.255|2.812|
|10%|classical_cvar|6|8|51.947%|6.3924|0.732%|8.281|10.840|
|10%|mpd_none|7|15|10.449%|5.0858|0.000%|0.000|2.557|
|10%|mpd_dense|7|15|15.674%|4.9324|0.837%|0.259|2.816|
|10%|mpd_cvar|7|15|9.985%|4.9504|0.488%|8.336|10.894|

## Per-task paired outcomes

Entries are accepted selected-plan J and fresh audited risk with exact95% CI, or REJECTED. Rejected slots remain in PER_PROPOSAL.csv with confidence bounds and all component counts/margins. Event counts overlap and cannot be summed into union failures.


### Budget 5%

|Task ID|Family|Nominal prepared|Dense correction|CVaR correction|
|---|---|---|---|---|
|1959097|classical|REJECTED|J 4.817; 1.66% [1.15%,2.31%]|REJECTED|
|1959097|mpd|J 4.073; 0.00% [0.00%,0.18%]|J 5.519; 1.66% [1.15%,2.31%]|J 3.956; 0.05% [0.00%,0.27%]|
|572764|classical|J 2.162; 0.00% [0.00%,0.18%]|J 2.017; 0.00% [0.00%,0.18%]|J 2.027; 0.00% [0.00%,0.18%]|
|572764|mpd|J 3.145; 0.00% [0.00%,0.18%]|J 3.066; 0.00% [0.00%,0.18%]|J 3.076; 0.00% [0.00%,0.18%]|
|1412961|classical|REJECTED|REJECTED|J 0.418; 0.63% [0.34%,1.08%]|
|1412961|mpd|J 2.247; 0.00% [0.00%,0.18%]|J 2.114; 0.00% [0.00%,0.18%]|J 2.114; 0.00% [0.00%,0.18%]|
|1930902|classical|REJECTED|J 9.672; 1.66% [1.15%,2.31%]|J 9.793; 0.00% [0.00%,0.18%]|
|1930902|mpd|J 6.521; 0.00% [0.00%,0.18%]|J 6.245; 1.46% [0.99%,2.08%]|J 6.256; 0.00% [0.00%,0.18%]|
|1635282|classical|J 3.910; 0.00% [0.00%,0.18%]|J 3.674; 0.00% [0.00%,0.18%]|J 3.675; 0.00% [0.00%,0.18%]|
|1635282|mpd|J 4.264; 0.00% [0.00%,0.18%]|J 4.172; 0.88% [0.52%,1.39%]|J 4.172; 0.88% [0.52%,1.39%]|
|990634|classical|J 2.659; 0.00% [0.00%,0.18%]|J 0.546; 0.00% [0.00%,0.18%]|J 0.549; 0.00% [0.00%,0.18%]|
|990634|mpd|J 2.811; 0.00% [0.00%,0.18%]|J 2.751; 0.00% [0.00%,0.18%]|J 2.759; 0.00% [0.00%,0.18%]|
|693071|classical|REJECTED|REJECTED|REJECTED|
|693071|mpd|J 1.080; 0.00% [0.00%,0.18%]|J 1.044; 1.86% [1.32%,2.54%]|J 1.048; 0.05% [0.00%,0.27%]|
|42576|classical|REJECTED|REJECTED|REJECTED|
|42576|mpd|REJECTED|REJECTED|REJECTED|

### Budget 10%

|Task ID|Family|Nominal prepared|Dense correction|CVaR correction|
|---|---|---|---|---|
|1959097|classical|REJECTED|J 4.817; 1.66% [1.15%,2.31%]|J 4.776; 4.10% [3.28%,5.05%]|
|1959097|mpd|J 4.073; 0.00% [0.00%,0.18%]|J 5.519; 1.66% [1.15%,2.31%]|J 3.952; 0.54% [0.27%,0.96%]|
|572764|classical|J 2.162; 0.00% [0.00%,0.18%]|J 2.017; 0.00% [0.00%,0.18%]|J 2.028; 0.00% [0.00%,0.18%]|
|572764|mpd|J 3.145; 0.00% [0.00%,0.18%]|J 3.066; 0.00% [0.00%,0.18%]|J 3.076; 0.00% [0.00%,0.18%]|
|1412961|classical|REJECTED|REJECTED|J 0.417; 0.29% [0.11%,0.64%]|
|1412961|mpd|J 2.247; 0.00% [0.00%,0.18%]|J 2.114; 0.00% [0.00%,0.18%]|J 2.115; 0.00% [0.00%,0.18%]|
|1930902|classical|J 10.587; 3.47% [2.72%,4.35%]|J 9.672; 1.66% [1.15%,2.31%]|J 9.734; 0.00% [0.00%,0.18%]|
|1930902|mpd|J 6.521; 0.00% [0.00%,0.18%]|J 6.245; 1.46% [0.99%,2.08%]|J 6.254; 0.15% [0.03%,0.43%]|
|1635282|classical|J 3.910; 0.00% [0.00%,0.18%]|J 3.674; 0.00% [0.00%,0.18%]|J 3.675; 0.00% [0.00%,0.18%]|
|1635282|mpd|J 4.264; 0.00% [0.00%,0.18%]|J 4.172; 0.88% [0.52%,1.39%]|J 4.172; 0.88% [0.52%,1.39%]|
|990634|classical|J 2.659; 0.00% [0.00%,0.18%]|J 0.546; 0.00% [0.00%,0.18%]|J 0.549; 0.00% [0.00%,0.18%]|
|990634|mpd|J 2.811; 0.00% [0.00%,0.18%]|J 2.751; 0.00% [0.00%,0.18%]|J 2.755; 0.00% [0.00%,0.18%]|
|693071|classical|REJECTED|REJECTED|REJECTED|
|693071|mpd|J 1.080; 0.00% [0.00%,0.18%]|J 1.044; 1.86% [1.32%,2.54%]|J 1.044; 1.86% [1.32%,2.54%]|
|42576|classical|REJECTED|REJECTED|REJECTED|
|42576|mpd|REJECTED|REJECTED|REJECTED|

## Dense versus CVaR: paired evidence

classical, 5%: CVaR-only accepted [1412961]; dense-only accepted [1959097]; common accepted 4. Paired mean CVaR-minus-dense J 0.033337, risk -0.004150.
mpd, 5%: CVaR-only accepted []; dense-only accepted []; common accepted 7. Paired mean CVaR-minus-dense J -0.218530, risk -0.006975.
classical, 10%: CVaR-only accepted [1412961]; dense-only accepted []; common accepted 5. Paired mean CVaR-minus-dense J 0.007137, risk 0.001563.
mpd, 10%: CVaR-only accepted []; dense-only accepted []; common accepted 7. Paired mean CVaR-minus-dense J -0.220525, risk -0.003488.

No superiority follows from a lower accepted-subset mean alone. Within each task, proposals and draws are paired. Optimizer/admission/audit seeds2026105100/5200/5300+task are fresh and separate from prior phases; audits never select/tune. Exact one-sided beta upper(.975,k+1,n-k), alpha=.05/2 for fixed two-candidate batch, plus finite/nominal-command checks determines admission, not point estimates. Lowest J admitted candidate is selected. Confidence is per request, not simultaneous over96requests. Selected audit2048 and unselected1024 share the same independent audit stream; their all-proposal1024 prefixes are directly comparable. Exact task/pathhash/samplecount reuse is reported.

## Latency and resource accounting

Main process 736.995 s, CPU user/system 736.62/1.04 s. Peak allocated GPU 664.9 MiB; RSS 2174.0 MiB. GPU-backed timings are synchronized wall, not kernel occupancy. Shared core initialization 0.126 s measured once. Incremental request includes cached loading, nominal checks, fresh correction/admission; research audit is separate. No generation/IK/80-step prep was rerun.

Historical generation/IK/nominal preparation per task is retained in MAIN_RESULTS.json task_preparation; prior meters are not silently added to claim cold end-to-end speed. Classical preparation includes IK, MPD generation and phase8 preparation are separate historical calls.

Counts were frozen before outcomes:1024admission/1024all-proposal audit/2048selected audit for every arm. Prior timing profiles and exact duplicate reuse enabled the900s cap; larger2048/4096audit would exceed it with preservation reserve. Every admission and correction is freshly charged. Failures/timeouts/nonfinite optimizers are reported in ARM_SUMMARY.csv; incomplete requests are explicit rather than dropped. No new tasks/proposal samples/model/noise tuning, installs/downloads/training/hardware.

Previous phases remain byte-identical to PREVIOUS_MANIFEST.json. No additional variant or follow-on is authorized by this report.

## Irreducible fixed-start risk under the declared model

Task 42576 has only 2.528 mrad start-joint clearance versus 4 mrad independent Gaussian initial offset. The analytic initial joint failure probability is 26.5826%, computed as one minus the product of per-joint Gaussian probabilities inside fixed limits. Full-trajectory union risk is at least this initial event risk, regardless of subsequent controls. Both 5%/10% targets are therefore infeasible for this task in the unchanged model. No cohort replacement or noise change occurred. This is a conditional model fact, not hardware calibration; INITIAL_RISK_FLOOR.json records all eight analytic values.

## Scientific interpretation

CVaR does not show an unconditional advantage over the same-eight-step dense deterministic correction. At 5% classical coverage is 5/8 for each, but CVaR gains task 1412961 and loses task 1959097. On their four common admitted tasks, CVaR-minus-dense mean J is +0.033337 and audited risk -0.4150 percentage points. At 10%, CVaR adds task 1412961 (6/8 versus 5/8); on five common admitted tasks its mean J is +0.007137 and audited risk +0.1563 percentage points. Coverage and risk/cost changes must therefore be reported together.

MPD's paired seven admitted tasks are the same for all variants. At 5%, CVaR-minus-dense mean J is -0.218530 and audited risk -0.6975 percentage points; at 10%, -0.220525 and -0.3488 percentage points. This is a useful diagnostic cost/risk improvement over deterministic cost-driven correction, not added coverage. Prepared MPD with no extra correction already has zero observed failures in each selected 2,048-draw audit (each exact 95% upper bound about 0.18%); extra correction saves some nominal cost while spending risk margin. Zero observed failures is not zero true risk.

MPD initialization adds diagnostic capability after equal nominal preparation: 7/8 no-extra admissions at both budgets versus classical 3/8 and 4/8. All-proposal risk is 10.45% versus classical 65.39%; every invalid slot is included. Nominal feasibility is 15/16 versus 7/16. This evidence is specific to reused proposals, one eight-task stratum and the fixed synthetic model. Classical still has explicit IK initialization; initialization is the factor being compared, and its generation privilege/cost is disclosed. No broad generalization or publication-level proof follows.

Correction steps are equal, computational effort is not: about 8.3 s/request for 32-scenario CVaR versus 0.26 s for one-scenario dense deterministic correction. Incremental correction-plus-admission requests are about 10.8–10.9 s versus 2.8 s, and 2.56 s with no correction. Historical preparation times are tabulated in HISTORICAL_PREPARATION.csv; none are silently charged as freshly executed cold latency. No selected audit point estimate or lower 95% confidence bound exceeds its requested budget. All 96 requests completed without exception, timeout or nonfinite optimizer loss.

Full combined union counts/intervals and overlapping individual event counts/rates/margins are in PER_PROPOSAL.csv and per-request raw JSON. Selected MPD dense/corrected failures arise from world/pose events; task 42576 has the fixed-start joint-risk floor. Classical corrected selected failures include cuRobo self-collision and pose events. Component counts overlap and are not additive. The complete report retains failures, including classical task 693071 and the universal rejection of 42576.
