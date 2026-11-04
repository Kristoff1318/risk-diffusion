# Held-out full-arm risk planning: terminal outcome

All 64 requests completed on the frozen 8-task cohort; no exception or timeout occurred. Conventional final-only CVaR correction increased classical admission from 3/8 to 6/8 at 5% risk, and 4/8 to 6/8 at 10%. Both ordinary MPD and MPD+the same correction admitted 0/8 at both budgets. No selected accepted plan had audited point risk or lower 95% bound above its requested budget. This is conditional simulated evidence, not a safety guarantee.

## Frozen cohort and benchmark

Source official validation positions 128:384: 256 examined, 207 eligible, 8 selected by the prespecified nominal clearance ranks [0,.02,.1,.25,.4,.55,.75,.95]. Exact ordered IDs: 1959097,572764,1412961,1930902,1635282,990634,693071,42576. Reference clearances: 2.935,6.732,13.308,17.105,18.700,21.574,31.536,73.729 mm. All source IDs, exclusions and nominal checks are in TASK_SELECTION.json. No stochastic filtering, replacement cohort, obstacle changes or noise/controller tuning occurred. Cached reference paths served only as task descriptors and were never planner initializers.

All arms use 24 free seven-joint controls, degree 5 spline with tripled start/end controls, 200 velocity commands over 10 s. Endpoint joints remain free; goal is the fixed prescribed panda_hand pose within 1 cm/2 deg. Classical proposals use two fresh full-pose DLS seeds, then 80 nominal trajectory steps each. MPD uses the genuine frozen checkpoint with two fresh unguided DDIM 15 draws. Initialization/generation time is charged every time; no historical proposals are scored as new output.

The phase6 controller/physics/geometry is unchanged: perfect joint position observed every 50 ms; nominal feedforward minus 2/s tracking error; physical initial offset. Main regime is the prespecified **doubled** synthetic noise: 4 mrad initial, 2 mrad/s persistent bias, 10 mrad/s step velocity noise. No encoder bias/delay, actuator saturation, dynamics/contact simulation or measured hardware accuracy is modeled. Requested commands equal applied commands; nominal requested-command violations and sampled command/executed-velocity failures reject. Both native world/self configurations, joint limits and terminal pose are hard events on 5 ms grid; sphere coverage and continuous-time safety are not proved. The native scene has 10 fixed spheres, no table.

Common physical cost J=dt sum(u_nom²)+.005dt sum((du_nom/dt)²). This is nominal kinematic effort/smoothness, not torque energy. Both CVaR arms apply exactly the same final-only routine: 32 shared physical draws, exact fractional tail at Delta=.05/.10, full 5 ms maximum signed normalized world/self/joint/command/executed/pose loss, lambda 500, normalized buffer .05, 8 Adam steps at .003 and gradient norm clip 20. Metres/radians are normalized by fixed .01 m/.02 rad/.1 rad/s/goal tolerances; zero preserves the failure sign. Return the lowest-cost sampled-feasible completed iterate, otherwise minimum sampled CVaR. Finite empirical CVaR is not a chance certificate.

Nominal classical refinement uses 50 ms geometry and averaged squared penalties; CVaR correction uses 5 ms trajectory maxima. This distinction, additional iterations and additional compute limit attribution of gains uniquely to stochastic risk. The 4-arm design does not include a same-compute deterministic final-correction ablation. It does isolate the same final risk correction on classical versus diffusion proposals.

## Independent admission, all proposals and selected audits

Fresh 1,024-draw admission for each of two candidates uses exact one-sided beta upper bound at alpha=.05/2. The minimum-cost admitted candidate is selected. Optimization, admission and audit seeds are entirely separate; common draws couple method comparisons. Every proposal is audited on 2,048 independent draws; a selected plan gets 4,096 draws from the same prespecified audit stream. All-proposal aggregates use the same 2,048 prefix even for selected plans. Audits never select, repair or change admission. Confidence is per fixed candidate batch/request, not a simultaneous guarantee across the whole study.

The requested 4,096/8,192 pair was not feasible for all 128 proposals under the cap: profiling 5.36/10.73 s already exceeds the total allowance before optimization. Fixed sample sizes were frozen before main results. Exact duplicate task/trajectory/samplecount audit reuse is reported; every fresh generation/refinement and every admission is charged. Audits are timed separately from request decisions.

|Budget|Arm|Accepted/8|All-proposal risk|Accepted audited risk|Mean request|
|---|---|---:|---:|---:|---:|
|5%|classical|3|65.28%|0.008%|5.41s|
|5%|classical_cvar|6|51.83%|0.244%|13.78s|
|5%|mpd|0|100.00%|—|2.61s|
|5%|mpd_cvar|0|99.97%|—|10.98s|
|10%|classical|4|65.28%|1.099%|5.38s|
|10%|classical_cvar|6|51.96%|0.525%|13.73s|
|10%|mpd|0|100.00%|—|2.60s|
|10%|mpd_cvar|0|99.93%|—|10.97s|

All-proposal means include 16 proposals/arm/budget and all failures; accepted means are conditional on different task subsets and must not be treated as a paired cost advantage. Per-proposal event decomposition, admission counts/upper bounds and audit confidence intervals are in PER_PROPOSAL.csv. Every selected plan and rejected request is retained in PER_REQUEST.csv; complete optimizer traces and packed per-execution flags are under outcomes/. BEST_ACCEPTED_PATHS.npz holds selected nominal trajectories/control coefficients.

## Paired findings and scientific conclusion

At 5%, classical/CVaR common accepted tasks are [1, 4, 5]; mean paired cost change -0.821278, mean audited risk change 0.0000 percentage points.
At 10%, classical/CVaR common accepted tasks are [1, 3, 4, 5]; mean paired cost change -0.825505, mean audited risk change -1.0925 percentage points.

Representative task 3 (ID 1930902): uncorrected classical is rejected at 5% but admitted at 10%, with independent 179/4096 joint-limit failures (4.370%, exact 95% interval in PER_REQUEST.csv). The common correction has 0/4096 failures and reduces J from 10.586859 to 9.771694. The raw nominal path passes zero-noise checks; this is an execution-uncertainty issue. Task 0 (ID 1959097) is a different kind of gain: the uncorrected candidate has nominal self-collision; correction repairs it and admits a plan with 35/4096 combined audit failures at 5%. Other repairs include nominal feasibility improvements; they are identified in NOMINAL_DIAGNOSTICS.json rather than presented as uniquely stochastic gains.

The two-start classical plus final CVaR pipeline adds useful practical coverage in this small regime at higher latency. Its acceptance/cost/risk improvement is measured; unique benefit over a same-effort deterministic correction is not established. With two proposals and eight correction steps, MPD provides no admitted plan and no demonstrated capability beyond classical proposals. This negative result is specific to the genuine checkpoint/DDIM 15 and frozen bounded pipeline; it does not establish that all diffusion planners fail. No further cohort or denoising-guidance search was run.

Neither selected classical pipeline solves all eight tasks. Task-specific rejection/failure traces remain in the report rather than being replaced. Nominal-only cohort selection did not ensure our bounded planners could find each recorded reference route. The nominal references are feasibility descriptors, not free classical solutions. A subsequent decision should address the documented failing proposal/repair behavior on this same cohort or add a same-effort deterministic correction ablation; do not move obstacles, increase noise, or find another cohort to manufacture a diffusion benefit.

## Effort and resources

Main process 811.010s; CPU user/system 809.38/2.00s. GPU peak allocated 682.6MiB; main RSS3284.8MiB. Core/model initialization0.116/0.284s, measured once and cold/amortized equivalents reported per request. Request latency includes fresh generation/correction, two admissions and selection; research audit is separate. No equal-compute claim: nominal arms use 160 trajectory gradients plus 80 batched DLS updates per request, risk arms add 16 optimizer gradients/512 physical-scenario gradient evaluations, MPD uses 15 denoiser calls per two-proposal batch. All share a 15 s proposal/correction cap. GPU-backed stage timings are synchronized wall times, not separately metered GPU kernel busy seconds.

Full stage timings, effort counts, exact audit reuse hits and request latency distributions are in ARM_SUMMARY.csv and the resource ledger. No training, downloads, installations, hardware or changes to original assets/defaults occurred. Geometry/checkpoint/normalization provenance hashes were verified. P0/P6 and original project preservation checks accompany the final status.

## Reviewed failure diagnosis and preservation

The MPD arms lack the nominal collision/pose refinement used in the historically successful MPD pipeline. All raw and corrected MPD proposals fail existing zero-noise validation. Their negative outcomes primarily diagnose insufficient nominal proposal repair, with secondary execution-risk failures; see FAILURE_DIAGNOSIS.md and the complete FAILURE_DIAGNOSIS.csv. This is not evidence that stochastic correction fails on nominally healthy MPD paths.

RESOURCE_LEDGER.json books 845.60 s metered commands plus a conservative 300 s allowance for unmetered inspection, edits, review, upload and commits: 1,145.60 / 1,200 s. This is an estimated conservative charge, not a measured total. The first CSV analysis failure is included. Original 225 files and 89 phase5/phase6 files were hash verified unchanged. Frozen source and task-bank hashes match.
