# Frozen fresh confirmation: terminal nominal-health gate failure

The authorized study stopped at its prespecified nominal-health gate. Classical supplied a healthy prepared candidate on2/4 fixed tasks; MPD on3/4. Both sources were required to reach3/4. No task replacement, extra IK seed, extra nominal step, correction or stochastic validation followed. GPU slot released.

This is a completed gate decision, not a completed risk-method comparison. MAIN_RESULTS.complete=false records that the16 planned method requests did not run. Their accepted=false placeholders MUST NOT be read as stochastic rejections. The initial analytic joint floor rules out none of the four tasks; machine-underflow zeros are not exact zero risk.

## Fixed cohort and observed preparation

|Task ID / position|Descriptor|Initial joint-event probability|Classical healthy slots|MPD healthy slots|Terminal method status|
|---|---|---:|---:|---:|---|
|158032 / 384|valid|numeric underflow|1/2|2/2|NOT_RUN_NOMINAL_HEALTH_GATE|
|175075 / 448|self_curobo|2.64209e-269|not generated|not generated|NOT_RUN_INVALID_DESCRIPTOR|
|1171877 / 512|valid|7.02027e-58|1/2|2/2|NOT_RUN_NOMINAL_HEALTH_GATE|
|1820403 / 576|valid|numeric underflow|0/2|2/2|NOT_RUN_NOMINAL_HEALTH_GATE|

Task175075 has a native cuRobo self-collision already at its nominal start/reference (minimum clearance-3.522mm). It fails the frozen descriptor validity gate. This is NOT a computed Gaussian collision-risk lower bound. The other three descriptors pass; none was substituted.

Every generated slot is retained below, including failures. J is nominal effort; nominal validity is a deterministic zero-error hard-event check, not a chance constraint or admission. Generic risk/CI fields in raw nominal_check results come from a one-execution deterministic validator and are NOT Monte Carlo risk estimates.

|Task|Source / slot|Nominal valid|J|Failing hard components|Source two-slot generation +80-step preparation, seconds|
|---|---|---|---:|---|---:|
|158032|classical / 0|True|3.103350|none|2.900782|
|158032|classical / 1|False|21.513783|joint;command_velocity;executed_velocity;orientation|2.900782|
|158032|mpd / 0|True|8.986035|none|1.877429|
|158032|mpd / 1|True|4.765619|none|1.877429|
|1171877|classical / 0|False|7.339637|joint;command_velocity;executed_velocity;position;orientation|2.808767|
|1171877|classical / 1|True|5.147854|none|2.808767|
|1171877|mpd / 0|True|10.014012|none|1.769487|
|1171877|mpd / 1|True|3.713913|none|1.769487|
|1820403|classical / 0|False|6.609834|world_mpd;world_curobo;joint|2.815318|
|1820403|classical / 1|False|5.880913|world_mpd;joint|2.815318|
|1820403|mpd / 0|True|7.390832|none|1.780397|
|1820403|mpd / 1|True|3.175864|none|1.780397|

The source time is shared by its two slots and two intended variants; repeated table entries are not extra executed time. All12 nominal refinements completed80 steps (960 gradient calls). Classical ran three fresh batched80-step full-pose IK procedures with two seeds each. MPD ran three fresh two-sample DDIM15 requests (45 denoiser calls) and six nominal refinements. The six MPD and two classical paths passed the nominal hard checks; all other classical slots remain saved.

## Actual stochastic counts and comparison limits

Admission:0 executions; corrected unselected audits:0; selected audits:0; prepared-input stochastic diagnostics:0; correction steps:0 for either method. No optimizer risk scenarios or risk draw streams were generated. There were16 deterministic full-event checks (four references plus12 prepared slots) and separate native endpoint checks.

Thus per-task stochastic acceptance, audited risk/CI, risk-correction cost/latency, surrogate calibration and dense-versus-native gain/loss are NOT MEASURED. No native energy was evaluated on fresh paths after the gate failed. No historical CVaR timing superiority or nonlinear safety claim follows.

The evidence concerns nominal initialization health: repaired MPD supplies six valid slots on the three eligible cases, whereas classical supplies two. This is a small fixed-cohort observation with an invalid fourth descriptor, not a successful risk-aware planning confirmation. It does not establish no risk headroom; that question was not reached.

## Timing and resources

Source metadata load/select:2.145850s; Core/CUDA/native assets initialization:0.113662s; genuine MPD model initialization:0.262638s; native invariant cache:0.013094s.

Fresh generation + nominal preparation total:13.952180s. Per-source warm timings are in the table; no end-to-end risk request was completed, so no cold/admission request latency is manufactured. Initializations and offline1.77GB metadata load are separately charged.

Runner process:18.062612s; timed command18.42s. Peak allocated GPU170,459,136bytes; RSS4,606,632KiB. No timeout, nonfinite or iteration shortening triggered the gate; the failure is nominal health. The original fully batched validator was used unchanged, no adaptive adapter.

Pre-outcome full-plan forecast290.21s fitted the309s remainder with unchanged sample counts and39s report/commit reserve. The prescribed gate stopped early; no counts were reduced to fit outcomes. Additional booking57.42s =18.42s measured command +39s estimated ancillary reserve (includes protocol/code/hash/report/preservation/commit work). Cumulative198.42/450s; remaining251.58s. Reserve is conservative bookkeeping, not a measured ancillary total.

Protocol committed7d53254 before outcomes; original asset/code hashes are checked and preserved. Original/default code, prior results and phase10/phase11 specifications remain unchanged. New sources/paths/tasks/nominal traces are preserved inphase12; no copied external assets.

## Decision

Stop this variant at the frozen gate. Do not replace175075, repair classical with extra steps/seeds, weaken model events, or run a surviving MPD-only comparison under this protocol. Any different study requires a new scientific question and explicit authorization. The covariance-energy risk benefit remains unconfirmed on fresh tasks. GPU slot released.
