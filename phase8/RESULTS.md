# Nominal MPD health repair: complete

**15/16 repaired proposals pass independent dense nominal validation; all eight tasks have a feasible repaired proposal.** Raw feasibility was 0/16. Saved original classical proposals have 7/16 nominally feasible paths, covering six tasks. This establishes nominal health, not stochastic acceptance or a fair end-to-end diffusion advantage. Phase7 remains an explicitly confounded comparison of raw MPD with IK-plus-nominally-refined classical paths.

## Exact frozen procedure

Code/protocol were committed in 0980f7a before outcomes. The same 16 saved raw MPD controls were reused, one pair per task. Frozen phase7 core.refine was called with delta=None, err=None, 80 steps, lr=.025, Adam, gradient clipping 20, and best-completed-iterate nominal selection. The objective is exactly the classical nominal objective: 50 ms averaged squared pose/world/self/joint/velocity penalties plus effort and smoothness. All 16 completed 80 gradients (1,280 total); no timeout, exception or nonfinite outcome.

No IK, target-joint attraction, endpoint projection, seed replacement, model inference, RNG draw, physical noise, CVaR or risk selection occurred. MPD seeds were retained and changed only by optimization. Goal is the same panda_hand pose with free terminal joints. Classical had two 80-step full-pose DLS IK seeds before nominal optimization; its different initialization privilege remains visible. No joint target was introduced to either nominal objective.

Representation stays 24 learned seven-joint controls, 29 total degree-5 spline controls with tripled start/learned end, 10 s duration, 201 states, 200 commands diff(q)/.05. Nominal execution has zero noise/tracking error. All raw/repaired paths and controls are in outcomes/PATHS.npz; every candidate has a full trace JSON.

## Independent validation

Independent NumPy URDF FK checks both native world/self sphere models and all configured pairs, joint bounds, finite values, terminal pose via SciPy rotation, and physical finite-difference requested/applied nominal velocities. It checks 2,001 states on the 5 ms piecewise-command grid, requiring 1 cm/2 deg pose tolerances. Qualified GPU validation agreed on all flags and margins to <1e-10; saved raw failures were independently reproduced. These are discrete nominal sphere checks, not continuous-time/hardware guarantees. Acceleration/jerk are outside the preserved hard-event contract.

## Complete task comparison

Minimum J among nominally valid slots per task is descriptive, not risk admission. J is integrated nominal squared velocity plus .005-weighted squared acceleration, not torque energy. Every failed slot is retained in PER_PROPOSAL.csv.

|Task ID|Repaired MPD valid / 2|Best valid MPD J|Saved classical valid / 2|Best valid classical J|
|---|---:|---:|---:|---:|
|1959097|2|4.072842|0|—|
|572764|2|3.144872|1|2.162342|
|1412961|1|2.247396|1|0.436749|
|1930902|2|6.521169|1|10.586859|
|1635282|2|4.263889|1|3.909927|
|990634|2|2.811472|2|0.564086|
|693071|2|1.080444|0|—|
|42576|2|1.633270|1|4.971279|

Across all 16 proposals including failures, mean J is 6.101727 raw, 5.085836 repaired, and 6.669380 saved classical. This is not a success-conditioned mean.

The remaining failure is task 1412961 slot 1: world margins -17.839 mm (MPD) and -15.806 mm (cuRobo), position error 2.179 mm, orientation 2.231 deg. Self/joint/velocity checks pass. Starting world penetration was 169.978 mm: substantial repair remains insufficient. J increases from 11.738135 to 14.344519. The fixed-budget best soft-penalty compromise is not a hard-feasibility solution; no continuation or replacement was attempted.

Nominally valid paths can still have narrow safety margins: task 1930902 slot 0 cuRobo world clearance 1.073 mm; task 42576 both joint margins 2.528 mrad. No stochastic evaluation of repaired paths was run, and historical audits did not select or tune them.

## Timing and resources

Nominal refinement sums to 14.090 s, mean 1.761 s per two-proposal task; shared initialization 0.133 s. Run internal wall 17.563 s; complete command meter 18.52 s wall, 17.44 user, 1.06 system. Validation and preservation are included. Saved-seed diagnosis incurs no fresh generation; historical classical IK/nominal proposal totals were 2.83–2.98 s/task, nominal slot optimization about .86 s. These are historical observations, not a fresh matched timing trial.

Peak GPU allocation 56.6 MiB. GPU stage times are synchronized wall times, not kernel occupancy. Ledger adds a conservative 60 s unmetered preparation/inspection/edit/review/commit allowance, booking 78.52 / 180 s, not a measured total. No installs/downloads/training/hardware changes.

## One fair follow-on design (not run)

Keep all eight tasks and two proposals/family. Generate fresh classical full-pose DLS seeds and fresh MPD DDIM controls, charging both generation paths including IK/model initialization. Both receive the same 80-step nominal preparation with the same free-terminal-joint pose goal; no cached reference or projection. Report nominal failures before risk correction and retain all failed slots.

Within each family compare eight deterministic dense maximum-loss correction steps against eight physical CVaR correction steps, using identical 5 ms constraints, normalization, effort cost, optimizer, gradient cap and penalty. Match the 32-scenario gradient work by explicitly evaluating 32 zero-error scenarios in the deterministic branch through the same implementation, versus 32 physical scenarios for CVaR. Report actual timings; equal scenario count is not an equal-time assertion. This controls extra iterations and denser checks.

Use one prespecified 5% budget and fixed independent optimizer/admission/audit seeds with candidate-batch confidence correction. Profile and freeze sample counts and caps before outcomes. Audit never selects or repairs; charge fresh proposals, IK, initialization, nominal preparation, correction, admission and verification. This four-arm design has not run and needs separate authorization.
