# Saved-outcome MPD diagnosis

This diagnosis reads existing outcomes and prior qualification evidence only. No new proposals, physical draws, FK experiments, optimization, training or audit-driven admission changes were made. FAILURE_DIAGNOSIS.csv contains all 128 proposals and both budgets; nominal checks are the already saved NOMINAL_DIAGNOSTICS.json.

## Main finding

All 16 raw MPD proposals are nominally invalid: position fails in 16/16, orientation in 15/16, and each world model in 8/16. No raw nominal self-collision, joint-limit or velocity failure occurs. Both CVaR branches start from these same invalid proposals; all 16 corrected proposals remain nominally invalid at each budget. Position and orientation each fail in 14/16; MPD-world fails in 8/16, cuRobo-world in 10/16. Thus 0/8 admission primarily measures missing nominal proposal repair, not stochastic failure of otherwise healthy MPD plans.

## Per-task raw proposal failures

W=world collision (both models), P=terminal position, R=terminal orientation. Each row lists both fixed proposal slots; raw proposals are identical across budgets. Self/joint/command/executed-velocity nominal events are absent in every row.

|Task / official ID|Slot 0 nominal failures|Slot 1 nominal failures|Slot 0 position / angle|Slot 1 position / angle|
|---|---|---|---|---|
|0 / 1959097|W,P,R|W,P,R|10.49 mm / 3.19 deg|25.77 mm / 4.73 deg|
|1 / 572764|P,R|P,R|69.54 mm / 7.91 deg|17.84 mm / 10.51 deg|
|2 / 1412961|W,P,R|W,P|28.72 mm / 12.80 deg|20.96 mm / 1.93 deg|
|3 / 1930902|W,P,R|P,R|25.60 mm / 14.68 deg|30.31 mm / 5.35 deg|
|4 / 1635282|P,R|W,P,R|22.07 mm / 5.22 deg|87.52 mm / 11.42 deg|
|5 / 990634|P,R|W,P,R|49.87 mm / 6.52 deg|67.54 mm / 6.42 deg|
|6 / 693071|P,R|W,P,R|59.04 mm / 16.52 deg|22.52 mm / 6.68 deg|
|7 / 42576|P,R|P,R|47.96 mm / 7.32 deg|29.97 mm / 7.25 deg|

Raw position errors range 10.49–87.52 mm, orientation 1.93–16.52 deg, against 10 mm / 2 deg tolerances. Task 2 slot 1 has world penetration 169.98 mm; task 0 slot 0 has cuRobo world margin -22.81 mm. Task 7 has positive nominal joint margin only 0.002528 rad; its two raw audits contain 562 and 570 joint failures / 2,048. That is genuine added execution risk, but both already fail position/orientation nominally. No MPD audit has a self-collision or command/executed-velocity failure. Audit event counts overlap; they cannot be added into union counts.

CVaR reduces some endpoint errors but never produces a fully valid path in eight steps. At 5%, corrected position errors remain 8.66–69.28 mm, angles 1.01–12.14 deg. World failures remain and cuRobo-world nominal failure count increases from eight to ten. This is insufficient global nominal repair from these initializers, not evidence of model misloading, successful nominal risk guidance, or all diffusion planners failing.

## Representation, conditioning and historical comparison

The same frozen strict EMA checkpoint is used: SHA256 1b367f3e34cc324fead97fb9f764eadb44c452eff240c34e6cee5f9e57b470d5. Exact saved SafeLimits normalizers and the unchanged adapters/native_runtime.py are reused. Conditioning uses qs plus EE position and flattened rotation matrix through the native ContextModelCombined, with normalized qs/position and unchanged rotation. TASKS.pt selects all context fields together from the official dataset. Goal validation uses that same ee_goal_pose in the panda_link0 to panda_hand frame. No fixed terminal joint goal is imposed.

Both current and successful historical inference use unguided DDIM 15, eta 0.5, desired 24 learned seven-joint controls. The degree-5 spline has 29 actual controls: tripled fixed start, 23 learned middle controls, tripled learned endpoint. The source-equivalence qualification reconstruction error is 8.88e-16; geometry agreement is 1.11e-16. Current duration is 10 s with 201 nominal states and commands diff(q)/0.05; historical native conversion also used 10 s, 128 states, and physical dt 10/127. Training metadata duration was 5 s; both inference studies stretch position curves to 10 s. Phase7 uses physically finite-differenced zero-order-held commands, not the native PhaseTimeLinear derivative scale. This grid/execution and derivative change is explicit; acceleration/jerk are not hard phase7 events. It does not explain nominal endpoint position errors.

The historical valid-task screen also had raw feasibility 0/128 in Base, but applied 75-step native cuRobo pose/collision-aware refinement and obtained 106/128 feasible slots, full 16/16 Base case coverage. Its source is scripts/screen_valid_task_pool.py and reports/valid-task-report.md in mpd-seeded-qualification-20261001. Phase7 ordinary MPD is raw sampling (guide=None) with NO nominal refinement. MPD+CVaR adds only eight steps per slot. Thus the phase7 label ordinary MPD denotes the unguided prior, not the historically qualified full refined MPD pipeline. Historical success is on different tasks and a different downstream solver; it is not a free success baseline for this cohort.

Qualification proves checkpoint loading, conversion, FK/event consistency and finite gradients. It did not require nominal MPD health on the eight new tasks before the comparison. There is no verified context/frame/control-count defect in the saved evidence, but strict loading does not establish calibration or generator accuracy. Only two draws per request were used versus eight in the historical screen.

## Effort and attribution confounds

Classical receives two fresh full-pose 80-step DLS IK seeds plus 80 nominal collision/pose refinement steps per slot (160 trajectory gradients/request). MPD receives two DDIM proposals and no explicit IK or nominal optimization. Both CVaR arms add 16 gradients/request over 32 physical draws (512 scenario-gradient evaluations). The shared 15 s cap is a ceiling, not equal effort: raw MPD spends about 0.05–0.06 s generating; classical about 2.8–2.9 s proposing/refining. Requests additionally spend about 2.55 s admitting both candidates.

Other confounds: nominal classical penalties use 50 ms average squared violations while CVaR uses 5 ms maximum violations; final risk correction adds both compute and denser checks; no same-effort deterministic final correction exists; proposal counts differ from historical tests; historical native optimizer/grid/motion constraints differ; the cohort is one nominal-clearance-ranked Base task stratum; ideal observations and synthetic doubled noise do not validate hardware execution. Accepted-set mean costs cover different subsets; only paired common tasks support cost comparisons.

## Selection and independent auditing

All 128 saved admission decisions and all 64 lowest-cost selections were checked against the recorded rule. Acceptance requires finite values, valid nominal command limits and exact one-sided Clopper–Pearson upper bound beta.ppf(.975,k+1,n-k) <= delta, not the empirical point estimate. Alpha .05/2 corrects the fixed two-candidate batch; it does not provide simultaneous study-wide confidence.

Optimization seed 2026104100+task (32 draws), admission seed 2026104200+task (1,024), audit seed 2026104300+task (4,096 stream) are separate. Common draws intentionally couple arms and budgets. Unselected audit uses the 2,048 prefix; selected uses 4,096, and all-proposal means use 2,048 for everyone. Selected/unselected audit streams are paired, not mutually independent; both are independent of optimization/admission. Exact task/path-hash/sample-count audit reuse is explicit (95 distinct cache entries); every admission remains freshly evaluated. Audit outcomes never change admission or plans.

The protocol/source freeze commit fcd69580384a72d627940dda8325b9cfce8dcd4d precedes outcomes. Profiles measured 4,096 total validation about 5.36 s and 8,192 about 10.73 s; those counts across 128 proposals exceeded the 1,200 s cap before planning. Frozen counts were 1,024 admission / 2,048 unselected audit / 4,096 selected audit, with eight risk steps, two candidates, 15 s proposal/correction ceiling and 1,080 s main guard. No cap-driven incomplete request occurred.

## One concrete next repair experiment (not run)

On the same eight tasks, same saved MPD controls and unchanged physics/geometry/controller, apply exactly the classical arm's 80-step nominal trajectory refinement to each of the two MPD initializers, with the same nominal objective and learning rate. First report zero-noise full-grid feasibility and endpoint errors; stop before stochastic guidance if nominal health is still zero. This single initializer-repair ablation tests whether the absent nominal repair explains the phase7 failure, without retraining, new cohort, extra draws or denoising guidance. Saved controls may be reused for this diagnostic, but any future end-to-end performance claim must charge fresh proposal generation. Existing audit samples must not select or tune the repair.
