# Closed-loop joint tracking qualification

Ordinary proportional tracking removed observed P0 endpoint and collision failures on this fixed bank. This is baseline/controller qualification, not a planning research advantage. It changes the disclosed execution/observation model; it does not prove P0 safe under unobserved calibration or estimation error.

## Model frozen before outcomes

The same four 10-second nominal paths, 200 commands at 50 ms, prescribed pose targets, both physical-noise regimes and exact same P0 random draws are retained. At each update the ideal encoder observes actual q perfectly. The command is nominal feedforward velocity minus 2/s times joint tracking error, held constant until the next update. Gain follows a declared time constant of ten update periods (0.5 s), not pose-success tuning. q0=qstart+e0; here e0 is **physical initial tracking offset**. No unobservable encoder bias, measurement noise, delay, rigidity, torque/contact dynamics or hardware calibration is modeled. P0 numerically uses the same e0, but its earlier calibration/estimation description cannot be silently equated to this observation assumption.

Saturation policy is explicitly **none** in this affine benchmark: requested velocity equals applied velocity. Neither commands nor executions are clipped or repaired. Over-limit commanded and noisy executed velocities each count as hard failures; an overspeed fixture confirms both reject. Native hard position/velocity limits, all world/self pairs, final pose requirements and 5 ms event grid remain unchanged. No endpoint re-anchoring, inpainting or extra final correction occurs. A real saturating actuator would be a separately disclosed nonlinear benchmark, not covered by these exact Gaussian formulas.

Tracking error obeys e[k+1]=a e[k]+dt(b+eta[k]), a=.9. The seven-joint Gaussian state moments and cross-time covariance are exact for the declared unsaturated affine model:

Cov(e[k],e[l]) = a^(k+l)Sigma_e0 + c[k]c[l]Sigma_b + dt² a^abs(k-l)(1-a^(2min(k,l)))/(1-a²)Sigma_eta, where c[k]=(1-a^k)/K.

The same base scales remain 2 mrad physical initial offset, 1 mrad/s constant per-execution bias, 5 mrad/s iid step tracking noise; doubled scales 4/2/10. Terminal standard deviation becomes 0.760886 mrad base and 1.52177 mrad doubled. Bias still contributes uncertainty; feedback attenuates its steady tracking displacement to b/K.

## Exact checks and execution evidence

Augmented 14-state covariance/crosscov recursion agrees with the closed form to 3.18e-22; 20,000 independent moment samples differ by -0.206% in terminal variance and pass cross-time comparison. Zero-noise tracking reconstructs the nominal bank to 2.78e-17 rad. Gain zero reproduces P0 to 3.11e-15 rad. Both grids sample identical held-velocity paths; 64/256 prefixes agree exactly.

GPU full geometry versus independent NumPy margins on 512 random configurations agrees to 2.78e-16; paired 64 event flags agree. The original elbow and isolated-self fixtures also reject on GPU, retaining all 56/61 world spheres and 280/694 configured self pairs. Overspeed tracking errors reject through the actual complete event validator. The P0 orientation fixture remains valid because terminal pose math/targets/tolerances are unchanged.

All eight cases produced 0/64 and 0/256 failures in every category: both world models, both self models, joints, commanded velocity, executed velocity, position and orientation. Pointwise exact two-sided 95% interval for 0/256 is [0, 1.431%], and one-sided 95% upper is 1.163%. For the eight fixed cases, a Bonferroni one-sided 95% upper is 1.963% per case. These are conditional on the synthetic model and discrete geometry; zero observations are not zero risk.

|Path|Regime|P0 arm/pose/union failures|Tracking arm/pose/union failures|
|---|---|---|---|
|0|base|0/106/106|0/0/0|
|0|double|0/233/233|0/0/0|
|1|base|0/152/152|0/0/0|
|1|double|20/244/244|0/0/0|
|2|base|1/147/147|0/0/0|
|2|double|24/241/241|0/0/0|
|3|base|0/91/91|0/0/0|
|3|double|0/222/222|0/0/0|

Arm means world/self/joint/velocity union; pose means terminal position/orientation union. In P0, collision failures overlapped pose failures. Counts are paired on 256 draws/case; they are not independent trials between controllers. Every failed P0 execution is retained. The 2.5 ms diagnostic changed no classification across 512 paired executions, without redrawing noise. Configured sphere geometry and dense-grid agreement do not prove mesh coverage or continuous-time safety.

Largest terminal position error across the bank is 4.820 mm; orientation 0.596 deg, versus unchanged 10 mm/2 deg tolerances. Minimum commanded velocity margin is 0.817 rad/s and noisy execution margin 0.801 rad/s. No saturation would have been needed for these samples.

## Profiling, deliverables, next study

The approved host route sees RTX4090/CUDA with the same interpreter. The previous sandbox GPU check was masking host access, not proof of driver loss. Nothing was restarted or changed in drivers/security/shared configuration. Pure Torch float64 FK/distances need no new extensions or installation. 64 execution profile: NumPy 4.954 s, GPU 0.111 s; comparison is validator throughput, not planner speed advantage. Full 8-case 256 GPU process takes 4.048 s; per-case evaluation ranges 0.404–0.466 s. Cold imports/GPU initialization are included in process time but excluded from per-case throughput. Main process RSS 860.1 MiB; GPU peak allocated 115.3 MiB. Timing/memory are not gradient/optimizer measurements.

Runnable tracking.py reuses the preserved P0 bank/source/assets read-only. PROTOCOL.json freezes the gain/model/source hashes. CHECKS.json and ADDITIONAL_CHECKS.json contain qualification evidence. Paired/per-event CSVs and FLAGS files retain all outcomes; DEMO_256_10.npz saves joint states, actual velocities and commanded velocities of four executions per case. TRACKING_COMPARISON.png uses only saved evidence. No optimizer, diffusion sampling, training, hardware or asset installation was run.

Next: one small held-out 8-task risk-planning comparison, with this fixed controller/noise/duration/target/geometry, and goal/free-redundancy semantics unchanged. Use existing sphere scenes and fresh start/pose-goal indices/seed that do not overlap this bank. Select by nominal geometry only before any stochastic scoring: collision-free feasible starts/goals and two deterministic route families, one with minimum full-arm clearance 0.5–3 mm and another with ≥10 mm where feasible. These are declared benchmark clearance strata, not noise/obstacle tuning. Keep the first 8 eligible tasks in fixed seed order; if fewer exist under a bounded candidate search, report the shortfall rather than moving obstacles or relaxing strata. Retain infeasible route attempts and all stochastic failures. Use the current relatively roomy four paths as baseline qualification, not as evidence that risk planning is unnecessary.

Compare one deterministic classical route/refinement baseline, one classical physical-risk refinement baseline, and one existing diffusion proposal with identical final risk refinement, all under the same cost and budget. A denoising-guidance variant can be optional within that one study, not another prerequisite chain. Freeze objective, candidate count and task/noise seeds before scoring; no new training. Require full-arm risk and terminal pose in all admissions, and select using fresh samples with multiple-candidate confidence correction.

A reasonable provisional common host-GPU request budget is 60 s including initialization, proposal and correction, with a 15 s allowance for 4,096-draw admission and a separate 25 s/8,192 audit allowance. The measured forward evaluator extrapolates to ~6.5 s for 4,096 draws and ~13 s for 8,192, but gradients, initialization/model inference and larger-draw buffers are unmeasured; these allowances should be checked within the study rather than represented as established performance. Reduce control horizon sampling only with a qualified common validator; keep 5 ms independent admission. Perfect encoder/known map assumptions and sphere/discrete limitations remain explicit.
