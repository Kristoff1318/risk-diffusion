# P0: stochastic full-arm execution qualification

Implemented and qualified a seven-joint kinematic tracking-error risk evaluator. The four fixed native reference paths remain 10 s; their 200 velocity commands preserve the nominal states. This is an uncalibrated synthetic open-loop model, not rigid-body dynamics, torque/contact simulation, measured robot accuracy, or a planner performance comparison.

## Frozen model and assets

q[k+1]=q[k]+.05*(u[k]+b+eta[k]); q0=qstart+e0. H=200. Base independent Gaussian joint scales: initial 2 mrad, persistent per-execution velocity bias 1 mrad/s, iid per 50 ms velocity error 5 mrad/s; doubled regime 4/2/10. No feedback, clipping, endpoint repair or target replacement. Noise is fixed within each command interval and reused at 5 ms/2.5 ms refinements.

Exact Cov(q[k],q[l]) = Sigmae0+k*l*dt²*Sigmab+min(k,l)*dt²*Sigmaeta. Terminal standard deviation is 10.7935 mrad base, 21.5870 mrad doubled. The bias contributes 85.8% of terminal variance. Joint covariance is exact; nonlinear FK event probabilities are evaluated by sampled hard events.

Prescribed cached panda_hand goal: position tolerance 1 cm and full SO(3) geodesic tolerance 2 deg; redundant final joints are unconstrained except native limits. Both commanded AND noisy executed velocities are checked against all seven native limits. Acceleration is not an event in this explicitly first-order model.

Read-only original URDF/config/dataset/license hashes are in PROTOCOL.json. First four existing selected EnvSpheres3D indices:575954,15095,1800199,778059. No inference, checkpoint load, training, asset download, installer, or sibling-project mutation. Models: 56 MPD world spheres/280 configured self pairs and 61 active cuRobo spheres/694 configured self pairs; cuRobo world buffer .004 m and native self offsets/ignored link pairs retained. Inactive attached-object placeholders excluded. Fixed gripper fingers are ±.04 m. The genuine scene has 10 obstacle spheres radius .15 m and **no table**; no unmodeled table clearance is claimed. Native sphere sets/allowed-pair masks do not prove physical mesh coverage.

## Qualification evidence

- All four zero-noise control rollouts reconstruct sampled nominal states: max error2.66e-15rad; all hard events pass nominally.
- Independent augmented 14-state covariance recursion/cross-time transition checks versus closed form: max error2.71e-20. Persistent bias-only trajectory check passes; empirical 20,000-draw variance/cross-time check passes.
- Both URDF FK agree exactly at saved endpoints; prescribed cached target max matrix/position error2.45e-07. Independent historical dense margin replay differs by <.001 m/rad across the differing grids.
- Same-path coarse/fine joint states at common timestamps match exactly: max error0.0. Every configured world/self pair is evaluated with bounded state/pair blocks.
- Constructed elbow obstacle gives world clearance-0.060m while EE clearance remains0.500m; isolated self fixture has clear world/joint limits and cuRobo self clearance-0.006334m; 10 deg orientation-only target fixture fails orientation while all other events pass.
- Unclipped joint-limit fixture rejects. Noisy executed overspeed fixture passes command velocity but fails executed velocity. Both URDF hard limits equal the config values.
- Initial preparation failed on manifest path spelling, and first FK qualification exposed the cuRobo URDF base ancestor. Both were repaired and committed before stochastic outcomes; failed attempt logs and partial bank retained. No noise, limits, duration or target tolerance was tuned.

## Primary outcomes: 256 executions per path/regime

Exact two-sided 95% Clopper–Pearson intervals below are pointwise model-based intervals, not simultaneous certificates or admission decisions. EVENT_COUNTS.csv and RUN_n256_sub10.json retain separate model collisions, all other event counts, marginal sums, minima, overlaps via saved flags and runtimes. Samples64/256 are shared prefixes, not independent repeated trials.

|Path|Regime|World MPD/cuRobo|Position/orientation|Union/n|Risk95% interval|
|---|---|---:|---:|---:|---|
|0|base|0/0|96/58|106/256|35.3–47.7%|
|0|double|0/0|210/200|233/256|86.8–94.2%|
|1|base|0/0|147/46|152/256|53.1–65.4%|
|1|double|19/15|228/195|244/256|92.0–97.6%|
|2|base|0/1|135/52|147/256|51.1–63.6%|
|2|double|13/24|221/199|241/256|90.5–96.7%|
|3|base|0/0|80/50|91/256|29.7–41.7%|
|3|double|0/0|192/189|222/256|81.9–90.6%|

No self-collision, joint-limit or command/execution-velocity failures occurred in these 256 samples, although the constructed fixtures reject those categories. Their pointwise two-sided 95% upper bound with 0/256 is 1.43%; that is not zero risk. World collisions overlap with terminal pose failures; add neither model collision counts nor position/orientation counts to obtain union risk.

The paired 2.5 ms diagnostic uses all 64 prefix executions in all 8 cases. Classification changes: 0. Grid agreement is a discrete diagnostic, not proof against between-grid collision. No new random draws were used.

The predetermined first path/base 1,024 CPU scalability probe observed 452/1024 combined failures (44.14%,95%CI 41.07–47.24%). Its first 256 event flags equal the primary run exactly. This single-case larger sample is a throughput probe, not selective bank scoring.

## Compute, deliverables and next step

CPU only: 64 primary all cases 39.57s ; 256 primary 156.74s ; 64 finegrid 78.17s ; 1,024 firstpath/base 77.06s. Chunk 256/512/1,024 flags identical; runtimes 5.14s, 4.86s, 4.70s. Frozen production chunk 512 retained. Main 256 peakRSS 232.0MiB ; 1,024 340.2MiB. Dataset freeze briefly needed4.4GiB but wrote only the derived four-path bank. GPU unavailable; no restart attempted. Ledger covers preparation failures and all measured commands;30s reserve covers read/edit/commit overhead.

Runnable code: panda_p0.py, extra_checks.py, profile_chunks.py, profile_scale.py. FROZEN_PATHS.npz contains nominal 201-state trajectories, 200 velocity commands, prescribed goal poses, 29 B-spline controls and official indices. DEMO_n256_sub10.npz saves the first four physical executions/velocities for each case. FLAGS files retain all categorical per-execution failures. DEMO.png visualizes saved trajectories/error/risk; obstacle points and joint-frame sketches are illustrative, not collision meshes. README.md gives isolated replay instructions; existing outcome files are guarded against overwrite.

Next: review a physically meaningful feedback/controller model or explicitly preserve open-loop infeasibility before planner comparison. At this 10 s duration, fixed bias dominates terminal error and all four frozen proposals substantially exceed 5/10% union risk. Do not reduce noise/tolerances or change targets based on these results. Planning cannot claim safety from collision reduction while omitting the dominant goal event. Keep these retained failure cases as qualification evidence; define future heldout route choices by nominal task properties and freeze any changed controller before outcomes.

Measured CPU cost is roughly .075 s/execution at 5 ms. A realistic equal planning request budget is about 400 s: at most 60–80 s proposal/refinement plus ~310 s independent 4,096-draw admission. A separate 8,192-draw audit would cost ~620 s/plan on this CPU and needs a separately approved research budget. These are extrapolations, not tested large-validator timings. Inner K=256 geometry alone costs ~19 s per full batch, so many gradient iterations require working GPU FK or a separately qualified acceleration strategy; no large risk-optimization sweep is justified yet. Use one-sided exact 95% admission upper bounds with selection correction for multiple candidates; the current pointwise P0 intervals are not that admission pipeline.
