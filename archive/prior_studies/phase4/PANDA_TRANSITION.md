# Transition design: stochastic full-arm Panda planning

This is a design only. No manipulator code, assets, checkpoint loads, simulation, GPU work, or hardware action was performed. Existing asset-reference and qualification documents were inspected read-only for provenance; their deterministic results are not stochastic qualifications.

## Engineering goal

The next capability experiment should establish whether a diffusion proposal plus risk-aware correction produces better **validated full-arm cost/risk/acceptance** than strong conventional planning under explicitly modeled execution uncertainty. It should not be another 2D closure variant or a deterministic Panda demonstration labeled risk-aware.

Start with one static, tabletop EnvSpheres3D scene and a seven-joint Panda, fixed base and gripper configuration, a known start and a joint/pose goal, and two route choices with different clearance/cost. Keep the map fixed for the first controlled experiment. Specify robot base transform, collision geometry, allowed self-collision pairs, joint/velocity limits, target transform, and control timing in a hashed task manifest. No external assets or new model training are needed for the initial qualification if existing local assets meet these gates.

## Physical stochastic model, controls, and endpoint

Use an explicitly declared position-servo/velocity-command abstraction:

q[k+1] = q[k] + dt*(u[k] + b + eta[k]), with q[0] = q_start + e0.

Here q and u have seven components; dt=.05 s and H=40 give a two-second open-loop command. e0 is initial joint-state estimation/calibration error; b is a constant per-execution velocity bias; eta[k] is independently sampled velocity tracking/process error. Their distributions and independence assumptions are part of the model, not inferred from diffusion spread. One initial **synthetic** regime can use e0 ~ N(0,(.002 rad)^2 I), b ~ N(0,(.001 rad/s)^2 I), eta[k] ~ N(0,(.005 rad/s)^2 I). These values are experimental assumptions, not measured Panda accuracy. A second physical-noise regime must be prespecified before final scoring; correlated errors can be added later without silently changing the meaning of risk.

Augment the state as z=[q,b]. Dynamics are affine, so joint-state Gaussian moments and cross-time covariance are exact under these assumptions. For independent e0,b,eta:

Cov(q[k],q[l]) = Sigma_e0 + k*l*dt^2*Sigma_b + min(k,l)*dt^2*Sigma_eta.

This supplies a useful physically meaningful correlated trajectory model without relying on the 2D sine closure. Nonlinear forward kinematics and distances still make collision-event probability approximate under analytic distance linearization. State-Gaussian exactness is not a full-arm chance certificate.

Commands obey actual Panda joint velocity limits from the qualified local configuration; executed joint positions obey the same joint limits. Do not invent numeric hardware limits or simulate clipping that hides a violation. A future acceleration-limit requirement should use explicit command differences/dt; a second-order actuator model is a separate benchmark change, not an assumed feature of this first-order model.

Nominal proposal conversion must preserve controls-to-states consistency: sample the existing joint B-spline at fixed physical times, derive u[k]=(q_nom[k+1]-q_nom[k])/dt, and roll those controls through the specified model. The validator evaluates these executed controls, not the unperturbed B-spline. Never repair stochastic execution with noiseless endpoint inpainting or re-anchor the prescribed target to the achieved endpoint.

Terminal requirements for the default study: prescribed end-effector position within .01 m, full orientation geodesic error within 2 degrees, and hard joint limits. Redundant terminal joints remain free, consistently with the checkpoint's pose-conditioned semantics. Freeze the exact prescribed target before outcomes; do not re-anchor it to the achieved endpoint. An optional joint-target subtest with .02 rad component tolerances would be a separately labeled benchmark applied equally to all methods, not a hidden extra constraint in the default study.

## Full-arm trajectory-risk event

Failure is the union of:

1. Any modeled robot link/gripper sphere or capsule colliding with any environment obstacle/table, including the final state.
2. Any disallowed self-collision pair along the executed motion.
3. Any executed joint-limit violation (and any added velocity/acceleration limits under declared semantics).
4. Terminal position/orientation failure; optional separately declared terminal joint failure.

Retain the complete native world sphere/link and self-pair sets. An end-effector-only distance or a few hand-picked links is unacceptable. The sphere/capsule geometry is an explicit conservative approximation only if coverage is qualified against the model; native collision spheres alone do not prove that every physical mesh surface is covered. Record exclusions/allowed contact pairs and geometry limitations.

For the first qualification, define events on a frozen dense time grid (e.g. .005 s, including both endpoints). Each execution's q path between control updates follows the stated constant velocity noise/bias model. Evaluate the same path on a .0025 s grid as a discretization diagnostic and retain differences. Do not label agreement on two grids a continuous-time proof. A continuous-time claim would require qualified swept-volume or interval/Lipschitz clearance bounds with verified link-motion limits; that is a later gate.

## Risk energy and dimension/time scaling

An analytic baseline linearizes signed full-arm distances around the nominal/mean joint configuration using every link's Jacobian and the **full joint covariance**. Its circle/sphere/plane tail energies must be labeled approximate because FK and distance are nonlinear. Include terminal pose uncertainty and joint-limit events, not only obstacle collision. Event-wise risk sums can be very conservative as horizon and active pair count grow; report event counts, marginal sums, overlaps, and actual joint failure rather than silently divide by the number of events or clamp the sum into a probability.

A sampled CVaR alternative uses common reparameterized physical noise paths and the maximum signed trajectory violation over every event. Normalize metres and radians by fixed physical clearance/pose tolerances before combining losses; the sign and zero threshold must preserve the exact declared failure event. State normalization choices because they change CVaR's tail severity objective even when the failure event is unchanged. Use the exact fractional empirical tail with K=256 initially, Delta=.05/.10, independent validation and audit. Finite sampled CVaR is not a chance certificate.

For d=7 (augmented d=14), H=40, L robot spheres, O environment primitives, and S self-pairs, direct collision work is roughly O(K*H*(L*O+S)) after FK; backpropagate through all controls and augmented dynamics. A naive analytic approach scales with H, event count, FK Jacobians, and covariance propagation. Exploit exact affine covariance and structured cross-time terms; avoid constructing a huge dense trajectory covariance merely to compute marginal distances. Use event-active screening only with conservative geometric reachability bounds, or record it as a heuristic with an all-pair independent validator.

Do not reuse lambda=8 or gamma=30 blindly after changing units/dimension. On a separate tuning fixture measure gradient norms, risk/cost contributions, and saturation before freezing final scales. Keep fixed maximum guide calls, measured total compute, and adaptive-stopping logic separate. Whole-request timing includes checkpoint/model initialization policy, proposal generation, correction, collision geometry evaluation, and independent acceptance.

## Baselines and independent validation

The initial comparison needs:

- Deterministic RRTConnect (where available from the existing sources) or collision-aware waypoint optimization, followed by the same time parameterization, uncertainty model and risk validator.
- Strong deterministic trajectory optimization with full-arm constraints and explicit length/effort/roughness J, with multiple initializations and all failures retained.
- Conventional chance-surrogate and sampled-CVaR trajectory refinement, using identical J, physical uncertainty, time caps and admission criteria.
- Unguided existing diffusion proposals; deterministic-guided/projection diffusion; physical-risk chance/CVaR guidance; final-only risk refinement to isolate denoising-time correction.

Compare at common risk budgets and measured compute; also plot accepted-plan cost against independently audited risk. Do not infer superiority from a single upper budget if one planner consumes more risk. Charge failed solve/initialization attempts and candidate-batch selection. If selecting among M validated candidates, adjust confidence for the fixed selection batch as in the current public planner; audit samples never influence selection.

An independent validator samples e0, b, eta with fresh seeds, recomputes FK/full-arm world/self distances and endpoint pose, and records per-execution hard failure. Reuse the independently structured NumPy/SciPy geometry validator pattern from the local adapters, but qualify its response to physical execution states and all pair sets. Report per-plan collision, self-collision, joint-limit, goal and combined failure with intervals. Start with 4,096 admission and 8,192 audit draws where practical; profile chunked FK before claiming a budget. Record every rejected proposal. Zero observed failures only supplies a finite upper risk estimate.

## Existing local assets and qualification boundaries

Read-only metadata inspected in this turn:

- `/home/kristoff/Documents/experiments/mpd-followthrough-mechanism-20261002/asset-references.json`
- Its `QUALIFICATION_REPORT.md`.
- `/home/kristoff/Documents/experiments/mpd-seeded-qualification-20261001/VALID_TASK_REPRODUCE.md`.

Those documents identify the genuine Panda joint-space MPD B-spline checkpoint in `assets/data_public/data_trained_models/launch_train_diffusion_models-v04_2024-09-18_08-12-54/.../checkpoints/ema_model_current_state_dict.pth`, with recorded SHA-256 `1b367f3e34cc324fead97fb9f764eadb44c452eff240c34e6cee5f9e57b470d5`, normalization statistics, and public source commit `3676cbf371b0e2a98663d8cef666f671e004e3da`. They also identify:

- `external/mpd-headless` and `external/mpd-splines-public` under the seeded-qualification project.
- Genuine headless Panda FK plus existing cuRobo geometry/configuration.
- Existing joint B-spline basis, normalization, and contextual end-effector goal handling.
- Full world/self geometry and independent dense validation adapters under the follow-through project's `prepared/` directory.
- Historical valid-task/sample banks as provenance and fixture sources, never free proposal-generation/timing credit.

These are candidates for reuse, not new stochastic qualifications. The referenced reports describe deterministic fixtures and another task's protocol; they do not establish physical-noise execution safety. Check the actual source/checkpoint/normalization hashes and licenses at a future authorized asset-qualification step. Preserve original projects and do not invoke their output-producing scripts into old directories. No checkpoint/asset bytes were loaded or copied in this turn.

## Coherent first capability experiment and release gates

The next capability is **physical-noise execution risk over the whole arm**, with exact affine joint uncertainty and nonlinear geometry—not another sine-moment correction.

First, a small CPU-compatible geometry/evaluator qualification, with no diffusion sampling/training: four fixed nominal Panda control paths, two declared physical-noise regimes, full-arm world/self and endpoint events, paired dense-grid checks, independent hard-event results. Include deliberate elbow collision, self-collision, and orientation-only failure fixtures. Profile 256/1,024 executions and chunk memory before fixing a larger validator budget. Freeze geometry and physics before final task outcomes.

Then define a small held-out planning test with two distinct feasible clearance/cost route families, using nominal-only task criteria fixed before stochastic outcomes. Evaluate deterministic, conventional risk-aware and diffusion risk-aware methods with the same controls-to-state conversion. Require at least one case where nominal end-effector safety differs from full-arm execution safety; retain difficult and infeasible cases instead of outcome-based curation.

Release gates: physical-model and endpoint semantics accepted; full-arm geometry/validator qualification; asset provenance/license and checkpoint conditioning validated; resource allocation and compute/storage budget approved; baseline proposal feasibility established before interpreting safety corrections. QSP retains the GPU. No manipulator run or training is authorized by this design alone, and no action was launched.
