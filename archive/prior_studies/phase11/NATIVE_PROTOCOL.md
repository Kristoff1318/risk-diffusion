# Device-native refactor: preparation and exact release-gated qualification

Scope this turn: CPU implementation/unit verification only. Adaptive validation
owns GPU. Cap90 command seconds/30MB. No fresh tasks or planner comparison.
Original CPU/hybrid code, frozen specification and prior failures are preserved.

NativeContext caches only native topology/ancestor masks, static sphere/world
geometry and EXACT pair IDs/order/duplicates/buffers, limits,2001 grid index/
interpolation arrays and exact fixed tracking/velocity covariance schedules.
PSD checks occur once per immutable schedule. Isotropic projection is
var(t)*sum(J^2), algebraically the same J C J^T for this fixed model; generic
nonisotropic covariance is rejected, not silently approximated.

Every NativeContext.build runs under torch.no_grad with a fresh detached
anchor. FK transforms, joint axes/origins, centers/point Jacobians, every
pair's clearance/normal/Jacobian/SD/singular mask and terminal pose Jacobians
are rebuilt. Points are vectorized by native link, retaining native sphere
and pair order. No prior-anchor geometry is cached. CPU/GPU reductions need
tolerance equivalence, not falsely claimed bitwise identity.

The existing GPUFrozen.evaluate is inherited unchanged; float64, unnormalized
4,323,770-event sum,penalty500,budget.05, boxes, contact/sigma0/near-coincident
semantics and frozen-surrogate derivative convention remain intact.
The terminal model deliberately reuses existing one-state CPU FK/SciPy log
branches. Only that endpoint7-vector and tiny detached terminal constants
cross the device boundary; this is disclosed and charged in future timing.
No new rotation-log approximation, custom kernel or mixed precision.

CPU unit tests on predeclared8-state old fixtures compare all pair entries,
native FK/point/clearance Jacobians, SDs/masks, event counts, per-component R,
J/E, nominal/spline gradients and active frozen-model finite differences.
Cache identity/fresh geometry tests enforce that only invariants are reused.
Contact/sigma0/singular/nonfinite/PSD tests and native-vs-reference clipped
one-step Adam arithmetic are unit checks, not a planning benchmark.
Production full-grid counts are derived and asserted; no CPU full-grid timing
or GPU availability query is part of these tests.

## GPU protocol — MUST wait for explicit slot release

1. Verify code/spec/fixture hashes and rerun CPU tests; create NativeContext
   on cuda:0 ONLY after release. Warmup/init timing is recorded separately,
   never omitted from a claimed cold pipeline.
2. Numeric gate BEFORE any speed claim: full2001-state old fixtures0–3,
   both native FK/centers/point and pair Jacobians, ALL clearance/mask/SD entries,
   4,323,770-event counts and per-side component sums against preserved CPU
   and hybrid CUDA reference. Assert masks/counts/order exactly; other arrays
   absolute1e-6 +relative1e-8; preserve original Jacobian absolute1e-6 and
   covariance absolute1e-18 checks. Zero/nonfinite/PSD/singular/nearpi tests.
3. Compare R/J/E, nominal and spline gradients on anchor and active-risk
   probes. Central FD1e-6 holds each model FIXED; tolerance absolute1e-4 +
   relative1e-5. Verify all model tensors detached and start derivative0.
4. At IDENTICAL old recorded/reconstructed iteration anchors, compare rebuilt
   models, R/J/E and gradients, capped gradients and per-slot Adam moment/update
   tensors. Then native and reference8-step trajectories and best-completed
   iterate,lr.003/cap20, must agree at declared tolerance. A threshold/tie/
   selected-iterate discrepancy is a numeric gate failure, not ignored.
   Use each timing slot's own fixed start/goal; no new proposals or admission.
5. Only after these gates pass, profile SAME two slots, same8 steps and full
   events. Separate cache creation,covariance setup,anchor construction and
   projection,terminal CPU work/transfer,evaluation,backward,Adam/checks/logging
   and synchronized COMPLETE request. Keep initialization/cold and warm
   timings distinct;2s whole-request target still applies. Memory and event
   accounting accompany all numbers. No probability/event pruning to pass.
6. If parity/runtime fails, preserve evidence and stop the prototype. Fresh
   plan remains stopped unless separately authorized; no automatic fresh run.

This protocol is prepared, NOT executed. There is no GPU speed estimate yet.
