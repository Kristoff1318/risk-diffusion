# Native refactor terminal gate PASS — old fixtures only

GPU slot explicitly released after completion; no further GPU calls or fresh
study follow this result. Numeric-first runner committed atc225288 before
outcomes. Code/spec at72dbd4b, CPU evidence1251126; all prior implementations
and failures remain unchanged. No energy/event/gradient/optimizer redesign.

## Equivalence before timing

Full2001-state checks on all FOUR saved phase5 fixtures compare every native
FK transform, sphere point/Jacobian, world/self pair clearance/Jacobian/SD/
mask, terminal box model, counts and component sums against retained NumPy
and hybrid CUDA reference. Total4,323,770 events each, including configured
self duplicates. Max all-entry block discrepancies across fixtures:
8.60e-16,1.69e-15,1.05e-15,9.44e-16. Numeric zero tails do not imply true zeros.

Contact/sigma0, coincident/near-coincident center primitives and zero/nearpi
terminal branches pass. CPU tests cover nonfinite/PSD rejection and fresh
anchor/cache identity. Geometry tensors are detached; only invariant topology/
pair definitions/covariance schedules are cached. Every iterate rebuilds its
anchor geometry and terminal pose Jacobians. Float64 throughout.

Active frozen spline finite-difference relative discrepancy3.44e-10;
initial-state free-control derivative0. The SAME frozen model is held fixed
for that test. No full derivative through rebuilt Jacobians is claimed.

Full8-step optimizer sequences on two old fixtures compare independently
advanced native and retained NumPy-reference gradients, clipped gradients,
Adam moments/updates and best selection. Maximum control update discrepancy
1.67e-14/1.02e-14. Both select completed iterate6, matching reference.
Each fixture retains its own start/goal. All checks pass BEFORE timing ran.

Unchanged-anchor surrogate component sums reproduce the previous GPU report;
each remains within5%. Existing independent hard audits are0/256 per fixture,
exact95% union interval[0,1.430636%], insufficient to calibrate tiny predicted
probabilities. Those prior measurements are reused, no new risk sampling.

## Measured fixed eight-step two-slot runtime

Whole synchronized request: **.264526s**, including newly constructed invariant
cache(.005827s), both independent Adam setups, all16 anchor rebuilds/gradients/
updates/selection checks, and selected-path extraction. Target2s: **PASS**.

|Measured stage across16 steps|Total seconds|Mean per step seconds|
|---|---:|---:|
|Native anchor construction, including terminal CPU/SciPy transfer|.197219|.012326|
|Frozen evaluation|.028649|.001791|
|Backward|.026027|.001627|
|Finite checks, selection, clipping and Adam|.004389|.000274|

Every stage synchronization is included in whole latency. Initial context
construction before qualification was.037397s; timed new context was.005827s
after CUDA/model kernels were warmed by numeric verification. CUDA startup,
URDF/asset initialization and spline loading are outside this correction
request and must be charged separately in any future cold full pipeline.
Optional detailed component scalar materialization occurs after request timing;
all finite checks and selection scalars remain INSIDE timing. This avoids
the old per-component logging synchronization without altering optimizer
decisions. CPU terminal branch behavior stays unchanged; bulk geometry tensors
never round-trip through NumPy. No forced per-step garbage collection.

Comparison with13.352060s prior hybrid is an engineering observation on the
same old timing slots/steps; the changed builder/caching/transfers/logging and
collection architecture explains the runtime change. It is NOT a fresh task
comparison or evidence of better risk/cost behavior versus dense/CVaR methods.
The retained original failure is a failure of its hybrid implementation.

Selected iterate diagnostics reproduce previous results: slot0 J3.9593113,
R.0032216669; slot1 J14.6393271,R.0097298816, dominated by terminal boxes.
These corrected paths have NO independent admission/audit. Saved native
paths are reproducibility evidence, not a risk-constrained performance claim.

Peak allocated GPU1,424,957,952bytes and RSS2,327,264KiB are for the ENTIRE
qualification, including simultaneous reference/native/parity temporaries;
isolated timing-stage peak was not reset/measured. Runner internal20.236573s,
full timed host command21.22s. No inference/new tasks/new sampling/training.

## Decision and resource reconciliation

Device-native numeric and old-fixture runtime gates PASS. This resolves the
implementation timing blocker; it does not authorize fresh outcomes. The
fresh plan remains stopped pending explicit authorization and updated total
resource forecast/admission-audit qualification. Do not change samples or
cohort posthoc to make the remaining cap fit.

This GPU phase books47s conservatively:21.22s timed command plus25.78s
estimated ancillary protocol/code/inspection/report/preservation/commit work,
within its90s cap. Including CPU implementation preparation20s and prior74s,
total booking141s against450s, remaining309s. Both ancillary allowances are
estimates, not measured command totals. Full phase11 namespace is under300KB,
well below separate30MB caps. No previous source/evidence is overwritten.

The phase10 old forecast~400s included bounded qualification~60s. Actual
booked preparation/qualification/audit now141s, leaving309s. Future eligibility
depends on recomputing a complete forecast using qualified timings and the
fixed validation counts; this report does not declare the fresh study funded
or feasible merely because correction passed2s. GPU slot is released.
