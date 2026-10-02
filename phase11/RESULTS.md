# CPU preparatory qualification — complete within its narrow scope

Prototype/specification committed before outcomes at33c740d. First attempt
preserved atf98c607: reversed-time covariance test incorrectly used F^-1.
The harness now uses the earlier-time covariance and F^abs(k-l); production
covariance equations and all frozen coefficients/event choices were unchanged.
An active-penalty derivative probe was added because the original safe anchor
has an inactive penalty. This is a finite-difference probe, not an optimizer
or a new path-selection pipeline. Failed evidence remains alongside the pass.

## Implemented

`covariance_energy.py` implements exact feedback cross covariance, dense-grid
variance, command/executed-velocity variance, native CPU FK/point Jacobians,
all individual world/self clearance expansions, the target-frame terminal
pose boxes, all joint/velocity marginals, full event counts, E/R/J and analytic
gradients with respect to201 nominal joint states. Geometry/pose Jacobians
and variances are fixed inside each FrozenEnergy object. Differentiating a
rebuilt objective is deliberately not claimed. See SPECIFICATION.md for
penalty500, budget normalization, edge rules and complete event definitions.

The energy evaluator is implemented. The eight-step Adam correction runner,
spline-control gradient mapping, GPU backend and resource-stop integration
are NOT implemented/qualified in this CPU phase. No correction was executed.
The existing dense correction remains preserved in phase7/phase9.

## Verification

Nine tests pass. The bounded native fixture is only slot0 of the preserved
phase5 bank, with8 predeclared dense state indices, every geometry pair at
those states, all200 velocity steps, and terminal boxes. No new task was opened.
The pose-Jacobian test also applies a disclosed synthetic target rotation to
exercise the nonzero log derivative; it does not assess task performance.

* Augmented covariance maximum absolute error:1.27e-21.
* Clearance derivative maximum absolute error:2.38e-10.
* Pose-box derivative maximum absolute error:1.89e-7, below frozen1e-6 tolerance.
* Native FK and hard geometry margins match.
* Safe-anchor frozen-energy directional derivative error:4.12e-11.
* Active-penalty probe derivative:-7,484,831.230 finite difference versus
  -7,484,831.263 analytic; relative discrepancy4.45e-9.
* Deterministic sigma0/contact semantics, coincident-center rejection,
  PSD rejection/roundoff tolerance, scalar Gaussian tails, dense covariance
  and cost derivatives pass. No covariance roundoff clipping in old fixture.

The bounded subset has22,876 events and R=5.24694e-9, entirely from terminal
boxes at machine precision. Printed zero geometry tails can reflect numeric
underflow and are not exact zero true risk. Production has4,323,770 elementary
events:2158 per state across2001 states,5600 velocity events and12 terminal
box events. No event normalization or pair pruning was applied.
The subset's low sum does NOT establish full-grid acceptance/usability,
nonlinear calibration, cost benefit or risk-repair headroom. No prototype
qualification failure from union conservatism has been established yet.

Two timed CPU test commands took0.40s each including interpreter startup;
successful suite internal time0.098s. Bounded model construction0.00564s is
an8-state observation, not a full-grid/GPU timing or a prediction of2s
two-candidate correction. Peak RSS successful command85,240KiB.

## Exact statements versus approximations

The affine joint tracking covariance and joint/command/executed-velocity
marginals are exact for the fixed independent Gaussian model, including
the persistent-bias covariance term. Initial joints remain immutable in
the intended spline parametrization. Summation bounds the union of the
specified affine events without requiring event independence.

Clearances, position and rotation log use local affine approximations;
Gaussian tails and boxes are only local-model surrogates. They provide no
bound on actual nonlinear whole-arm trajectory risk without remainder
control. Sequential frozen-Jacobian updates are not the gradient of a fully
differentiated nonlinear moment objective. Independent hard simulation
admission/audits remain required. The accepted analytic initial-risk floor
is preserved and unaffected by this prototype.

## Remaining work after GPU release, before fresh outcomes

1. Freeze/commit the GPU backend and spline-gradient mapping with the SAME
   objective, all pairs/grid and edge conventions; cross-check values,
   derivatives and event counts against this CPU model. Singular events
   must stop optimization, not proceed because a probability has zero slope.
2. Implement the sequential eight-step Adam runner, lr.003/cap20, best-iterate
   rule, and common2s two-candidate ceiling. Verify initial state stays fixed.
   Count rebuilding plus gradients/selection in correction latency. One slow
   step may exceed a deadline: report actual overrun, never hide it as2s.
3. Full2001-state old-fixture evaluation and synchronized profiling, including
   both proposals and memory, followed by old healthy-fixture usability.
   A bound that fails on every healthy fixture or fails the runtime gate stops
   this prototype; no changed coefficients, pairs or hidden allocation redesign.
4. Full nonlinear validator parity, terminal-box conservatism/error disclosure,
   and any independent old-fixture surrogate calibration check required by
   phase10. Its sampling/GPU work has not run and remains separately metered.
5. Reconcile the450s total ledger and frozen validation sample counts BEFORE
   opening any fresh outcomes. No fresh execution is authorized by this report.

The unchanged four-task/four-arm phase10 plan remains logically applicable,
but its feasibility is UNCONFIRMED. The CPU results do not justify expanding
its cohort, dropping events or promising that the2s/450s gates will pass.
Twenty conservative booked preparatory seconds are credited toward450,
leaving430s; these include0.80s timed tests and estimated ancillary commands,
edits, inspection/preservation/reporting/commits, rather than pretending all
of that overhead was measured. No new assets, GPU, inference, training,
hardware, optimizer, sampling or fresh geometry/proposals were used.

All edits are additive inphase11. Original/default code and phase10 documents
are unchanged. The git baseline/tag and working branch are retained.
