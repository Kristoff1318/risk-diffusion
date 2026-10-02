# Old-fixture GPU qualification protocol

User released the GPU for90 additional command seconds/30MB only.
No fresh outcomes, proposals, inference or planner comparison are authorized.
The frozen phase11 specification, coefficients and events remain unchanged.

Backend architecture for this first timing qualification: build the complete
analytic frozen local model with the existing NumPy CPU builder, transfer
its constants to CUDA float64, and use differentiable Torch evaluation.
There is no new GPU geometry approximation or Jacobian convention. Count CPU
construction, transfers and synchronization in the correction wall time;
GPU-only evaluation speed is not substituted for complete correction latency.
If this architecture fails the2s two-candidate eight-step gate, stop it; no
automatic rewrite to a faster builder in this turn.

Before execution, commit this adapter/protocol/qualification runner.
Full-grid parity and event counts on phase5 first path. Active-risk spline
probe holds that same frozen model fixed and displaces final free control0
by .03rad. Central finite differences1e-6; same tolerances as phase11.
Spline basis/initial controls exactly reuse the phase7 quintic24-free-control
representation; reconstruct all four saved phase5 paths, retain fixed starts.
CPU-to-GPU energy, component sums and gradients must agree within absolute
1e-6 plus relative1e-8; CPU/GPU tiny-tail differences may underflow and are
not called true zeros. No zero-variance smoothing or event-count division.

Evaluate full-grid unoptimized surrogate on all FOUR established saved paths.
Compare its separate component sums against the already independently measured
phase6 RUN_256_10.json DOUBLE-regime risks/component counts, with exact CIs.
These old sampled measurements use the unchanged paths/model and no surrogate
selection. Reuse their measurements explicitly; no new risk samples needed.
Box sums versus norm-based terminal failure rates are not identical events.

One bounded timing request uses saved old paths0 and1 as its two slots solely
to measure eight-step correction each, never claim a task comparison. Each
slot keeps its OWN start and goal. Adam8 steps,lr.003,cap20, frozen model rebuilt
each iteration. Report all iterations, R/J/components, synchronized actual
wall including initialization/build/transfer/gradients/selection. Full eight
steps run for timing if the overall60s process guard permits; the2s target
is assessed afterward, not presented as an enforced deadline. No fresh plans.
Save tiny corrected-path/control arrays for reproducibility, no model tensors.
Singular local models or nonfinite values stop immediately.

Main timeout65s, internal guard60s; allowance25s for inspection, code/spec,
reporting, preservation and commits. All command work including old-fixture
optimization is credited against450s total; prior CPU phase booked20s.
If full grid does not fit, the timing fails, or all four healthy old fixtures
are unusable, report that terminal gate failure and release GPU explicitly.
