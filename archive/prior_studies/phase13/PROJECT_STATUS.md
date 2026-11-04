# Current capabilities and scientific decisions

The project supplies a reproducible synthetic Panda whole-arm validator with
both sphere geometries, self-collision, joint/command/executed-velocity limits,
and terminal1cm/2deg tolerances at5ms checks. It models perfect joint observation,
feedforward plus K2/s feedback, Gaussian initial offset, persistent bias and
step noise. It does not establish rigid-body, hardware or observation-error safety.

Useful verified improvements are equal nominal preparation for learned MPD
initialization, the old-fixture device-native covariance-energy implementation,
and optional decision-equivalent lazy physical validation. Phase8 nominal
preparation restored15/16 MPD slots; phase9's prepared MPD admitted7/8 tasks
with or without further correction. Sampled CVaR reduced MPD cost/risk slightly
relative to dense correction on common tasks but gave no coverage gain and cost
far more compute. Phase11 native energy matched numeric/optimizer semantics and
passed the old-fixture timing gate. Phase13 lazy validation preserves fixed-N
selection for immutable streams, with measured benefit depending on the case.
These are separate initialization, optimization and validation capabilities.

Closed or rejected directions remain preserved: phase7's unequal-preparation
comparison is confounded; analytic sine moment closure improved moments with
little guidance-frontier benefit; the hybrid covariance builder missed its
runtime gate; sequential B/C admission can change fixed-N selection and was
not integrated. Phase12 fresh covariance confirmation stopped at its frozen
nominal-health gate (classical2/4, MPD3/4; task175075 descriptor invalid). No
replacement, surviving-arm comparison, or covariance sweep follows.

Task42576's immutable initial-joint event has exact Gaussian probability
26.58255% under the declared model, a lower bound on complete-execution failure.
It is not an empirical confidence bound and later control cannot remove that
initial event. The5% and10% budgets are unattainable on that task in this model.
Zero observed failures elsewhere are finite-sample evidence, not zero-risk proof.

Remaining questions concern reliable nominal initialization on valid fresh
descriptors, risk-constrained cost headroom under strong feedback, local distance
surrogate calibration and approximation error, and transfer to more realistic
arm dynamics and observation uncertainty. Exact tracking covariance alone does
not certify nonlinear collision/pose probabilities. A new scientific study
needs a new frozen question and authorization; no further experiment is queued.
