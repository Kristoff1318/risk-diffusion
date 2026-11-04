# Frozen covariance-energy prototype v1

Scope: CPU equations and saved phase5 fixtures only. No new tasks, sampling,
planning, inference or training. This specification and implementation are
committed before any derivative/energy qualification outcomes are inspected.
The phase10 design and confirmation plan are preserved verbatim.

## Objective and derivative convention

E = J + 500 [max(0, (R - delta)/delta)]^2, delta=.05.
J = .05 sum velocity^2 + .005*.05 sum acceleration^2, unchanged.
R is the unnormalized sum of EVERY elementary event probability. No division
by event count, independence product, top-k pruning, or tuned allocation.
This is direct union summation, not an allocation heuristic. No risk buffer.
The future sequential optimizer would use eight Adam steps, lr=.003,
gradient norm cap20, and rebuild the local model at each iterate. Best completed
iterate: lowest J among R<=delta, otherwise lowest R. The optimizer is not run
or supplied in this preparatory phase; the implemented interface evaluates E
and its analytic gradient with respect to the 201 nominal joint states.

Geometry and pose are AFFINE expansions at the current nominal anchor.
Clearance/pose Jacobians and projected variances are frozen for evaluation and
differentiation of that iterate's surrogate. No distance Hessian or derivative
through the Jacobian is claimed. Finite differences hold the frozen model
fixed. Rebuilding it on either side tests a different function and is not a
valid gradient check. Rebuilding each iterate is a sequential approximation,
not the full gradient of a nonlinear Gaussian-moment objective.

## Model and complete events

dt=.05, H=200, K=2, initial SD .004 rad, bias SD .002 rad/s,
step velocity SD .010 rad/s; independent joints and mutually independent
initial/bias/step noise. Exact covariance uses the existing linear tracking
law, including the persistent-bias cross term in executed velocity.
Dense state covariance is for linear interpolation between adjacent command
states; it is not terminal covariance copied across time.

Production event set: all 2001 states at 5ms; both native sphere models, all
10 native world spheres and all configured self pairs INCLUDING duplicates;
two sides of each of seven physical joint limits at every state; two sides
of seven command and executed velocity limits at each of 200 steps; terminal
position and target-frame rotation-log component boxes, each coordinate's
half-width respectively .01/sqrt(3) m and radians(2)/sqrt(3).
The terminal FK uses the MPD native tree as the existing hard validator does.
The component boxes are contained in the norm-tolerance balls for the local
linearized pose, hence sufficient but conservative for that local model.
Box events differ from the hard validator's norm events and are disclosed.
Counting BOTH boxes' sides gives 12 terminal events.

Affine joint/command/executed velocity Gaussian marginals are exact under the
declared tracking model. Geometry/rotation/position marginals are approximate
for nonlinear events. Union summation bounds the affine surrogate union, not
the nonlinear whole-arm union. Temporal and spatial dependence is retained
in the covariance model but no exact joint union integration is performed.

## Numeric edge rules

Signed margin<0 fails; exactly zero is nonviolating, matching the validator.
For sigma=0 probability is exactly 1 if margin<0 else 0; derivative=0 away
from the discontinuity (also returned as0 at the boundary, explicitly not a
smooth derivative there). Positive sigma always uses Gaussian tails, with
no epsilon SD floor or probability clipping to manufacture acceptance.

Nonfinite inputs/covariance/Jacobians raise an error. Covariance symmetry and
PSD tolerance are absolute1e-14 plus relative1e-10 times matrix scale.
Materially indefinite matrices are rejected. Only tolerated roundoff-negative
eigenvalues/projected variances are clamped to zero and disclosed in metadata.

Center separation<=1e-12 m is singular: its event probability is forced to1,
its clearance derivative is0, no division by zero occurs, and the model is
marked invalid for optimization/admission. This is conservative rejection,
not an artificial normal for penetrating/coincident spheres. Analytic FK
Jacobians retain every configured ancestor axis and origin.
Rotation log within1e-6 rad of pi is also invalid: all its terminal orientation
events are forced to1 with zero derivative. No unstable pi-axis derivative.

## Qualification fixed before outcomes

Unit checks: augmented covariance parity, dense interpolation parity,
executed-velocity bias covariance, sigma-zero semantics, covariance PSD
rejection, coincident-center rejection, affine marginal tail values,
analytic cost derivative, scalar frozen-risk derivative and pose box semantics.
Old fixture checks: phase5 first saved path only; analytic FK/point/clearance/
rotation-log derivatives versus centered differences at fixed perturbation
1e-6 rad; native hard geometry margin parity; frozen-surrogate directional
finite differences with1e-6 nominal perturbations.
Numerical tolerances: covariance absolute1e-18; clearance/pose derivatives
absolute1e-6; frozen-energy relative1e-5 plus absolute1e-4.

Bounded CPU energy check uses exactly first-path dense indices
[0,1,10,500,1000,1500,1999,2000], all pairs at those states, all 200 velocity
steps and terminal pose. This subset is a labelled numeric/derivative fixture,
never a production risk estimate. Production full-grid count is reported.
If this subset already R>.05, the full positive sum cannot satisfy the budget
at that same anchor; report it without changing coefficients or pairs. This
does not prove all other anchors fail, nor true nonlinear infeasibility.
No automatic redesign. GPU full-grid timing, full-gradient backend parity and
old healthy-fixture usefulness remain deferred to a released GPU slot.

Budget:120 aggregate command seconds /25MB, credited against450s/50MB future
confirmation cap. All preparatory evidence remains phase11; no phase10 edits.
