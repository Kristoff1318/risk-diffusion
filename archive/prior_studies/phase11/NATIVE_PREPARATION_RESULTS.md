# Cached/device-native preparation ready

Implementation/protocol frozen at72dbd4b before CPU checks.
Five CPU unit tests pass across the predeclared bounded old fixtures.
All-entry native block discrepancy at most3.33e-16; active frozen spline
finite-difference relative error3.39e-10. Native FK/points/Jacobians, all pairs,
component sums, nominal/spline gradients, cache identity/fresh-anchor behavior,
numeric edge rules and one-step clipped Adam arithmetic agree with reference.
Test command wall1.20s; suite internal.275s. No GPU queried or executed during
this CPU preparation. No new task/geometry/proposal or sampling/inference.

New code: native_energy.py. Existing CPU/hybrid implementations, energy
specification, gradient convention, event set and old outputs remain unchanged.
The refactor caches ONLY static topology/geometry definitions, exact pair
arrays and trajectory-independent covariance/grid/limit tensors. Every anchor
rebuilds FK, centers, means, Jacobians, SDs, masks and terminal pose. Float64.
Terminal SciPy one-state calculation remains for exact branch parity.
NATIVE_PROTOCOL.md defines the required numeric gates before GPU timing.

CPU readiness does not establish GPU equivalence or speed. The full-grid GPU
protocol and eight-step optimizer sequence must pass against retained reference
before any speed claim. Fresh confirmation remains stopped and unauthorized.

Conservative preparation booking20s (1.20s timed tests plus18.80s estimated
ancillary commands/edits/inspection/report/preservation/commits), within90s.
Cumulative booking94s against450; remaining356s BEFORE the separately authorized
next GPU check. No claim that its2s gate will pass or that356s suffices for a
future fresh confirmation. All counts and costs would need reconciliation.
