# Terminal old-fixture qualification: numeric pass, runtime gate failure

GPU slot explicitly released on completion. No further GPU calls, fresh-task
inspection or fresh study follow this result. Code/protocol were frozen at
801db1a before outcomes. The frozen phase11 objective/event set is unchanged.

## Full event and backend verification

CUDA float64 on RTX4090 evaluates the analytic CPU-built frozen models.
All four saved phase5 paths enumerate exactly4,323,770 events each:2,001
states times2,158 geometry/joint events,5,600 velocity sides and12 terminal
box sides. No pair pruning, independence product, probability clipping or
normalization by event count. Only the fixed5% budget normalization applies.

First full-grid CPU/GPU energy, cost, component sums and nominal-state gradient
parity pass; maximum nominal gradient discrepancy8.88e-16. Quintic spline
reconstruction error8.88e-16; the first state has zero free-control derivative.
The active-risk spline probe reuses the SAME frozen model: directional finite
difference-27,716,181.40698 versus analytic-27,716,181.41651, relative error
3.44e-10. CPU/GPU control-gradient maximum absolute discrepancy2.98e-8.
This verifies the frozen affine surrogate and B-transpose gradient mapping,
not a derivative through rebuilt Jacobians or nonlinear Gaussian moments.

## Unchanged anchors versus independent existing measurements

The independent measurements are explicitly reused from phase6/RUN_256_10.json,
DOUBLE regime, same saved paths/targets,5ms hard validation and K2 controller.
No new risk draws ran. Each path's audit is0/256 union failures, exact two-sided
95% interval[0,1.430636%], and0/256 for every individual component.
These wide intervals do not calibrate the very small surrogate probabilities.
Terminal component boxes are stricter local-model events than nonlinear
position/orientation norm failures, so their estimates are not identical events.

|Old saved slot|World MPD sum|World cuRobo sum|Self sums|Joint sum|Command/executed velocity sums|Terminal box sum|Total R|Existing hard union|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|0|2.584e-253|2.743e-115|0*|0*|0*|5.246942e-9|5.246942e-9|0/256; CI[0,1.430636%]|
|1|1.421e-84|4.578e-110|0*|1.227e-64|0*|4.367070e-6|4.367070e-6|0/256; CI[0,1.430636%]|
|2|6.373e-89|6.517e-57|0*|2.159e-151|0*|1.507284e-6|1.507284e-6|0/256; CI[0,1.430636%]|
|3|2.269e-110|6.552e-99|0*|2.007e-204|0*|1.367444e-11|1.367444e-11|0/256; CI[0,1.430636%]|

*Machine-precision zero tails may underflow; no exact zero-risk claim.
Complete per-side counts/sums and the original audit counts are preserved in
GPU_QUALIFICATION.json. All four anchors are locally valid and R<=.05.
Thus naive union conservatism did NOT make these four healthy fixtures
unusable. It is still a local surrogate, not a nonlinear risk certificate;
no challenging boundary-case calibration or general conservatism conclusion
follows from these unusually safe anchors.

## Eight-step timing: terminal failure

The timed request uses old slots0 and1, each with its own unchanged start and
target, solely as two timing slots. No proposal generation, risk admission,
planner-method comparison or independently audited corrected-plan claim.
Both completed all8 Adam steps,lr.003,gradient cap20, with local model rebuilt
per iterate. All geometry/Jacobians/projected variances use the same frozen
specification; no full-Jacobian-gradient substitution. Initial state preserved.

Synchronized two-slot request wall:13.352060s, versus2s target (**FAIL**).
The measurement includes CPU construction, GPU transfers, evaluation/backward,
optimizer setup/steps, best-completed-iterate selection and path extraction.
Per-step recorded wall is approximately.71–.88s; full-grid CPU construction
at unoptimized anchors is.69–.71s. The CPU builder is the dominant architectural
cost; isolated CUDA evaluation latency must not be presented as correction
latency. The full8 steps were run to measure the gate, not silently truncated
or reported as a2s deadline-enforced success. The two slots are different
fixtures, so historical sampled-CVaR timings are context, not a paired speed
superiority test.

Best completed iterate6 for each slot: R=.00322167 / .00972988 and
J=3.95931 /14.63933. These are optimizer diagnostics ONLY. Corrected paths
are saved in OLD_TIMING_PATHS.npz and have no independent risk audit.
They do not establish risk-constrained cost improvement.

Peak allocated GPU474,019,328bytes; peak RSS2,225,572KiB. Main host command
wall17.92s; runner internal wall16.91s. The sandbox attempt could not see CUDA
and is preserved separately (.90s); the authorized host route succeeded.

## Decision and budget

Stop this unchanged prototype at its declared runtime gate. No hidden
redesign, geometry pruning, changed coefficients or normalization followed.
The phase10 fresh confirmation plan's2s gate fails for this implementation;
the unchanged plan must NOT proceed to fresh outcomes. A different backend
would be separate explicitly authorized work, not an automatic rerun.

Additional conservative booking44s =18.82s timed commands plus25.18s estimated
ancillary inspection/code/report/preservation/commit allowance. This is within
the90s additional cap; prior booking20s yields64s total against450, leaving386s.
Ancillary allowance is estimated, not a measured total. Artifact growth remains
well below30MB. No new assets, installs, training, hardware, inference or new
tasks were used. Existing original files and phase10 documents are unchanged.
