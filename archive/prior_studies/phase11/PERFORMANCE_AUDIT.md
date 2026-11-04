# CPU engineering audit of the13.35s implementation path

No benchmark, GPU call, optimizer or fresh task was run for this audit.
The frozen fresh plan remains stopped. The runtime failure applies to this
CPU-builder/CUDA-evaluator implementation, not analytic risk energies generally.
All conclusions below use saved timings plus code inspection.

## What the CPU builder actually does

`qualify_gpu.py:126` first synchronizes/copies201x7 nominal states from CUDA
to CPU and constructs `cpu.FrozenEnergy`. This is not model training/loading,
an asset download or32 stochastic FK executions. `panda_p0.Arm()` is created
once at runner line41: URDF/YAML/scene parsing is NOT repeated each step.

`FrozenEnergy.__init__` then does:

1. Copy/check anchor; rebuild the2001 indices, interpolation weights and dense
   nominal path. Recompute2001 scalar tracking variances in a Python loop.
2. Materialize2001x7x7 isotropic covariance matrices and run batched symmetric
   eigen-decompositions/PSD reconstruction every iterate, although the entire
   covariance schedule is independent of the nominal path and already qualified.
3. For EACH native robot tree, traverse FK at2001 states, copy7-axis/origin
   ancestor arrays per link, and reconstruct every sphere's center and3x7
   point Jacobian in a Python loop. Rebuild static radii and self-pair integer
   index arrays. Compute all world and configured self separations/normals,
   signed clearances, all seven clearance derivatives and singular masks.
4. Project covariance through each clearance Jacobian using a general
   `J C J^T` contraction over4,290,144 geometry events; check variances and
   store full clearance/Jacobian/SD/mask arrays.
5. Run terminal FK again for one state, SciPy rotation log and pose-box
   Jacobians/projected variances. Assemble validity metadata.

`GPUFrozen.__init__` then uploads ALL new arrays plus repeated invariants
(indices, variances, limits, anchor and recomputed velocity SDs). Geometry
constants alone are313,180,512bytes/iterate:240,248,064bytes of Jacobians,
68,642,304bytes of clearances/SDs,4,290,144bytes of masks. Across16 iterations
that is5,010,888,192bytes of logical geometry upload. This is calculated
payload, not measured PCIe traffic or bandwidth.

CUDA evaluates the frozen affine probabilities/objective and backward;
`gpu.numeric` converts each component and scalar to host floats each iterate,
introducing synchronization points. Adam/clip/selection then run on CUDA.
The loop deletes large objects and calls `gc.collect()` AFTER the recorded
per-step timer. The surrounding request timer includes that collection,
optimizer initialization, path extraction and other outside-step overhead.

## Recorded costs and their limits

|Timing slot|Eight recorded step times, seconds|Sum|Mean|
|---|---|---:|---:|
|0|.881660,.797654,.725607,.713472,.826607,.786648,.795565,.846888|6.374102|.796763|
|1|.782343,.728080,.875405,.784468,.798878,.838161,.874181,.780914|6.462429|.807804|

Total recorded step intervals12.836532s; complete request13.352060s.
Outside-step residual.515528s combines collection/setup/extraction/other work;
no separate attribution is justified from these logs.

At the four unoptimized anchors, CPU builders took.710052,.693083,.691886,
.692469s. Corresponding combined build/transfer/evaluate/backward times were
.916814,.716859,.715647,.716135s. Warm residuals on slots1–3 were
.023776,.023762,.023665s. First-slot residual.206762s includes first-use
overhead. Those intervals do not isolate transfer from evaluation/backward,
and do not measure the optimizer steps. Per-iteration CPU build, FK, covariance,
projection, upload and autograd subcosts were NOT separately instrumented.
Thus “CPU construction dominates” is supported by the anchor measurements,
but an exact FK-versus-projection percentage is not known.

## Minimal semantics-preserving proposal — not implemented or authorized to run

One refactor: retain the existing frozen objective/evaluator and add a
GPU-resident analytic local-model builder with a reusable invariant context.

* Cache once: native topology/topological ordering/ancestor masks, joint
  origins/axes, sphere links/local centers/radii, world centers/radii, EXACT
  configured pair IDs/order/duplicates/buffers, limits, dense indices and
  interpolation weights, covariance/velocity schedules and spline basis.
  Check PSD once per immutable covariance configuration. In this isotropic
  fixed model, projection `var(t)*sum(J^2)` is algebraically identical to
  `J C J^T`; retain checks/roundoff metadata and compare against the oracle.
* At EACH iterate, use `no_grad`/detached anchor to rebuild native FK, centers,
  analytic point/clearance Jacobians, normals/clearances, projected SDs and
  singular masks on CUDA float64. Vectorize over states and spheres/pairs;
  a short static joint-tree loop can remain. Frozen tensors must stay detached
  during objective backward. Do NOT cache old Jacobians, SDs or clearances
  across changed anchors, introduce distance Hessians, or drop duplicates.
* Reuse the existing differentiable frozen affine evaluator and per-slot Adam,
  gradient cap/best-iterate rule. Cache invariant device tensors, keep the
  bulk geometry tensors on device, transfer only necessary scalar diagnostics
  and final paths. Avoid repeated forced garbage collection if lifetimes are
  bounded. Accumulate optional component logs on device until after timing;
  preserve selection/nonfinite checks and charge their actual synchronization.

The FK/point/clearance/projection equations have direct Torch equivalents:
this is a backend engineering port, not a new energy or statistical assumption.
It is NOT just caching the current model object, since anchor-dependent
geometry must change. Rotation-log branch/singularity parity is the delicate
part: minimal port may retain the existing one-state CPU/SciPy pose builder
and transfer its tiny detached12x7 model rather than invent a new rotation-log
approximation. Its synchronization cost would need measurement. Full GPU
rotation-log implementation would require the same branch/edge qualification.
No candidate batching, kernels, mixed precision or compilation is necessary
for the minimal proposal, and none is assumed in its cost estimate.

## Exact equivalence tests required before any timing rerun

1. Unchanged source fixture hashes; both native FK transforms, center/point
   Jacobians, ALL pair clearances/Jacobians/masks/SDs at all2001 states against
   the CPU oracle. Preserve pair ordering/duplicates and per-component counts;
   exact total4,323,770. Analytic covariance/velocity arrays remain the oracle.
2. Existing sigma0/contact tests, coincident and near1e-12 centers, nonfinite
   inputs, PSD tolerance, and rotation-log nearzero/nearpi branches. Invalid
   models reject optimization. No SD floor, smoothed event or tail clipping.
3. Per-component sums,R,J,E, nominal and spline gradients against existing
   frozen CPU/CUDA results: absolute1e-6 plus relative1e-8. Clearance/pose
   Jacobian absolute1e-6. On active-risk probes, central finite differences
   1e-6 must HOLD THE NEW FROZEN MODEL FIXED, with the original derivative
   tolerance absolute1e-4 plus relative1e-5. Initial state derivative exactly0.
4. Sequentially rebuild both implementations at IDENTICAL recorded old
   iteration anchors; compare outputs/gradients/clipped updates and8-step
   trajectories, per-slot Adam state and selected best iterate. Float reduction
   differences near threshold/ties must be surfaced, not silently change
   selected candidates. No claim of bitwise identity across CPU/GPU reductions.
5. Separately instrument construction/projection, transfers, evaluation,
   backward, Adam/checks/logging and collection; synchronized WHOLE two-slot
   request remains the2s gate. Same event count and temporary-memory accounting
   accompany timing. All this is proposed future qualification, not run now.

## Evidence-based cost expectation

This refactor targets the measured~.69–.71s full builder and repeated313MB
upload each step. Cached invariants remove definite repeated operations but
their isolated cost is unmeasured, so caching alone cannot be assigned a
speedup. GPU-native rebuild cost is also unmeasured; no numerical end-to-end
speed prediction is defensible yet.

The observed warm evaluator/transfer/backward residual is~23.7ms per anchor,
which shows the current fixed-model CUDA evaluation is not itself a.7s
operation. If that residual and the.5155s request overhead persisted,16
residual intervals plus overhead would cost~.895s BEFORE any new GPU builder
cost. This is a conditional accounting scenario, not an achievable bound or
speed forecast: optimizer/active-loss/state differences can change it.
With the measured outside-step overhead unchanged, the2s gate affords92.78ms
TOTAL per step; subtracting the warm23.7ms residual leaves about69.1ms per
step for new construction and other added costs. Whether that is attainable
requires profiling the port, not extrapolating CPU/GPU hardware ratios.

Therefore the terminal result is an implementation runtime-gate failure.
It does not establish that the analytic energy is universally slow, nor that
a straightforward port will pass2s. No energy redesign or rerun follows this
audit; adaptive validation retains the GPU slot and the fresh plan is stopped.
