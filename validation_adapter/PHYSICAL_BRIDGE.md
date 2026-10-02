# Current physical validator bridge

The bridge is an opt-in utility. Existing fully batched validator/study scripts
are untouched. `validate_current(...,mode='fixed')` is the default; lazy must be
explicit. No B/C or stopped-sample early admission exists here.

```python
from fractions import Fraction
from validation_adapter import Request
from validation_adapter.physical_bridge import (
    freeze_candidate, PhysicalBackend, validate_current,
)

# Inject the application's CURRENT qualified API and fixed robot instance.
# Freeze costs/eligibility using its existing rules BEFORE validation.
cs = [freeze_candidate(slot, executed_nominal_float64[slot],
                       cost=frozen_costs[slot], eligible=frozen_eligibility[slot],
                       expected_path_hash=frozen_path_hashes[slot])
      for slot in (0, 1)]
req = Request(request_id, immutable_model_id, admission_seed, Fraction(1, 20))
backend = PhysicalBackend(current_core_api, current_robot_core, goal_pose,
                          model_id=immutable_model_id,
                          synchronize=torch.cuda.synchronize)
result = validate_current(cs, req, backend)  # fixed N1024, alpha.025 default

# Optional measured configuration; never infer that small feedback is cheap.
backend = PhysicalBackend(current_core_api, current_robot_core, goal_pose,
                          model_id=immutable_model_id,
                          synchronize=torch.cuda.synchronize,
                          partial_feedback=True, kernel_rollouts=16)
result = validate_current(cs, req, backend, mode='lazy', chunk_size=16)
```

The physical bridge requires201x7 float64 paths,3x4 finite goals,N1024 and
alpha exactly.025. It makes private immutable snapshots, checks declared path/
goal hashes and original full error-bank identity, then slices e/v/u arrays
without reseeding. Nine correlated event columns reduce to one whole-execution
failure. Original physical API/model must stay fixed during the synchronous
session; identity is supplied by the caller, not inferred from a file name.

Nominal validity is NOT newly imposed as admission eligibility. Keep the current
finite/command rule exactly, along with original nominal costs and slot tie-break.
The bridge does not silently repair trajectories, change noise, or cache sampled
outcomes. Default fixed executes BOTH full candidates. Lazy can skip the later
candidate or reject after deterministic fixed-N futility, but certifies only atN.

Metrics distinguish attempted/executed/examined trials, full-bank construction,
actual source validator wall, geometry rollout chunks and EXPLICIT sync calls.
Underlying.cuda/.cpu operations may add implicit syncs. Geometry chunks are not
actual CUDA kernel launch counts. Timing includes bank copies/hashes and repeated
endpoint/statistical setup when feedback calls are small. A failed backend or
mutation yields no candidate; uncertain execution is never claimed free.

Only the prespecified saved-request kernel16/chunk16 smoke is measured here.
Larger feedback/kernel batches, GPU launch costs or other physical models may
erase speed gains. Fully batched fixed mode remains available/default; audit
and research scripts are not rerouted automatically.
