# Fixed-horizon validation adapter (opt-in integration)

This package is an engineering acceleration adapter. It imports only the Python standard library. It does not import historical artifacts, planners, robot models, Torch, NumPy or SciPy, and does not generate noise or execute a GPU backend itself. Sequential B/C remain experimental and are deliberately excluded.

`validate(candidates, request, backend)` defaults to fully scheduled `mode='fixed'`: evaluate both frozen candidates in slot order with N1024, exact one-sided CP alpha.025 per candidate, then choose lowest nominal J with slot tie-break among eligible admitted candidates. Supply explicit `eligible` derived from the current pipeline's finite/command checks. Eligibility is not replaced by point risk or nominal optimization. NaN cost is rejected as invalid API input; infinity remains ordered and is not silently added as an eligibility rule. Custom alpha can only tighten the .025 maximum; integration should retain .025 exactly.

Only `mode='lazy'` opts in to A: cost/slot order; certify at full N only; reject once examined failures exceed the largest fixed-N passing count; skip later candidate after the preferred candidate is certified. The two current boundaries are37 at epsilon.05 and83 at epsilon.10,N1024,alpha.025. Boundaries use exact integer/rational binomial comparisons, with no floating tolerance. Source fixed-N CP(beta upper)<=epsilon selection is decision-equivalent under the same eligibility and ordered trial streams. For an N so small that no count can pass, lazy can reject without execution.

A backend declares `partial_feedback`. False is the conservative default: lazy acquires N trials in one feedback call per visited candidate, still permitting candidate skipping. True allows chunk feedback (default16); final partial chunk is acquired exactly, no overshoot beyond N. Whole-batch draws are charged even if futility is detected partway through returned bits. `examined` may therefore be less than `executed`. The adapter never labels a stopped-sample CP interval valid; a returned candidate receives only a full-N fixed-CP test certificate.

## API and dependency boundary

Immutable metadata: `Candidate(slot,cost,eligible,path_hash,payload)` and `Request(request_id,model_id,seed,epsilon,horizon=1024,alpha_per_candidate=Fraction(1,40))`. Exactly slots0/1 are supported. Metadata is frozen; caller/backend must also keep underlying trajectory payload and model immutable. Generic payloads cannot be independently fingerprinted by this package; a robot bridge must verify the declared path hashes against actual trajectory bytes before validation and prohibit mutation during it.

Backend contract: `open(request,candidates)->session`; session exposes `partial_feedback`, `evaluate(candidate,start,stop)->Batch(tuple_of_boolean_union_failures)`, and `close()`. `evaluate` executes every whole trajectory in the specified half-open sample-index slice, returns every executed trial and preserves index order. It must not conceal extra backend-paid trials; if a backend requires padding, choose a compatible horizon/chunk or implement an accounting-aware bridge before integration. Do not substitute cached outcomes for physical execution in a latency benchmark. Return synchronously with execution complete (including necessary CUDA synchronization) so host `backend_seconds` means completed validation wall time. Actual kernel/transfer counters belong to the bridge, not this generic adapter.

`FrozenBankBackend` is a small supplied helper: its factory is called once per validation request with full original N; it returns `Bank(payload,sample_count)`. A supplied slice function slices that bank without calling RNG. A supplied evaluator returns one row of declared binary event columns per whole execution. `whole_execution_failures` reduces columns with any, validates width/count, and never flattens timesteps. For phase9-like errors, preserve exact original `noise(N,seed,scale)` shapes/RNG call order, then original tracking/interpolation, and slice each e/v/u array. Reseeding separate chunk-length generator calls is not equivalent. Candidate evaluations share the same world indices. Repeated `validate` calls open fresh sessions/factories: the same seed regenerates the same full stream, not the next global RNG segment or a reused mutable cursor. New independent requests must receive correctly allocated new seeds; repeating a seed is not fresh risk evidence.

Example bridge outline (not executed here):

```python
from fractions import Fraction
from validation_adapter import Bank, FrozenBankBackend, Request, validate

# Existing application supplies physical noise/model and frozen candidate payloads.
backend = FrozenBankBackend(
    factory=lambda req: Bank(existing_full_errors(req.horizon, req.seed), req.horizon),
    slice_bank=lambda errors, a, b: tuple(array[a:b] for array in errors),
    evaluate_events=lambda candidate, errors: existing_whole_execution_events(candidate.payload, errors),
    event_count=9,
    partial_feedback=False,  # enable only after integration review
)
request = Request(request_id, declared_model_id, admission_seed, Fraction(1,20))
result = validate(frozen_candidates, request, backend)  # original behavior default
# Opt-in after review: validate(..., mode='lazy', chunk_size=16).
```

Output includes selected slot/cost or explicit rejection, full-N certificate with path/model/seed/sample-index identity, per-candidate statuses including unresolved/skipped, examined/executed/attempted counts, backend calls, backend/open/close/adapter/total wall timings. Backend failures or close failures return no candidate. After uncertain failed execution, `executed_counts_exact=False` distinguishes confirmed executed counts (a lower bound) from attempted draws; do not report unknown failures as free or successful trials. API/configuration errors raise before backend execution.

## Guarantees and performance limits

Conditional on two fixed trajectories and IID whole-execution trials under the declared model, each fixed-N test false-admits with probability<=.025; union gives P(return a candidate with truep>epsilon)<=.05. Common world draws across candidates are allowed; correlated timesteps/events are not independent trials. This is not P(unsafe|returned)<=.05, hardware safety or a novel research theorem. Lazy changes allocation, not fixed-N admission/selection. B/C and their near-threshold changed-selection evidence remain isolated experiment results, not part of this production adapter.

Lazy may lose when GPU batches are large, launch/host synchronization is costly, endpoint checks/setup repeat per chunk, or full noise generation dominates. For a cheap eligible preferred candidate, lazy still needs N draws to certify; original may evaluate two candidates efficiently in parallel. Backend chunk feedback must not be assumed cheap. Use `partial_feedback=False` for costly/nonstreaming backends and retain fixed mode by default until the integration review measures deployed end-to-end behavior. The existing measured8-request backend16 experiment is supporting engineering evidence, not proof for every backend.

## CPU regression and integration plan

Run from repository root using an existing NumPy/SciPy environment:

`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/kristoff/miniconda3/envs/h28-safe-mpd/bin/python -m unittest validation_adapter.test_adapter -v`

Production package has no scientific-stack dependency; only tests use NumPy for saved NPZ files and SciPy as an independent CP oracle. Saved-stream tests read original source phase9 outcomes at the explicit SOURCE constant; integration can change only that test fixture locator to the receiving checkout or an approved fixture location. No historical inputs are copied into this package.

Tests cover all96requests/192streams: original default and five lazy configurations (chunks1,16,31,1024 plus unavailable chunk feedback). Additional tests cover thresholds/neighbor beta CP checks, all-success/all-failure and threshold-adjacent front/back streams, ties and reversed input order, eligibility, preferred unresolved, exhaustiveN8streams, final partial chunks, repeated seed/call identity, illegal repeated/skipped indices, event union/width validation, malformed backend results, fail-closed exceptions/close failures and impossible boundaries.

Minimal integration: cherry-pick only the dedicated adapter commit (three files in validation_adapter/), or apply only those paths. Do not cherry-pick earlier experimental commits containing historical replay/GPU artifacts. No existing risk-project files are modified by the adapter commit. Review one narrow bridge to the CURRENT execution validator, with current eligibility/cost/path/model semantics; do not overwrite newer risk-energy work or import old phase7 core into production. Keep fixed mode as default; expose lazy as explicit configuration only after bridge regression and deployed timing review. No integration/merge, GPU or planner run was performed for this adapter preparation.
