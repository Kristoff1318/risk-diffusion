# Opt-in lazy physical validation: integration result

Completed on branch `improve/nonlinear-risk-calibration`. Pre-integration tag
`pre-lazy-validation-integration-2026-10-02` pins `d214e492`. Source `e679813`
was inspected and cherry-picked as `67a1ee8`: only the three minimal adapter
files, with no experimental branch merge. The bridge and prespecified smoke
were committed as `a561dd4` before outcomes. The source worktree was clean and
there was no overlap. See `PRESERVATION.json`: all 1,128 pre-integration tracked
files were byte-identical before the status append. No original core, physical
model, controller, obstacles, noise, historical result or default was changed.

`validate_current` defaults to fully batched fixed N. Explicit lazy mode orders
frozen eligible candidates by cost and slot, rejects only after irreversible
fixed-N futility, and skips later candidates only after preferred full-N
certification. It never early-admits. Original N1024, per-candidate CP alpha.025,
physical model, path hashes, costs, eligibility, seeds and sample identities
are preserved. One full original noise/tracking bank is generated per request,
then sliced without reseeding. Caller supplies a qualified immutable current
physical model; snapshots and hash checks protect paths, target and error bank.

## Verification

All 14 CPU tests passed (1.60s command): all 192 saved streams/96 requests,
independent CP thresholds, exhaustive small-N cases, ties, eligibility,
fail-closed behavior, immutable physical bridge and full-batch fallback.

Exactly two prespecified saved requests were executed once per mode against
unchanged `phase7/core.py::audit`, with geometry chunk16 and feedback chunk16.
Every acquired nine-event row matched the original saved stream at its original
index. Fixed and lazy selected the same original slot; every certificate uses
all 1,024 draws. These repeated admission seeds supply parity evidence, not
independent risk evidence.

| Saved request | Mode | Selected slot | Executed / examined draws | Source audit calls | Geometry chunks | Explicit syncs | Physical validator seconds | End-to-end validation seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
|t1_mpd_none_d0.05|fixed|0|2048 / 2048|2|128|6|2.572395|2.774657|
|t1_mpd_none_d0.05|lazy|0|1024 / 1024|64|64|130|1.291508|1.488094|
|t0_mpd_dense_d0.1|fixed|1|2048 / 2048|2|128|6|2.528016|2.726149|
|t0_mpd_dense_d0.1|lazy|1|1984 / 1980|124|124|250|2.505792|2.708354|

The healthy request certifies slot0 (one failure) and skips higher-cost slot1.
The difficult request rejects cheaper slot0: full-N has90 failures; lazy reaches
its84th failure at examined draw956, executing the complete960-draw chunks.
Slot1 is then certified with15/1024 failures. Costs remain3.970341 and5.518832.
The previously explored sequential B/C rule would select the cheaper slot after
64 zero failures on this saved case. Those variants are excluded; no B/C ran.

The healthy case had a large saved-work benefit. The difficult case saved only
64 executed draws and had negligible timing benefit. Two warm, fixed-order
measurements do not establish general speed or optimal chunk size. Small chunks
increase source calls and synchronization. Geometry chunks are not CUDA launch
counts; actual kernel events and implicit CPU-transfer synchronization were not
profiled. End-to-end timing includes original full-bank construction, copies,
hashes, private tensors, real validator calls and adapter decisions. Bank
construction/tracking took0.093–0.097s per call; it is not assumed free.

An initial smoke import picked top-level `core.py` instead of phase7 core. It
stopped before GPU allocation or validation. That attempt's JSON/log/time are
retained; import precedence was corrected and committed as `e18cfbb` before any
physical execution. The same frozen pair, seeds and settings were retained.

Successful smoke command10.81s; process9.875s; peak allocated GPU684,407,296bytes;
RSS1,435,576KiB. See raw `GPU_SMOKE.json`, logs and `RESOURCE_LEDGER.json`.
Engineering conservative booking78.43/180s, with output cap30MB. This ledger is
separate from the stopped phase12 scientific booking198.42/450s. No new planning,
optimization, model inference, correction sweep, independent audit or task
replacement occurred. GPU process exited and slot released.

Use the opt-in example in `validation_adapter/PHYSICAL_BRIDGE.md`. Original
study scripts stay on their existing fully batched validator. Phase12 remains
stopped at its nominal-health gate; this engineering result changes neither
its scientific conclusion nor the future-study authorization requirement.
