# Frozen phase7: complete and preserved

All 64 requests completed; no further planning job is running. Study source and cohort were frozen in fcd69580384a72d627940dda8325b9cfce8dcd4d before outcomes. Do not rerun study.py in this directory: it deliberately refuses to replace outcomes/.

RESULTS.md presents the four-arm comparison. FAILURE_DIAGNOSIS.md explains why the raw MPD arms fail nominal health and how this differs from the historically refined MPD pipeline. FAILURE_DIAGNOSIS.csv retains all 128 nominal/audit/admission categories and margins. MAIN_RESULTS.json and outcomes/t*.json are primary immutable terminal records; packed flag arrays retain per-execution evidence. PLANS.npz and BEST_ACCEPTED_PATHS.npz retain nominal trajectories and controls.

PROTOCOL.json fixes every task, sample count, physical assumption, effort ceiling and source hash. TASK_SELECTION.json records all eligible/excluded tasks. PROVENANCE.json covers reused read-only assets. QUALIFICATION.json and BASELINE_QUALIFICATION.json document the preceding correctness gates. No nominal MPD health gate was imposed, a material design limitation.

analyze_saved.py produced the existing summary/nominal diagnostics using recorded paths. diagnose_saved.py reads those saved diagnostics/results, checks admission and selection logic, and creates the failure report/ledger; it does not generate or optimize plans. Its original delivery snapshot reports the pre-existing preservation verification, which was separately repeated before commit. Historical failed analysis logs are retained.

RESOURCE_LEDGER.json separates measured command time from a conservative unmetered allowance. LIBRARY_DELIVERY.json records the successful private image delivery. Library metadata was persisted locally without changing image bytes. All original 225 project files and 89 phase5/phase6 tracked files matched preserved hashes at delivery.

No additional repair experiment has run. The recommended next experiment is the single nominal-refinement ablation in FAILURE_DIAGNOSIS.md, preserving this cohort and physical model.
