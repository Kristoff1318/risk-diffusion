# Risk planning study status

Phase7 is preserved as a confounded comparison: classical included IK plus nominal refinement while MPD was raw. Its zero MPD admissions do not diagnose risk correction on a healthy nominal generator.

Phase8 applied identical 80-step nominal refinement to the same 16 MPD samples without IK/projection, restoring 15/16 nominally valid slots over 8/8 tasks. It did not evaluate stochastic acceptance.

Phase9 is a completed diagnostic replay of the same proposals after equal nominal preparation, comparing no correction, eight dense deterministic steps and eight sampled-CVaR steps. Classical admissions are 3/5/5 at 5% and 4/5/6 at 10%; MPD is 7/7/7 at both. CVaR has mixed classical gains, and improves MPD cost/risk relative to dense correction without added coverage. No-extra prepared MPD already has zero observed selected audit failures. Correction steps match; sampled-gradient compute/runtime does not. Cached replay is not a cold or fresh held-out benchmark.

Task 42576 has a 26.58% initial joint-violation floor under the fixed physical uncertainty, so 5%/10% cannot be met by later commands. It remains in all arms. All 96 requests completed; no more variants or follow-on experiments have run.

See phase9/RESULTS.md and its raw/paired CSVs, confidence bounds, historical preparation times and resource ledger. Previous phases remain unchanged. No broad generalization or safety guarantee is established.

Phase10 adds a scientific synthesis, an analytic pre-planning initial-joint-event diagnostic and one four-task/four-arm covariance-energy confirmation plan. Only old task metadata and new index IDs were read; no fresh planning, generation, risk sampling or training ran. The recommended next cap is 450 command seconds / 50 MB, subject to separate execution authorization. See phase10/SCIENTIFIC_SYNTHESIS.md and phase10/FRESH_CONFIRMATION_PLAN.md.

Phase11 froze the covariance-energy objective and qualified its local affine
derivatives and complete event accounting on old fixtures. The original hybrid
CPU-builder/CUDA-evaluator failed the 2 s correction gate (13.35 s). A separately
authorized invariant-cache/device-native refactor preserved all 4,323,770 events,
float64, frozen-Jacobian semantics and optimizer sequences, then passed the old
two-slot/eight-step timing gate at 0.265 s. This is warm old-fixture engineering
evidence, not fresh risk-aware planning performance. Both implementations and
all failed/passing evidence are retained in phase11.

Phase12 received explicit authorization for the frozen four-task confirmation.
Protocol/code/assets were committed before outcomes. The study terminated at
its prescribed nominal-health gate: task 175075 failed native descriptor
validity due to cuRobo self-collision; classical preparation was healthy on
2/4 tasks, MPD on 3/4, below the required 3/4 for both sources. All four cases
remain. All 12 generated paths completed their 80 nominal steps; failures,
costs and source timings are preserved. No correction, admission, stochastic
prepared-input diagnostic or audit ran, so fresh dense-versus-covariance risk
benefit remains unassessed. No replacement, extra preparation or surviving-arm
comparison follows automatically. GPU slot released. See phase12/RESULTS.md,
STATUS.json and raw MAIN_RESULTS.json. Cumulative conservative booking is
198.42/450 s, with 251.58 s unspent; the variant is stopped, not scheduled to
use that remainder.

Phase13 integrated only the minimal decision-equivalent lazy validator commit
(e679813 cherry-picked as67a1ee8) and a narrow opt-in bridge to the current
unchanged physical validator. Pre-integration tag pinsd214e49; all1,128 original
tracked files were verified unchanged before this append. Fully batched fixedN
remains default. Lazy may deterministically reject at fixed-N futility or skip
higher-cost candidates after full-N certification; sequential B/C is absent.
All14 CPU tests and both prespecified real saved-request GPU parity cases passed,
including the changedB/Ccase. Every acquired event row and selected slot matched.
Healthy case2048→1024draws,2.775→1.488s; difficult case2048→1984draws,
2.726→2.708s. Chunk16/sync overhead and full bank/hash setup are charged; no
universal speed claim. No new planning/inference/optimization/independent audit,
no phase12 replacement or covariance sweep. GPU slot released. See phase13/
RESULTS.md, PROJECT_STATUS.md, GPU_SMOKE.json and engineering resource ledger.
