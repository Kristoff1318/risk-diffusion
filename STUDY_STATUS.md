# Risk planning study status

Phase7 is preserved as a confounded comparison: classical included IK plus nominal refinement while MPD was raw. Its zero MPD admissions do not diagnose risk correction on a healthy nominal generator.

Phase8 applied identical 80-step nominal refinement to the same 16 MPD samples without IK/projection, restoring 15/16 nominally valid slots over 8/8 tasks. It did not evaluate stochastic acceptance.

Phase9 is a completed diagnostic replay of the same proposals after equal nominal preparation, comparing no correction, eight dense deterministic steps and eight sampled-CVaR steps. Classical admissions are 3/5/5 at 5% and 4/5/6 at 10%; MPD is 7/7/7 at both. CVaR has mixed classical gains, and improves MPD cost/risk relative to dense correction without added coverage. No-extra prepared MPD already has zero observed selected audit failures. Correction steps match; sampled-gradient compute/runtime does not. Cached replay is not a cold or fresh held-out benchmark.

Task 42576 has a 26.58% initial joint-violation floor under the fixed physical uncertainty, so 5%/10% cannot be met by later commands. It remains in all arms. All 96 requests completed; no more variants or follow-on experiments have run.

See phase9/RESULTS.md and its raw/paired CSVs, confidence bounds, historical preparation times and resource ledger. Previous phases remain unchanged. No broad generalization or safety guarantee is established.
