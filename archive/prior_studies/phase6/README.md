# Tracking benchmark replay

Frozen model and source: PROTOCOL.json. Full evidence: RESULTS.md. Preserve existing phase5/phase6 files. No install or download is needed.

An isolated sibling replay folder retains access to the unchanged phase5 source and bank:

```bash
cd /home/kristoff/Documents/experiments/research/risk_diffusion
mkdir phase6_replay
cp phase6/tracking.py phase6/PROTOCOL.json phase6/CHECKS.json phase6/PROFILE.json phase6_replay/
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /home/kristoff/Documents/experiments/mpd-seeded-qualification-20261001/.venv-headless/bin/python phase6_replay/tracking.py run 64 10
```

Use the supported host approval route when executing the chosen CUDA backend. The profile-selected backend is stored in PROFILE.json; host GPU denial must be reported, not bypassed. The source checks its own frozen hash and the unchanged P0 source/bank before runs. Output guards refuse to overwrite existing results. `qualify` runs affine feedback/covariance checks; `profile` compares independent NumPy and host GPU geometry. `run 256 10` evaluates 256 executions at 5 ms; `run 64 20` evaluates identical 64 physical paths at 2.5 ms. Gains/noise are not runtime-tunable.

Saved demonstration: DEMO_256_10.npz contains q[4,201,7], actual velocity v[4,200,7] and commanded velocity u[4,200,7] per path/regime. TRACKING_COMPARISON.png plots saved shared-noise evidence only.

The original P0 DEMO.png was privately delivered as library_file_id `libfile_5f5851ab7f508191a443ed19f84c4b60`. Its bytes remain unchanged; returned Library version/identity metadata was persisted locally.
