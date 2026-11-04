# Runnable Panda P0 demonstration

All code uses existing local runtime/assets; no install is required. Study specification: PROTOCOL.json. Full results: RESULTS.md. Saved motion: FROZEN_PATHS.npz (q[4,201,7], u[4,200,7], target[4,3,4]); sampled motions: DEMO_n256_sub10.npz; visual: DEMO.png.

Existing output names are guarded. To replay without overwriting evidence, create an isolated subfolder inside risk_diffusion and copy only the code, frozen specification, bank and qualification:

```bash
cd /home/kristoff/Documents/experiments/research/risk_diffusion
mkdir phase5/replay
cp phase5/panda_p0.py phase5/PROTOCOL.json phase5/FROZEN_PATHS.npz phase5/QUALIFICATION.json phase5/replay/
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /home/kristoff/Documents/experiments/mpd-seeded-qualification-20261001/.venv-headless/bin/python phase5/replay/panda_p0.py run 64 10
# Same physical noise, denser geometry diagnostic (approximately80s CPU for8 cases):
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /home/kristoff/Documents/experiments/mpd-seeded-qualification-20261001/.venv-headless/bin/python phase5/replay/panda_p0.py run 64 20
```

The saved source verifies the frozen bank and original asset SHA256 values before sampling. The first run hashes the1.77GB original dataset read-only; no full dataset deserialization is needed for replay. `freeze` is a one-time asset extraction and intentionally refuses to replace the bank. Independent qualification: `panda_p0.py qualify` and `extra_checks.py`, in a fresh copy if preserving all evidence. Noise draws are reproducible1024-sized streams so64/256 prefixes and coarse/fine grids coincide. Neither saved bank nor sample motion is new planner output; this is a synthetic kinematic tracking-error study, not hardware certification.
