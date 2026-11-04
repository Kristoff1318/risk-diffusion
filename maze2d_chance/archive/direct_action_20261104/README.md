# Native Maze2D chance-guided diffusion baseline

Implementation and bounded experiment for slides 13–14 of the original `main.pptx`.
**The audited experiment failed the safe-generation criterion:** no accepted plan,
no healthy held-out no-noise plan, nominal failure 97.39%, guided failure 100%.
See [the full report](results/REPORT.md) for per-plan confidence intervals,
calibration limitations, fidelity, uncertainty, costs, and runtime.

The script reuses the older repository **read-only** for its pretrained
TemporalUnet/GaussianDiffusion definitions and checkpoint. It does not use the
older risk dynamics or planning scripts. No new training or downloads are needed.
The cached normalization limits were obtained from the checkpoint's dataset
configuration by `probe.py`; the main runner never rebuilds the dataset.

Run from this worktree using the existing native D4RL environment:

```bash
OPENBLAS_NUM_THREADS=1 /home/kristoff/miniconda3/envs/diffuser/bin/python maze2d_chance/validate.py
OPENBLAS_NUM_THREADS=1 /home/kristoff/miniconda3/envs/diffuser/bin/python maze2d_chance/baseline.py --seeds 3 --executions 200 --strength .002 --out reproduced
```

`--out` must be fresh to preserve previous evidence. The source writes its
protocol before sampling. The archived `results` run predates that protocol-file
convenience; its frozen settings are in `results/manifest.json` and the report.
`report.py` renders the archived `results` directory only.

- `baseline.py`: direct-action DDPM guidance; differentiable native free
  transition, local covariance, native box/goal CDF probabilities, literal
  unnormalized softplus energy, stochastic native executions and exact audit.
- `validate.py`, `validation.json`: numerical dynamics, covariance and full-energy
  gradient validation, plus collision-free native start/goal checks.
- `probe.py`, `probe.json`, `check_sampler.py`: disclosed initial nominal-health
  and RNG diagnostics, independent of final held-out generation.
- `pilot`, `pilot_scale`: all preliminary outcomes, retained to disclose tuning.
- `results`: 36 plans, 7,200 independent within-plan stochastic executions;
  counts and exact per-plan intervals, plans, denoising traces, representative
  native/noise/free-model rollouts, report and equal-aspect trajectory figure.
- `slide_evidence`: exact six embedded equation PNGs and white-background JPGs
  used for inspection. Original deck SHA-256:
  `f5f827c82c5fed9319f96733a8bb09482db4a389278b0c3467766aaf96896ec1`.

Native MuJoCo contact dynamics remain authoritative during evaluation. The free
local Gaussian model and its CDF constraints are approximations, so a surrogate
union sum is not a universal safety bound. Rejection is distinct from generation
quality; no accepted sample means the false-safe fraction is undefined.

Map verification: `verify_map.py` checks exact array equality against the legacy
loader/renderer, native registration and LARGE_MAZE, and every physical wall.
See `results/map_verification.png` and `.json`. The corrected trajectory figure
uses the familiar legacy orientation and complete boundary at equal physical
aspect; only the original image viewport cropped walls, never the simulation.
