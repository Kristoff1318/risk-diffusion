# Maze2D Risk Diffusion

This repository contains its own Diffuser inference source, pretrained model,
normalization/configuration, nominal execution code, and experimental chance
guidance. It does **not** import or load files from another project folder.

The research target is the chance formulation in slides 13–14 of
[main.pptx](main.pptx), evaluated on native D4RL `maze2d-large-v1`.

## Layout

| Location | Contents |
|---|---|
| `src/diffuser/` | Local temporal model, diffusion sampler, Policy, normalizers, environment loader and utilities |
| `assets/maze2d-large-v1/` | Local pretrained checkpoint, JSON configuration, normalization and asset hashes |
| `maze2d_chance/` | Nominal runner, local asset loader, experimental chance guidance and verification scripts |
| `maze2d_chance/results/nominal_local_20261104/` | Current report, plans, native executions, plots and verification evidence |
| `third_party/diffuser/` | Upstream MIT license, source provenance and copied-file hashes |
| `main.pptx`, `maze2d_chance/slide_evidence/` | Research specification |

Historical `archive/` and `maze2d_chance/archive/` directories may exist locally.
They are ignored, untracked, and unnecessary for running this repository.

## Run

The tested runtime is Python 3.8, PyTorch 1.9.1/CUDA 11.1 and native D4RL/MuJoCo.
[environment.yml](environment.yml) records the runtime dependencies.
MuJoCo 2.0 binaries, system libraries and a compatible GPU/driver are installed
runtime prerequisites; they are not project source or vendored binaries.

With that environment active, from repository root:

```bash
python maze2d_chance/nominal_baseline.py --out results/my_nominal_run
python maze2d_chance/verify_nominal.py
python maze2d_chance/verify_portability.py
```

Use a fresh output directory for each run. Scripts locate code and assets from
their own file location; they do not require an editable install. Optional
`pip install --no-deps -e .` installs the local Diffuser package after preparing
the runtime. No training, dataset download or other checkout is needed for
inference. Fresh environment installation has not been tested in this task;
execution and relocation were checked in the existing documented runtime.

## Current evidence

The [nominal baseline report](maze2d_chance/results/nominal_local_20261104/REPORT.md)
uses the original Diffuser sampler and state-reference feedback controller for
800 steps, with no guidance, added noise, added obstacles, or retraining.
It reaches the goal in **9/10** fixed-seed cases, with **4/10** collision-free
goal-reaching cases. Every saved array matches the previous external-source
reproduction exactly. A relocated-copy test blocks all old-project/archive
access and reproduces a saved episode exactly.

`maze2d_chance/chance_guidance.py` contains the Gaussian moment propagation,
wall/goal probabilities, union sum, softplus energy and denoising gradient from
the earlier direct-action attempt. **It remains experimental:** it describes
direct actions, not the restored state-tracking controller, and is not enabled
by the nominal runner. No new budget-compliant risk result is claimed.

See [the experiment guide](maze2d_chance/README.md) for scope and verification.
