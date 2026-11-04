#!/usr/bin/env bash
# Run from the research workspace root. Outputs are regenerated in place.
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/home/kristoff/miniconda3/envs/diffuser/bin/python}"
for seed in 0 1 2; do
  "$PYTHON_BIN" -m risk_diffusion.revision.run_nominal --seed "$seed" --output "risk_diffusion/revision/outputs/nominal_seed${seed}"
done
"$PYTHON_BIN" -m risk_diffusion.revision.run_nominal --layout staggered --output risk_diffusion/revision/outputs/staggered_seed0
"$PYTHON_BIN" -m risk_diffusion.revision.run_nominal --strength .35 --output risk_diffusion/revision/outputs/nonlinear_seed0
"$PYTHON_BIN" -m risk_diffusion.revision.checks
"$PYTHON_BIN" -m risk_diffusion.revision.run_sweeps
"$PYTHON_BIN" -m risk_diffusion.revision.run_guidance
"$PYTHON_BIN" -m risk_diffusion.revision.run_projection
"$PYTHON_BIN" -m risk_diffusion.revision.conditional
"$PYTHON_BIN" -m risk_diffusion.revision.conditional --ordinary-only
"$PYTHON_BIN" -m risk_diffusion.revision.conventional --margin 1 --maxiter 80 --ftol 1e-7 --output risk_diffusion/revision/outputs/conventional
"$PYTHON_BIN" -m risk_diffusion.revision.conventional --margin .8 --maxiter 300 --ftol 1e-5 --output risk_diffusion/revision/outputs/conventional_margin08
"$PYTHON_BIN" -m risk_diffusion.revision.audit_boundary
"$PYTHON_BIN" -m risk_diffusion.revision.diversity
"$PYTHON_BIN" -m risk_diffusion.revision.report
"$PYTHON_BIN" -m risk_diffusion.revision.manifest
