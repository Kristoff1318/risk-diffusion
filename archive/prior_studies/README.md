# Risk-bounded control diffusion

The current implementation and evidence follow the [original user plan](ORIGINAL_PLAN.txt), with a [stage-by-stage status](PLAN_STATUS.md).

Start with [RESULTS.md](RESULTS.md), [DESIGN.md](DESIGN.md), and [VERDICT.md](VERDICT.md). The corrected implementation is in `revision/`; original pilot code and outputs remain available, with its former reports archived in `legacy_20260922/`.

## Reproduce

From `/home/kristoff/Documents/experiments/research`:

```bash
bash risk_diffusion/revision/reproduce.sh
```

The script uses `/home/kristoff/miniconda3/envs/diffuser/bin/python`; override `PYTHON_BIN` for another compatible environment. Recorded environment: Python scientific stack with PyTorch 1.9.1+cu111, NumPy 1.21.6, SciPy 1.10.1, Matplotlib 3.3.4, CUDA on an RTX 4090. Training scripts currently require CUDA. Timings depend on hardware; stochastic training can vary across software/hardware versions. The script regenerates outputs in place.

For focused checks using the saved nominal checkpoint:

```bash
/home/kristoff/miniconda3/envs/diffuser/bin/python -m risk_diffusion.revision.checks
```

## Outputs

- `revision/outputs/summary/trajectories.png`: selected difficult examples, including rejected proposals.
- `revision/outputs/summary/{requested_risk,calibration,safety_cost,runtime}.png`: research plots.
- `revision/outputs/{guidance,projection,conditional,conditional_ordinary}/results.json`: per-plan measurements.
- `revision/outputs/boundary_audit.json`: independently confirmed Gaussian false-safe case.
- `revision/outputs/diversity.json`: repeated samples at identical conditions.
- `revision/outputs/manifest.json`: source/artifact hashes and runtime versions for this revision.

The public `revision.planner.plan` samples a fixed batch, applies independently evaluated confidence bounds with a batch correction, selects the lowest-cost accepted candidate, or returns an explicit failure. It does not claim robustness to unmodeled noise or continuous-time collision safety.
