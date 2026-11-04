# Maze2D Risk Diffusion

This is the standalone Maze2D project. Its code lives directly in this folder;
there is no nested `maze2d_chance` project. Diffuser source and the required
checkpoint/configuration/normalization are local. No other project folder is
needed at runtime.

## Visual comparison

[Compact side-by-side comparison](results/chance_comparison/comparison_overview.png)
shows nominal versus 5% and 10% chance guidance on the same start, with 32
already-audited noisy executions per panel. Blue paths are safe, red paths fail.
It shows the successful wall-only risk-repair case and its added-obstacle case.
Paths show the agent center; collision checks include its physical radius 0.1.

- [All three wall-only starts](results/chance_comparison/walls.png)
- [All three starts with added obstacles](results/chance_comparison/added_obstacles.png)
- [Full report and all failures](results/chance_comparison/REPORT.md)
- [Nominal baseline trajectories](results/nominal_baseline/trajectories.png)

The comparison completed 18 proposals and 9,216 native stochastic executions.
In one wall-only case, nominal failed in 34/256 independent audit executions;
both guided versions had 0/256 and passed the 5%/10% validation budgets.
The other two wall cases remained unsafe. None of the added-obstacle cases
passed. All cases remain in the report; there is no general safety claim.

## Organization

| Location | Purpose |
|---|---|
| `chance_guidance.py` | Slides 13–14: Gaussian constraints, risk sum, softplus energy, autodiff denoising guidance |
| `tracking_dynamics.py` | Executed mean/covariance through the original feedback controller |
| `nominal_baseline.py` | Original unguided sampler and 800-step tracking execution |
| `compare_tracking.py` | Frozen paired guided comparison; existing outcomes protected |
| `evaluate_tracking.py` | Independent native MuJoCo noise, collision/goal checks, confidence bounds |
| `environment.py` | Maze2D Large v1 and optional physical circular obstacles |
| `runtime.py` | Repository-local model/source loading |
| `qualify_tracking.py`, `verify_*.py` | Dynamics, covariance, gradient, replay and portability checks |
| `report_comparison.py`, `visualize_comparison.py` | Figures and tables from saved outcomes/replayed audit draws |
| `pilot_tracking.py` | Disclosed guidance-strength pilot on separate seed 10 |
| `src/diffuser/` | Original local Diffuser inference modules |
| `assets/maze2d-large-v1/` | Required local checkpoint, configuration and scaling data |
| `results/` | One nominal baseline and one guided comparison |
| `main.pptx`, `slide_evidence/` | Slide specification and inspected equations |
| `third_party/diffuser/` | MIT license, provenance and copied-source hashes |

Historical material is outside the repository at
`/home/kristoff/Documents/risk_diffusion_storage/`, consolidated into
`prior_studies/` and `maze2d_attempts/`. It is optional local history and is
neither tracked nor imported. Required model data stays inside this repo.

## Slide-to-code mapping

| Slide step | Implementation |
|---|---|
| 13, 1–3: mean, dynamics Jacobian, covariance | `tracking_dynamics.propagate_tracking` |
| 13, 4–6: constraints and Gaussian probabilities | Wall/circle/goal functions in `chance_guidance.py` |
| 13, 7–8: union sum and budget | `TrackingChanceModel.risk_reference` |
| 14, 9–10: residual and softplus | `TrackingChanceModel.energy` |
| 14, 11–13: energy gradient and score correction | `sample_guided` |

The existing checkpoint generates reference states. The controller uses
`u = p_ref - p + (v_ref - v)`, with native clipping. Mean and covariance are
propagated through this actual feedback system, rather than treating the
reference as the executed path. This explicitly adapts the slides' control
representation to the pretrained state-reference diffusion model.

Guidance differentiates the energy through the denoiser's clean reference,
normalization, controller and dynamics, in the final 32 of 256 denoising stages.
Walls are always included. The added-obstacle scene contains physical MuJoCo
cylinders of radius 0.35 at (3,4), (5,8), (7,5), in agent coordinates.
The native validator checks the complete 800-step execution with independent
Gaussian velocity perturbations of standard deviation 0.005 after each step.
The surrogate is locally approximate; it is not a universal risk certificate.
There is no CVaR, projection, extra clearance objective, or new training.

## Normalization data

`assets/maze2d-large-v1/normalization.npz` is a small NumPy archive with four
arrays: observation/action minima and maxima from the checkpoint's training
data. It scales positions, velocities and actions to approximately [-1,1] and
back. It is required model data, not a second baseline or uncertainty model.
The [asset README](assets/maze2d-large-v1/README.md) shows how to inspect it.

## Run and verify

The tested runtime is Python 3.8, PyTorch 1.9.1/CUDA 11.1, D4RL and MuJoCo 2.0.
`environment.yml` records dependencies. Native MuJoCo binaries/system libraries
and a compatible GPU are runtime prerequisites. Fresh installation was not
performed; execution and relocation were tested in the existing runtime.

From this project root, in that environment:

```bash
python nominal_baseline.py --out results/my_nominal_run
python verify_nominal.py
python verify_comparison.py
python qualify_tracking.py
python verify_portability.py
python report_comparison.py
python visualize_comparison.py
```

The comparison/pilot refuse to overwrite existing protocols. The current
reports preserve the exact pre-flattening source snapshots and source hashes
beside their results. Structural relocation does not change the saved trials.
The compact visualization replays the first 32 existing audit draws; its flags
match the saved audit exactly, and it supplies no new independent evidence.

The nominal baseline is one method and one canonical result folder,
`results/nominal_baseline/`: 9/10 goal-reaching and 4/10 collision-free cases.
Those deterministic checks are separate from the noisy guided comparison.
