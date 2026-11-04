# Maze2D Chance

The active source and assets are entirely repository-local. No script depends
on an RB Diffuser checkout, historical archive, or cached dataset download.
Installed Python/D4RL/MuJoCo runtime dependencies remain necessary; see the
root environment specification.

## Source

- `nominal_baseline.py`: original state-reference feedback execution for a
  full 800-step native episode. Calls the unchanged local Diffuser Policy and
  sampler; no custom nominal sampler, guidance, added noise or added obstacles.
- `runtime.py`: verifies and loads `../assets/maze2d-large-v1/`; imports only
  `../src/diffuser/`. Configuration is plain JSON, not external pickle configs.
- `chance_guidance.py`: the slide equations, split into readable functions
  with slide numbers and step numbers beside the matching code. It currently
  rolls out generated actions; it does not take sampled reference states as
  its mean trajectory and is not connected to the nominal tracking controller.
- `verify_nominal.py`: independent saved-command/controller replay and active
  MuJoCo contact checks. Only `[:env.sim.data.ncon]` contact entries are active.
- `verify_portability.py`: copies the active repo to a temporary location,
  excludes archives, blocks access to the old project, and checks exact
  reference/action/execution/contact reproduction. Also checks a nonzero
  direct-action chance-energy derivative against finite differences.
- `../src/diffuser/`: locally maintained inference subset. Nine original
  source files are byte-identical to the previously used implementation;
  package initializers expose only the needed modules.

## One nominal baseline

`nominal_baseline.py` is the runner; `results/nominal_baseline/` holds its
recorded outputs. There is only one nominal method. The previous name
"nominal local" meant the same run after copying dependencies into this repo;
it was not a different algorithm. Older named runs are kept only in the
ignored local archive.

## Current baseline results

See [report](results/nominal_baseline/REPORT.md) and
[trajectories](results/nominal_baseline/trajectories.png).

Ten fixed environment/sampling seeds 0–9 reached and finished in the goal in
9/10 cases. Six episodes touched walls; four were collision-free goal-reaching
episodes. These are deterministic nominal checks, not a stochastic risk bound.
The 384-state reference is sampled once; the controller tracks its endpoint
with zero target velocity for the rest of the 800-step episode.

The local run reproduces every saved array from all ten previous baseline
episodes exactly (`cleanup_verification.json`). Its provenance uses portable
repository-relative paths. `verification.json` checks original sampler/Policy
parity; `replay_verification.json` checks all ten executions;
`portability_verification.json` checks a relocated run with old paths blocked.

## Run and verify

Activate the documented Python/CUDA/MuJoCo environment, then run from repo root:

```bash
python maze2d_chance/nominal_baseline.py --out results/my_nominal_run
python maze2d_chance/verify_nominal.py
python maze2d_chance/verify_portability.py
```

The nominal runner refuses to overwrite an existing run. The verifier defaults
to `results/nominal_baseline`; select another with `--run results/NAME`.
Assets are tracked locally, including the unchanged approximately 29 MB
checkpoint. No absolute interpreter or old-project path is required by source.

## Research scope

The required mechanism is slides 13–14 of [the deck](../main.pptx): mean rollout,
local dynamics linearization, covariance propagation, constraint linearization,
Gaussian violation probabilities, union sum, softplus energy, and autodiff
guidance during denoising. No CVaR, arms, projection or risk-conditioned training.

The guide's integration status is **not yet connected to the nominal
state-tracking controller**. It currently computes states by rolling out
sampled actions. The nominal planner instead tracks the sampled position and
velocity reference with feedback. These are different execution models.
Calling the guide "experimental" referred to this unfinished integration and
the absence of a demonstrated risk bound; it was not a separate algorithm.
This cleanup preserves the existing equations and makes that distinction
explicit rather than silently changing the model.

### Where the slides appear in code

| Slide and step | Function or method |
|---|---|
| 13, 1–3: mean, Jacobian, covariance | `propagate_mean_and_covariance` |
| 13, 4–5: wall constraint and linearization | `wall_violation_probabilities` |
| 13, 6: Gaussian violation probability | `gaussian_violation_probability` |
| 13, 4–6: goal constraint | `goal_violation_probability` |
| 13, 7–8: sum risk across events | `DirectActionChanceModel.risk` |
| 14, 9–10: residual and softplus energy | `DirectActionChanceModel.energy` |
| 14, 11–13: autodiff and score correction | `sample_direct_actions` |

The native Maze2D dynamics, walls, agent radius, normalization, and circular
goal are labeled as environment adapters. The retained terminal-wall check
extends the slide's obstacle indexing to include the final state; this
difference is called out at the risk sum. There is no hidden CVaR or extra
safety optimizer.

### What normalization.npz is

`../assets/maze2d-large-v1/normalization.npz` stores four arrays: the original
training-data minima and maxima for position/velocity and actions. Diffuser
uses them to scale physical values to approximately [-1, 1] and back. They are
necessary checkpoint scaling data, not a new experiment or uncertainty model.
See the asset README for the array names and a short inspection command.

## Local history

`archive/` here and `../archive/` hold optional historical attempts on this
computer. Both are ignored and removed from the Git index. Nothing imports them.
Old commits retain their history; removing index entries does not rewrite Git
history. The former 97.39% result used a different execution pipeline and a
faulty contact-buffer check and does not describe the current nominal baseline.
