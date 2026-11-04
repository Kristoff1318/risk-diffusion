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
- `chance_guidance.py`: experimental direct-action Gaussian covariance,
  native box/goal linearization, CDF event probabilities, union sum, literal
  softplus energy, and gradient correction during denoising. It is retained
  locally for review and further development; nominal execution does not use it.
- `verify_nominal.py`: independent saved-command/controller replay and active
  MuJoCo contact checks. Only `[:env.sim.data.ncon]` contact entries are active.
- `verify_portability.py`: copies the active repo to a temporary location,
  excludes archives, blocks access to the old project, and checks exact
  reference/action/execution/contact reproduction. Also checks a nonzero
  direct-action chance-energy derivative against finite differences.
- `../src/diffuser/`: locally maintained inference subset. Nine original
  source files are byte-identical to the previously used implementation;
  package initializers expose only the needed modules.

## Current baseline results

See [report](results/nominal_local_20261104/REPORT.md) and
[trajectories](results/nominal_local_20261104/trajectories.png).

Ten fixed environment/sampling seeds 0–9 reached and finished in the goal in
9/10 cases. Six episodes touched walls; four were collision-free goal-reaching
episodes. These are deterministic nominal checks, not a stochastic risk bound.
The 384-state reference is sampled once; the controller tracks its endpoint
with zero target velocity for the rest of the 800-step episode.

The local run reproduces every saved array from all ten previous baseline
episodes exactly (`localization_parity.json`). Its provenance uses portable
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
to `results/nominal_local_20261104`; select another with `--run results/NAME`.
Assets are tracked locally, including the unchanged approximately 29 MB
checkpoint. No absolute interpreter or old-project path is required by source.

## Research scope

The required mechanism is slides 13–14 of [the deck](../main.pptx): mean rollout,
local dynamics linearization, covariance propagation, constraint linearization,
Gaussian violation probabilities, union sum, softplus energy, and autodiff
guidance during denoising. No CVaR, arms, projection or risk-conditioned training.

The direct-action chance model is **not** yet qualified for the nominal
reference-tracking controller. That dynamics/representation mapping must be
explicit before another guided comparison. Its successful finite-difference
check validates a mathematical derivative, not obstacle safety or risk-budget
compliance. The nominal runner intentionally does not invoke it.

## Local history

`archive/` here and `../archive/` hold optional historical attempts on this
computer. Both are ignored and removed from the Git index. Nothing imports them.
Old commits retain their history; removing index entries does not rewrite Git
history. The former 97.39% result used a different execution pipeline and a
faulty contact-buffer check and does not describe the current nominal baseline.
