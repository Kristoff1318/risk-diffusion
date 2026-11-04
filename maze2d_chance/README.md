# Maze2D Chance

## Current baseline

Start with the [nominal reproduction report](results/nominal_baseline_20261104/REPORT.md)
and [trajectories](results/nominal_baseline_20261104/trajectories.png).

The first priority is reproducing the older RB Diffuser's unguided planning
pipeline. `nominal_baseline.py` reuses its unchanged `Policy`, diffusion sampler,
EMA checkpoint, and normalization classes. It samples a 384-state position and
velocity reference once, then runs the original feedback controller for the
full 800-step native episode, including final-position tracking after the
reference ends. It does not execute the generated action sequence directly.

The ten predetermined environment/sampling seeds 0–9 reached the goal in 9/10
cases and finished in the goal in the same 9/10. Six episodes touched walls;
four were collision-free goal-reaching episodes. These are fixed-seed,
no-added-noise observations, not a measured stochastic risk guarantee.

`verification.json` confirms exact sample equality between the reused original
Policy and original sampler under the same random seed. Source/checkpoint
hashes and the protocol precede outcomes. Every sampled reference, generated
action sequence, executed trajectory, controller command, and per-step contact
flag is retained. Plots use the reference screenshot's orientation, complete
boundary, and equal physical proportions.

`replay_verification.json` independently confirms all ten recorded controller
command sequences, native trajectories, and active-contact flags exactly.
It also demonstrates the stale-contact-buffer bug on real saved episodes.

## Reproduce

Run from `/home/kristoff/Documents/risk_diffusion` with a fresh output directory:

```bash
OPENBLAS_NUM_THREADS=1 /home/kristoff/miniconda3/envs/diffuser/bin/python maze2d_chance/nominal_baseline.py --out results/nominal_reproduction
```

To verify the saved current run without generating new plans:

```bash
OPENBLAS_NUM_THREADS=1 /home/kristoff/miniconda3/envs/diffuser/bin/python maze2d_chance/verify_nominal.py
```

The existing CUDA environment and local pretrained checkpoint are required.
There is no training, dataset rebuild, download, guidance, added noise, added
obstacle, or replanning. The old repository
`/home/kristoff/Documents/rb-diffuser` is read-only. `normalization.npz` contains
the previously cached checkpoint-dataset extrema, used with the original
normalization classes.

## Scope and next question

The intended risk mechanism remains exactly the chance formulation and
diffusion guidance in slides 13–14 of [the deck](../main.pptx). No CVaR, arms,
projection, or risk-conditioned training is part of current work.

Restoring reference tracking does not itself resolve the slides' control-based
representation: the risk model and guidance gradient must describe the actual
executed controller/dynamics. That mapping must be explicit before another
risk comparison. Nominal goal success is also distinct from collision-free
goal success; neither can be substituted for whole-execution risk compliance.

## Historical attempts

- `archive/direct_action_20261104/`: the initial custom-sampler, direct-action,
  383-step experiment and its pilots, plots, tests, and reports. It is not the
  nominal reproduction. Its collision checker iterated the full MuJoCo contact
  buffer, including stale entries; its collision/risk figures require
  reevaluation and must not support conclusions about this restored baseline.
- `archive/nominal_contact_buffer_attempt/`: first reproduction with correct
  sampling/tracking but the same contact-buffer accounting bug. It has the
  same aggregate 9/10 goal and 6/10 contact counts as the corrected run, but
  overcounts contact durations. Preserved for transparency.
- `archive/nominal_baseline_20261104_setup_attempt/`: setup stopped before
  episode outcomes because the environment returned a tuple goal; repaired
  by converting the goal to a NumPy array.
- `slide_evidence/`: inspected equations extracted from the source PowerPoint.

Only the first `env.sim.data.ncon` contacts are active. The current reproduction
checks that active prefix, not all 100 allocated contact slots.

Two prior comparison images had already been locally deleted before cleanup;
their deletion is preserved. Historical move hashes are recorded in
`archive/direct_action_20261104/MOVE_MANIFEST.json`.
