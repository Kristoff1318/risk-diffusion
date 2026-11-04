# Superseded integration experiment

These files are preserved unchanged from the first Maze2D attempt. They used
a custom sampler and direct generated-action execution for 383 steps, rather
than the original Diffuser state-reference controller and 800-step episode.
The old collision checker also inspected all allocated MuJoCo contact slots,
including stale entries beyond `env.sim.data.ncon`.

Consequently, the old 97.39% nominal failure headline is not a nominal Diffuser
reproduction result, and its collision/risk statistics require reevaluation.
The current nominal reproduction lives at
`../../results/nominal_baseline_20261104/REPORT.md` (relative to this directory,
use the active Maze2D README for navigation).

This archive retains source, outputs, pilots, and reports without rewriting
their historical claims. MOVE_MANIFEST.json verifies byte preservation for
the relocated existing files. Two comparison images already deleted locally
before cleanup remain absent. Cached normalization extrema were copied here
to retain the original script-relative dependency. Historical scripts may
still contain old paths; do not run them as current experiment instructions.
