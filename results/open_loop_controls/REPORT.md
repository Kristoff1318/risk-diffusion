# Slides 13–14 with controls played open loop

`open_loop_controls.py` runs the slides' formulation as written. The Diffuser sample's
**control** channels u_0..u_382 are executed blind in MuJoCo, with no tracking
controller. Risk (slide 13) is propagated through the free point mass. Guidance
(slide 14) pushes on the controls.

## Setup

- Same checkpoint, goal (7, 9), noise (velocity kicks of std 0.005 per tick),
  γ = 10 and guidance in the last 32 of 256 denoising steps as `compare_tracking.py`.
- Start positions are the original experiment's walls-only seeds 0–2, read from
  `results/chance_comparison/results.json`. The pilot uses the original seed-10 start.
  This environment (gym 0.23) draws different starts for the same seed than the
  original gym 0.18. A first run made that mistake; its outputs are kept locally
  but not committed.
- CPU, one thread, so Diffuser samples differ from the GPU samples in
  `chance_comparison`. To compare within this run, each unguided sample is also
  executed with the original tracking controller.
- Strength pilot on seed 10: nine values, from 2e-5 to 20. Selection: fewest
  zero-noise failures, then latest first wall touch, then smallest risk sum. Chose
  **0.02** (first touch at tick 338 vs 212 unguided). Values ≥ 2 wreck the plan
  (first touch at ticks 30–50).
- Each plan gets 256 validation and 256 independent audit runs. Failure = any wall
  contact (ticks 0..383) or ending ≥ 0.5 from the goal at tick 383.

## Results

| Seed | Tracking, same sample (audit) | Blind controls, unguided | Blind, guided 5% | Blind, guided 10% |
|---|---|---|---|---|
| 0 | 256/256 fail (wall grazes; goal reached) | 256/256, risk sum 102 | 246/256, sum 34.3 | 246/256, sum 34.3 |
| 1 | 256/256 fail (wall grazes; goal reached) | 256/256, sum 236 | 256/256, sum 211 | 256/256, sum 211 |
| 2 | 256/256 fail (wall grazes; goal reached) | 256/256, sum 270 | 256/256, sum 129 | 256/256, sum 129 |

No guided plan passes either budget.

- **Without noise, all three blind-control plans touch a wall,** at ticks 149, 95
  and 74. After the first touch the ball bounces and the remaining controls drive it
  in the wrong direction; most runs also miss the goal.
- **Spread at the end:** about 0.07–0.09 cell-widths for blind controls, against
  about 0.001 for tracking.
- **Guidance lowers the predicted risk sum** (102→34, 236→211, 270→129), but does
  not remove the wall touches.
- **The 5% and 10% plans are identical,** because every risk sum is far above 1.
- **On these CPU samples, tracking also grazes walls in all three cases.** On the
  original GPU samples, seed 1 was clean. Whether a plan grazes depends on the sample.

`comparison.png` shows every case. `results.json`, `protocol.json`, `pilot/` and
the per-plan `.npz` files hold all outcomes.
