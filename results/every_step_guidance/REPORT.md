# Guidance at every denoising step (slide 13: "at each diffusion step")

`every_step_guidance.py` changes one thing from earlier runs: guidance acts in all
256 denoising steps instead of the last 32. It runs in two setups:
- **tracking**: the planned states are followed by the original controller for 800 ticks.
- **blind**: the planned controls are played open loop for 383 ticks.

Everything else is unchanged: the original starts, velocity noise 0.005, γ = 10,
budgets 5% / 10%, and 256 validation + 256 audit runs. CPU, one thread per setup.

## Strength pilot (seed 10, not a test start)

Selection: fewest zero-noise failures, then latest first wall touch, then smallest risk sum.

- **Tracking: chose 0.02.** The guided plan had no wall touch and a predicted risk
  sum of 0.011; the unguided plan touched at tick 208 (sum 42). The weaker strengths
  did not help.
- **Blind: chose 0.02.** First wall touch moved from tick 212 to 343 (sum 72 → 18).
  0.02 was the largest strength tried in both setups.

## Test starts (audit failures out of 256; any wall touch or missing the goal fails)

| Seed | Setup | Unguided | Guided 5% | Guided 10% |
|---|---|---|---|---|
| 0 | tracking | 256, sum 52, touch t152 | 256, sum 42, touch t290 | 256, sum 45, touch t289 |
| 1 | tracking | 256, sum 149, touch t109 | 256, sum 73, touch t109 | 256, sum 63, touch t109 |
| 2 | tracking | 256, sum 73, touch t89 | 256, sum 44, touch t92 | 256, sum 44, touch t92 |
| 0 | blind | 256, sum 102, touch t149 | 256, sum 66, touch t234 | 256, sum 66, touch t234 |
| 1 | blind | 256, sum 236, touch t95 | 256, sum 130, touch t203 | 256, sum 130, touch t203 |
| 2 | blind | 256, sum 270, touch t74 | 256, sum 111, touch t99 | 256, sum 111, touch t99 |

No guided plan passes either budget. Guidance lowers every predicted risk sum
and often delays the first wall touch. Under tracking, every remaining failure
is a brief graze of a corner while the ball still reaches the goal. With a
position spread of about 0.001, a graze makes every noisy run fail.

Figures: `tracking.png`, `blind.png`. All outcomes are in `<setup>/results.json`,
`<setup>/pilot/` and the per-plan `.npz` files.
