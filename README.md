# Risk Diffusion: Maze2D Chance

Current work tests the chance-guidance formulation in slides 13–14 of
[main.pptx](main.pptx) on native D4RL `maze2d-large-v1`.

**First priority: reproduce the original nominal Diffuser baseline before
interpreting risk-guidance comparisons.**

- [Active experiment and instructions](maze2d_chance/README.md)
- [Nominal reproduction report](maze2d_chance/results/nominal_baseline_20261104/REPORT.md)
- [Nominal trajectories](maze2d_chance/results/nominal_baseline_20261104/trajectories.png)
- [Archived studies and preservation details](archive/README.md)

The current baseline uses the original pretrained model, original Diffuser
sampler, and original feedback controller tracking sampled positions and
velocities for 800 steps. It has no risk guidance, added execution noise,
added obstacles, or new training.

Ten fixed seed cases reached the goal in 9/10 episodes; 4/10 reached it without
wall contact. Goal reaching and obstacle avoidance are reported separately.
This is a deterministic nominal reproduction, not evidence of a stochastic
risk bound. Chance guidance remains to be evaluated against this restored
execution pipeline while explicitly maintaining fidelity to the slides.

Work is on branch `codex/maze2d-chance-baseline` in this Documents folder.
Earlier reports, arms, inspections, and numbered phases are preserved under
`archive/prior_studies/`. The earlier direct-action Maze2D experiment is under
`maze2d_chance/archive/`; its 97.39% figure does not describe this baseline.
