# Verdict

**2026-09-26 reassessment:** The interpretation below is provisional and is superseded by the [trajectory inspection and methodological review](inspection_20260926/REASSESSMENT.md). Risk below Delta does not violate a risk bound; the conditional-training and refinement comparisons are confounded. The saved experiments establish improvement in this implementation, but do not yet select the best framework.

The simplest method supported by this pilot is **nominal control diffusion plus deterministic projection**, with the caveat that its low realized risk depends on this obstacle, goal width, and process noise. It has lower measured risk and batch planning time than the default chance guide, at a modest path-length cost. Chance guidance is the more appropriate method when an explicit stochastic budget matters, but its fixed-strength sampler was overconservative and did not track the requested Δ closely.

Do not claim that this is a risk-bounded planner yet. At requested budgets .05, .10, and .20, chance-guided realized risk remained near .02–.03. The one-map result establishes that its surrogate can produce safe plans under mild nonlinear stochastic dynamics, not that it uses a risk budget accurately or transfers to harder systems.

Sampled CVaR can reduce risk at higher strength, but its default configuration had worse risk and uses 800 differentiable rollout draws per sampled plan batch. Risk-conditioned training did not help, despite adding marginal and unsafe labeled routes. Projection plus chance guidance added cost without an observed safety benefit.

The next decisive experiment is a qualitatively different geometry and stronger nonlinear/noise regime, with a conventional chance-constrained or sample-based optimizer as a baseline. It should sweep λ jointly with Δ and test whether selected plans meet held-out upper confidence limits rather than point estimates alone. A larger robotics benchmark is premature until risk control survives that test. Closest methods include Diffuser's guidance/inpainting, SafeDiffuser, Risk-Guided Diffusion, Constrained Diffusers, and nonlinear risk-bounded trajectory optimization; see `LITERATURE.md` for citations.
