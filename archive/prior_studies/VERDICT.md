# Verdict: current original-plan pilot

## Simplest supported diffusion planner

Use a verified ordinary-feasible control diffusion prior, chance guidance during denoising, and independent confidence-bound validation with explicit failure when no candidate passes. Keep genuine sampled CVaR available as an alternative. Apply nominal constrained projection only when a feasibility repair is useful. Risk-conditioned training is an optional research variant: matched experiments do not yet justify its additional labeling pipeline as the default.

This recommendation is scoped to the tested known-model 2D system. It does not turn the local Gaussian surrogate into a valid nonlinear probability bound. A confirmed case predicts 9.94% risk while independent execution measures 11.58% [11.34%, 11.83%]; validation rejects it. The original stochastic execution requirement therefore needs an explicit acceptance mechanism.

## What went wrong in the first pilot

1. The nominal diffusion prior was insufficiently reliable, so safety layers also repaired generator failures.
2. A soft clearance optimizer was described as projection, obscuring what supplied its benefit.
3. Approximate CVaR loss/threshold handling changed the intended objective.
4. Conditional models had unmatched training and ambiguous missing-risk conditioning.
5. Surrogate feasibility, actual risk, and accepted-plan success were not cleanly separated.
6. Treating risk below Delta as a failure of budget control was too strong. The relevant unresolved test is cost improvement as the budget loosens.

The revision repairs those problems while preserving the control-sequence representation and denoising-time guidance in the original plan. Main nominal feasibility now exceeds 99% across three seeds. B2 improves acceptance from 91/96 to 96/96 at Delta=.10 under moderate noise. Projection contributes little after that repair; ordinary-data risk conditioning adds no acceptance after guidance in the matched pilot.

## How to improve the method next

1. Preserve the current healthy-prior gate, exact event/loss definition, independent acceptance, and explicit rejection semantics.
2. Address nonlinear distribution error directly. Test improved moment propagation or empirical calibration on held-out boundary-focused cases; a fixed Gaussian risk buffer has already failed in the stronger nonlinear setting.
3. Add explicit cost pressure and measure an accepted-plan cost/risk frontier against conventional stochastic optimization. Current risk-only guidance underuses the budget, and conventional cost optimization finds substantially shorter accepted moderate-noise routes.
4. Test whether a diffusion proposal buys measurable runtime, diversity, or nonconvex-solve reliability under matched risk and compute budgets. Current results do not establish that advantage.
5. Reconsider conditioning only if broader labeled support or adaptive correction produces a reproducible benefit beyond the matched nominal + guidance model.

## Larger benchmark and novelty

A modest expansion within nonlinear 2D is justified to test calibrated acceptance and cost tradeoffs across more layouts/seeds. Escalating to a manipulator or claiming a new risk-bounded diffusion framework is premature. Diffuser, Risk-Guided Diffusion, SafeDiffuser, constrained diffusion, classical chance-constrained planning, and CVaR optimization already provide the main ingredients. The strongest current finding is a controlled diagnosis of what works and what fails, including a concrete nonlinear false-safe surrogate case. A larger robotics contribution requires a substantive improvement in calibration or a demonstrated advantage over conventional planning.

See [RESULTS.md](RESULTS.md), [DESIGN.md](DESIGN.md), and [PLAN_STATUS.md](PLAN_STATUS.md) for the evidence and original-plan mapping.
