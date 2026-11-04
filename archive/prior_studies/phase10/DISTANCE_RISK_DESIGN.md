# Covariance-based distance-risk energy: design, not a claimed certificate

The project question is risk-bounded planning for stochastic nonlinear/full-arm systems. The declared joint tracking error is a special tractable case: its dynamics are linear Gaussian, while full-arm FK, world/self distances, pose norms and trajectory-union events are nonlinear. Gaussian joint covariance does not make those distances or union events Gaussian.

## Exact uncertainty that can be reused

Let dt=0.05, K=2, a=1-dt*K=0.9, c_k=(1-a^k)/K. With independent physical initial error e0, persistent bias b and step noises eta:

e_k = a^k e0 + c_k b + dt sum_{i=0}^{k-1} a^(k-1-i) eta_i.

The exact cross covariance is

C_kl = a^(k+l) Sigma0 + c_k c_l Sigmab
       + dt^2 a^|k-l| (1-a^(2 min(k,l))) / (1-a^2) Sigmaeta.

For the doubled model, Sigma0=(.004)^2 I, Sigmab=(.002)^2 I and Sigmaeta=(.010)^2 I. The same formulas already passed augmented-state covariance qualification in phase6; no model fitting is needed. Dense-grid error at fraction r of step k is (1-r)e_k+r e_{k+1}, so its exact covariance uses C_kk, C_{k,k+1}, C_{k+1,k+1}. Do not use a constant terminal variance at early times.

Command error is -K e_k; its covariance is K^2 C_kk. Executed velocity error is -K e_k+b+eta_k, with covariance K^2 C_kk+Sigmab+Sigmaeta-2K c_k Sigmab. The bias cross term matters. Joint-position and command/executed-velocity scalar inequalities have exact Gaussian marginals in this model. Their time-union probability is not a product of marginals: errors are strongly correlated across time.

## Proposed local energy

For each native sphere-world or configured self pair, retain its individual signed clearance g_i(q), not only the nearest pair. At nominal q_bar,

mu_i approximately g_i(q_bar), s_i^2 = J_i C(t,t) J_i^T,
p_i approximately Phi(-mu_i/s_i), J_i = derivative of g_i with respect to q.

World-pair derivatives use the center's kinematic Jacobian projected on the separation direction. Self-pair derivatives use the difference of the two center Jacobians. Compute analytic differentiable point Jacobians once per time/robot model where possible; do not do 32 full physical FK passes by default. Sufficiently near zero separation, nonsmoothness and unstable normals require a declared safe handling rule, not invented gradients.

An initial prototype can minimize unchanged J plus a penalized excess of R_hat=sum_i p_i over the risk budget, retaining all geometry, joint, velocity and terminal-pose component terms. Use exact scalar Gaussian marginals for affine joint/velocity inequalities. For terminal position/rotation-log norms, a componentwise box inside the tolerance ball gives a conservative union surrogate **for the local linearized pose model**; it is not exact for nonlinear pose. Use the same risk budget and hard independent validator as current phases.

Summing marginal event probabilities is a valid upper bound for the **linearized events** without assuming independence. It is not a valid upper bound for actual nonlinear distances unless linearization remainders are controlled. Native collision models plus 2,001 time states create millions of correlated events: naive equal risk allocation or summation may be far too conservative. The first pilot should expose this failure, not silently drop pairs, switch to a tuned top-k list, or claim calibrated union risk. Report the surrogate and each component alongside true nonlinear sample estimates.

A safety-oriented extension would require a verified Taylor remainder or Lipschitz bound over a declared Gaussian confidence region, plus its outside-region probability and union allocation. That extra conservatism/cost is not established here and is not part of the proposed first confirmation. Neither Hessian-free first-order moments nor a mean-distance Gaussian approximation supplies such a certificate automatically.

## What remains approximate or untested

- FK transforms Gaussian joint errors into non-Gaussian center distributions; Euclidean norms and minima add skew/nonsmoothness.
- Cross-time and cross-pair dependence is strong. Assuming independent collision events would understate or overstate trajectory risk.
- Derivatives of s_i with respect to controls may require distance Hessians; detaching covariance/Jacobians changes the optimizer and must be disclosed.
- Boundary proximity, self-collision switching and terminal orientation can invalidate local approximations. No input-family or geometry tuning is authorized to rescue them.
- Faster expected geometry reuse is a hypothesis. Pair/time enumeration, automatic differentiation and higher derivatives could erase the advantage.
- Exact covariance depends on this ideal tracking model. Nonlinear dynamics, uncertain observation, saturation or non-Gaussian disturbances would require a different propagation method.

Gaussian belief-space chance surrogates are established prior art, not a new idea by themselves. For example, [Chen et al., Constrained Iterative LQG for Real-Time Chance-Constrained Gaussian Belief Space Planning](https://group.iiis.tsinghua.edu.cn/~isrlab/publication/2021/iros2021jianyu/IROS2021Jianyu.pdf) uses Gaussian belief approximations and transformed chance constraints for nonlinear planning. A contribution here would require verified whole-arm risk/cost/latency improvement, with explicit approximation error and full event accounting, rather than naming covariance propagation as novel.

Recommendation: qualify exact covariance and local derivatives on the established saved fixtures, profile a fixed prototype, and test its independently audited behavior on four untouched tasks. Keep sampled CVaR's measured 8.3 s correction as the comparator; do not promise that the new energy will outperform it.

