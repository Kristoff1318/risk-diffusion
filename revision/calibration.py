"""Separate temporal union, tangent geometry, and Gaussian propagation error."""
import numpy as np
import torch
from .risk import chance

def decompose(u,start,goal,p,draws=4096,seed=60001):
    # Guidance moments only define tangent planes and predicted marginals.
    with torch.no_grad():
        a=chance(torch.tensor(u[None],dtype=torch.float64),torch.tensor(start[None],dtype=torch.float64),torch.tensor(goal[None],dtype=torch.float64),p)
    mu=a['mean'][0].numpy();cov=a['covariance'][0].numpy();pred=float(a['risk'][0])
    obs=np.array(p.obstacles);rng=np.random.default_rng(seed)
    x=start[None]+p.initial_sigma*rng.standard_normal((draws,2));exact=[];tangent=[];means=[];covs=[]
    for k in range(p.horizon+1):
        means.append(x.mean(0));covs.append(np.cov(x.T))
        if k<p.horizon:
            for cx,cy,r in obs:
                center=np.array([cx,cy]);n=(mu[k]-center);n=n/max(np.linalg.norm(n),1e-12)
                exact.append(np.linalg.norm(x-center,axis=1)<r)
                tangent.append((x-center)@n<r)
            d=p.strength*np.column_stack((np.sin(2*x[:,1]),np.sin(2*x[:,0])))
            x=x+p.dt*(u[k]+d)+p.process_sigma*rng.standard_normal((draws,2))
    for sign in (-1,1):
        for dim in range(2):
            event=sign*(x[:,dim]-goal[dim])>p.goal_half;exact.append(event);tangent.append(event)
    exact=np.stack(exact,1);tangent=np.stack(tangent,1)
    joint=exact.any(1).mean();sum_exact=exact.mean(0).sum();sum_tangent=tangent.mean(0).sum()
    return {'true_risk':float(joint),'mc_marginal_sum':float(sum_exact),'mc_tangent_sum':float(sum_tangent),
            'gaussian_sum':pred,'union_excess':float(sum_exact-joint),'geometry_excess':float(sum_tangent-sum_exact),
            'gaussian_error':float(pred-sum_tangent),'max_mean_error':float(np.linalg.norm(np.array(means)-mu,axis=1).max()),
            'terminal_covariance_error':float(np.linalg.norm(covs[-1]-cov[-1])),
            'min_covariance_eigenvalue':float(np.linalg.eigvalsh(cov).min())}
