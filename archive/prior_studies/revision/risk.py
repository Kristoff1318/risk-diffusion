"""Analytical Gaussian union surrogate and exact sampled upper-tail CVaR."""
import math
import torch
from .problem import Problem

def drift(x,p):
    return p.strength*torch.stack((torch.sin(2*x[...,1]),torch.sin(2*x[...,0])),-1)

def simulate(u,start,p,standard_noise=None):
    if standard_noise is None:
        standard_noise=torch.zeros((1,len(u),p.horizon+1,2),dtype=u.dtype,device=u.device)
    x=start[None]+p.initial_sigma*standard_noise[:,:,0];xs=[x]
    for k in range(p.horizon):
        x=x+p.dt*(u[None,:,k]+drift(x,p))+p.process_sigma*standard_noise[:,:,k+1];xs.append(x)
    return torch.stack(xs,2)

def hard_loss(states,goal,p):
    obs=torch.tensor(p.obstacles,dtype=states.dtype,device=states.device)
    violations=obs[:,2]-(states[:,:,:-1,None,:]-obs[None,None,None,:,:2]).norm(dim=-1)
    terminal=(states[:,:,-1]-goal[None]).abs()-p.goal_half
    return torch.cat((violations.flatten(2),terminal),-1).max(-1).values

def sampled_cvar(loss,delta):
    """Exact fractional empirical upper-tail mean = min_eta RU objective.

    Sorting supplies valid piecewise gradients. No smoothing or detached
    quantile approximation. Supports noninteger K*delta without ceil bias.
    """
    if not 0<delta<=1:raise ValueError('delta must be in (0,1]')
    values=loss.sort(dim=0,descending=True).values
    mass=delta*len(values);count=int(math.floor(mass));fraction=mass-count
    total=values[:count].sum(0)
    if fraction>1e-10:total=total+fraction*values[count]
    return total/mass

def mean_covariance(u,start,p):
    b=len(u);eye=torch.eye(2,dtype=u.dtype,device=u.device).expand(b,-1,-1)
    mu=start;cov=p.initial_sigma**2*eye;means=[mu];covs=[cov]
    for k in range(p.horizon):
        cross=2*p.dt*p.strength*torch.cos(2*mu)
        one=torch.ones_like(mu[:,0])
        A=torch.stack((torch.stack((one,cross[:,1]),-1),torch.stack((cross[:,0],one),-1)),-2)
        cov=A@cov@A.transpose(1,2)+p.process_sigma**2*eye
        mu=mu+p.dt*(u[:,k]+drift(mu,p));means.append(mu);covs.append(cov)
    return torch.stack(means,1),torch.stack(covs,1)

def chance(u,start,goal,p):
    mu,cov=mean_covariance(u,start,p)
    obs=torch.tensor(p.obstacles,dtype=u.dtype,device=u.device)
    diff=mu[:,:-1,None,:]-obs[None,None,:,:2];dist=diff.norm(dim=-1).clamp_min(1e-7)
    normal=diff/dist[...,None]
    var=torch.einsum('bhoi,bhij,bhoj->bho',normal,cov[:,:-1],normal).clamp_min(1e-12)
    obstacle=.5*torch.erfc((dist-obs[None,None,:,2])/torch.sqrt(2*var))
    sigma=torch.diagonal(cov[:,-1],dim1=-2,dim2=-1).clamp_min(1e-12).sqrt()
    goal_parts=torch.stack([.5*torch.erfc((p.goal_half-sign*(mu[:,-1]-goal))/(math.sqrt(2)*sigma)) for sign in (-1,1)],-1)
    return {'risk':obstacle.sum((1,2))+goal_parts.sum((1,2)), 'obstacle':obstacle,'goal':goal_parts,'mean':mu,'covariance':cov}

def nominal_cost(u,start,p):
    x=simulate(u,start,p)[0];length=(x[:,1:]-x[:,:-1]).norm(dim=-1).sum(1)
    effort=.02*p.dt*u.square().sum((1,2));rough=.01*(u[:,1:]-u[:,:-1]).square().sum((1,2))
    return length+effort+rough

def energy(u,start,goal,p,kind,delta,noise=None,gamma=30,cost_weight=.0,penalty='softplus'):
    if kind=='chance':
        residual=chance(u,start,goal,p)['risk']-delta
        risk_energy=torch.nn.functional.softplus(gamma*residual)/gamma if penalty=='softplus' else residual.clamp_min(0).square()
    elif kind=='cvar':
        if noise is None:raise ValueError('CVaR requires explicit fixed rollout noise')
        risk_energy=sampled_cvar(hard_loss(simulate(u,start,p,noise),goal,p),delta)
    elif kind=='deterministic':
        risk_energy=hard_loss(simulate(u,start,p),goal,p)[0].clamp_min(0).square()
    else:raise ValueError(kind)
    return risk_energy+cost_weight*nominal_cost(u,start,p)
