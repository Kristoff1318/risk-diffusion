"""Small control-space diffusion and nonlinear stochastic navigation testbed."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

H = 20
DT = 0.20
RADIUS = 0.43
NOISE = 0.025
GOAL_HALF = 0.30


def drift(x):
    return torch.stack((0.12 * torch.sin(2*x[..., 1]), 0.12 * torch.sin(2*x[..., 0])), -1)


def rollout(u, start, noise=None):
    """u: [B,H,2]; noise: [K,B,H,2] or None. Returns [K,B,H+1,2]."""
    if noise is None:
        noise = torch.zeros((1,) + tuple(u.shape), device=u.device, dtype=u.dtype)
    x = start[None].expand(noise.shape[0], -1, -1)
    states = [x]
    for k in range(u.shape[1]):
        x = x + DT * (u[None, :, k] + drift(x)) + noise[:, :, k]
        states.append(x)
    return torch.stack(states, 2)


def data(n=2500, seed=0, height_range=(.56,1.18)):
    rng = np.random.default_rng(seed)
    starts = np.column_stack((-1.5*np.ones(n), rng.uniform(-.18,.18,n))).astype('float32')
    goals = np.column_stack((1.5*np.ones(n), rng.uniform(-.18,.18,n))).astype('float32')
    paths = np.zeros((n,H+1,2),dtype='float32')
    for i in range(n):
        side = 1.0
        height = rng.uniform(*height_range)
        knots = np.array([starts[i],[-.62,side*height],[.62,side*height],goals[i]],dtype='float32')
        # Smooth fixed-time path; risk levels vary mainly with clearance.
        for k,t in enumerate(np.linspace(0,3,H+1)):
            j=min(int(t),2); a=t-j; a=a*a*(3-2*a)
            paths[i,k]=(1-a)*knots[j]+a*knots[j+1]
    p=torch.tensor(paths)
    u=((p[:,1:]-p[:,:-1])/DT-drift(p[:,:-1])).numpy()
    return starts,goals,u


class Denoiser(nn.Module):
    def __init__(self, conditional=False, width=192):
        super().__init__(); self.conditional=conditional
        self.net=nn.Sequential(nn.Linear(H*2+5+(1 if conditional else 0),width),nn.SiLU(),
                               nn.Linear(width,width),nn.SiLU(),nn.Linear(width,H*2))

    def forward(self,z,t,start,goal,risk=None):
        b=z.shape[0]; extras=[z.flatten(1),start,goal,t.reshape(b,1)]
        if self.conditional:
            extras.append(risk.reshape(b,1) if risk is not None else torch.zeros((b,1),device=z.device))
        return self.net(torch.cat(extras,1)).reshape_as(z)


def schedule(T=40):
    betas=torch.linspace(.0001,.15,T)
    return torch.cumprod(1-betas,0)


def train(starts,goals,controls,conditional=False,risks=None,epochs=30,seed=0,device='cpu'):
    torch.manual_seed(seed)
    u=torch.tensor(controls,device=device); s=torch.tensor(starts,device=device); g=torch.tensor(goals,device=device)
    mean=u.mean((0,1),keepdim=True); scale=u.std((0,1),keepdim=True).clamp_min(.25)
    y=(u-mean)/scale; model=Denoiser(conditional).to(device); opt=torch.optim.Adam(model.parameters(),lr=.001)
    a=schedule().to(device); risk=torch.tensor(risks,device=device,dtype=torch.float32) if risks is not None else None
    losses=[]
    for ep in range(epochs):
        perm=torch.randperm(len(y),device=device)
        for ix in perm.split(128):
            t=torch.randint(len(a),(len(ix),),device=device); eps=torch.randn_like(y[ix]); at=a[t,None,None]
            z=at.sqrt()*y[ix]+(1-at).sqrt()*eps
            r=risk[ix] if risk is not None else None
            if conditional and r is not None:
                r=r* (torch.rand(len(ix),device=device)>.15).float() # classifier-free dropout
            loss=F.mse_loss(model(z,t.float()/len(a),s[ix],g[ix],r),eps)
            opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    return model,mean.detach(),scale.detach(),losses


def chance_risk(u,start,goal,return_parts=False):
    """Local Gaussian linearization and Boole sum for circle and goal box."""
    b=u.shape[0]; mu=start; cov=torch.zeros((b,2,2),device=u.device,dtype=u.dtype)
    eye=torch.eye(2,device=u.device,dtype=u.dtype).expand(b,-1,-1)
    vals=[]
    for k in range(H):
        a12=DT*.24*torch.cos(2*mu[:,1]); a21=DT*.24*torch.cos(2*mu[:,0])
        A=torch.stack((torch.stack((torch.ones_like(a12),a12),-1),
                       torch.stack((a21,torch.ones_like(a21)),-1)),-2)
        cov=A@cov@A.transpose(1,2)+(NOISE**2)*eye
        mu=mu+DT*(u[:,k]+drift(mu))
        dist=mu.norm(dim=-1).clamp_min(1e-5)
        normal=mu/dist[:,None]
        sigma=torch.sqrt(torch.einsum('bi,bij,bj->b',normal,cov,normal).clamp_min(1e-9))
        vals.append(torch.erf((RADIUS-dist)/(sigma*math.sqrt(2)))/2+.5)
    goal_parts=[]
    for j in range(2):
        sigma=cov[:,j,j].clamp_min(1e-9).sqrt()
        for sign in (-1,1):
            z=(sign*(mu[:,j]-goal[:,j])-GOAL_HALF)/sigma
            goal_parts.append(.5*(1+torch.erf(z/math.sqrt(2))))
    parts=torch.stack(vals+goal_parts,1)
    risk=parts.sum(1)
    return (risk,parts) if return_parts else risk


def trajectory_loss(states,goal,beta=18):
    # Signed violation: positive is collision or missed goal box.
    obs=RADIUS-states[:,:,1:,:].norm(dim=-1)
    err=(states[:,:,-1,:]-goal[None]).abs()-GOAL_HALF
    raw=torch.cat((obs,err),-1)
    return torch.logsumexp(beta*raw,dim=-1)/beta


def cvar(u,start,goal,delta=.1,K=64,beta=18,noise=None):
    if noise is None: noise=torch.randn((K,)+tuple(u.shape),device=u.device)*NOISE
    losses=trajectory_loss(rollout(u,start,noise),goal,beta)
    # Differentiable empirical upper-tail CVaR; quantile threshold is detached.
    eta=torch.quantile(losses.detach(),1-delta,dim=0)
    return eta+F.softplus(beta*(losses-eta[None])).mean(0)/(beta*delta)


def deterministic(u,start,goal):
    states=rollout(u,start)[0]
    obs=RADIUS-states[:,1:].norm(dim=-1)
    end=(states[:,-1]-goal).abs()-GOAL_HALF
    return F.softplus(15*torch.cat((obs,end),1)).mean(1)/15


def energy(u,start,goal,kind,delta=.1,K=64,gamma=15):
    if kind=='chance': return F.softplus(gamma*(chance_risk(u,start,goal)-delta))/gamma
    if kind=='cvar': return cvar(u,start,goal,delta,K)
    if kind=='sdf': return deterministic(u,start,goal)
    raise ValueError(kind)


def project(u,start,goal,kind='sdf',delta=.1,steps=15,lr=.12):
    z=u.detach().clone().requires_grad_(True); opt=torch.optim.Adam([z],lr=lr)
    initial=u.detach()
    for _ in range(steps):
        obj=(energy(z,start,goal,kind,delta)+.015*((z-initial)**2).mean((1,2))).sum()
        opt.zero_grad(); obj.backward(); opt.step()
        with torch.no_grad(): z.clamp_(-2.5,2.5)
    return z.detach()


def terminal_correct(u,start,goal,iterations=5):
    """Common goal-conditioning correction for every sampler variant."""
    with torch.no_grad():
        for _ in range(iterations):
            end=rollout(u,start)[0,:,-1]
            u=u+(goal-end)[:,None,:]/(H*DT)
    return u


def sample(model,mean,scale,start,goal,kind=None,delta=.1,lam=0.,steps=30,projection=False,
           conditional_risk=None,seed=0,K=64,projection_steps=12,gamma=15):
    device=mean.device; torch.manual_seed(seed); b=len(start); a=schedule().to(device)
    start=torch.as_tensor(start,device=device,dtype=torch.float32); goal=torch.as_tensor(goal,device=device,dtype=torch.float32)
    z=torch.randn((b,H,2),device=device); ids=np.linspace(len(a)-1,0,steps,dtype=int)
    t0=time.perf_counter(); grad_norms=[]; corrections=0
    model.eval()
    for i,ti in enumerate(ids):
        with torch.no_grad():
            tt=torch.full((b,),float(ti)/len(a),device=device)
            r=None if conditional_risk is None else torch.full((b,),conditional_risk,device=device)
            eps=model(z,tt,start,goal,r)
            x0=(z-(1-a[ti]).sqrt()*eps)/a[ti].sqrt()
            x0=(terminal_correct(x0*scale+mean,start,goal)-mean)/scale
        if kind and lam:
            with torch.enable_grad():
                xx=x0.detach().requires_grad_(True)
                e=energy(xx*scale+mean,start,goal,kind,delta,K,gamma).sum()
                grad=torch.autograd.grad(e,xx)[0]
            norm=grad.flatten(1).norm(dim=1).mean().item(); grad_norms.append(norm)
            x0=(x0-lam*grad.clamp(-3,3)).detach(); corrections+=1
        if projection and i>=len(ids)//2 and i%4==0:
            x0=((project(x0*scale+mean,start,goal,'sdf',delta,projection_steps)-mean)/scale).detach()
        if i+1<len(ids):
            nxt=ids[i+1]; z=a[nxt].sqrt()*x0+(1-a[nxt]).sqrt()*eps
        else: z=x0
    u=terminal_correct((z*scale+mean).clamp(-2.5,2.5),start,goal).clamp(-2.5,2.5).detach()
    return u,{'seconds':time.perf_counter()-t0,'gradient_norm':float(np.mean(grad_norms)) if grad_norms else 0,
              'corrections':corrections,'guidance_rollouts':corrections*K if kind=='cvar' else 0}
