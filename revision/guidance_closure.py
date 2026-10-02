"""Opt-in matched CPU sampler for closure-guidance ablations.

Historical sampling remains unchanged in guidance.py. This loop reproduces its
nominal/chance/CVaR paths and adds closure energy with explicit timing only.
"""
import time
import numpy as np
import torch
from .nominal import schedule
from .risk import energy,nominal_cost
from .sine_moments import chance_sine

def sample(model,mean,scale,starts,goals,problem,kind=None,delta=.1,lam=8.,steps=64,
           seed=0,K=256,cost_weight=.1,gamma=30):
    if mean.device.type!='cpu':raise ValueError('This bounded comparison is CPU-only')
    s=torch.as_tensor(starts,dtype=torch.float32);g=torch.as_tensor(goals,dtype=torch.float32)
    generator=torch.Generator().manual_seed(seed)
    z=torch.randn((len(s),problem.horizon,2),generator=generator)
    noise_generator=torch.Generator().manual_seed(seed+900001)
    noise=torch.randn((K,len(s),problem.horizon+1,2),generator=noise_generator) if kind=='cvar' else None
    alphas=schedule();ids=np.linspace(len(alphas)-1,0,steps,dtype=int)
    began=time.perf_counter();gradient_seconds=0.;trace=[]
    for j,i in enumerate(ids):
        active=kind is not None and lam>0 and float(alphas[i])>=.5
        with torch.set_grad_enabled(active):
            z=z.detach().requires_grad_(active)
            v=model(z,torch.full((len(s),),i/(len(alphas)-1)),s,g)
            clean=alphas[i].sqrt()*z-(1-alphas[i]).sqrt()*v
            epsilon=(1-alphas[i]).sqrt()*z+alphas[i].sqrt()*v
            if active:
                t=time.perf_counter();controls=clean*scale+mean
                if kind=='closure':
                    residual=chance_sine(controls,s,g,problem)['risk']-delta
                    e=torch.nn.functional.softplus(gamma*residual)/gamma+cost_weight*nominal_cost(controls,s,problem)
                else:e=energy(controls,s,g,problem,kind,delta,noise,gamma,cost_weight)
                grad=torch.autograd.grad(e.sum(),z)[0]
                gradient_seconds+=time.perf_counter()-t
                epsilon=epsilon+lam*(1-alphas[i]).sqrt()*grad
                clean=(z-(1-alphas[i]).sqrt()*epsilon)/alphas[i].sqrt()
                trace.append({'step':int(i),'energy_mean':float(e.detach().mean()),'gradient_norm':float(grad.detach().flatten(1).norm(dim=1).mean())})
            z=(alphas[ids[j+1]].sqrt()*clean+(1-alphas[ids[j+1]]).sqrt()*epsilon).detach() if j+1<len(ids) else clean.detach()
    return z*scale+mean,{'sampling_seconds_batch':time.perf_counter()-began,'energy_and_backward_seconds_batch':gradient_seconds,'diffusion_steps':steps,'guidance_steps':len(trace),'unique_guidance_rollouts_per_plan':K if kind=='cvar' else 0,'guidance_rollouts_per_plan':K*len(trace) if kind=='cvar' else 0,'trace':trace}
