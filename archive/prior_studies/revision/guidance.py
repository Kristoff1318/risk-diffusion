"""Risk gradient modifies the diffusion score during DDIM denoising."""
import time
import numpy as np
import torch
from .nominal import Denoiser,schedule
from .risk import energy,chance,sampled_cvar,hard_loss,simulate
from .problem import Problem

def load(path,device='cuda'):
    c=torch.load(path,map_location=device);m=Denoiser(extra_condition=c.get('extra_condition',0)).to(device);m.load_state_dict(c['model']);m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m,c['mean'].to(device),c['scale'].to(device)

def sample(model,mean,scale,starts,goals,p=Problem(),kind=None,delta=.1,lam=1.,steps=64,seed=0,K=256,cost_weight=0.,gamma=30,projection=False,final_only=False,condition_budget=None,cfg_scale=1.):
    device=mean.device;s=torch.as_tensor(starts,dtype=torch.float32,device=device);g=torch.as_tensor(goals,dtype=torch.float32,device=device)
    gen=torch.Generator(device=device);gen.manual_seed(seed)
    z=torch.randn((len(s),p.horizon,2),device=device,generator=gen)
    # Independent generator keeps initial diffusion noise matched across methods.
    noisegen=torch.Generator(device=device);noisegen.manual_seed(seed+900001)
    noise=torch.randn((K,len(s),p.horizon+1,2),device=device,generator=noisegen) if kind=='cvar' else None
    extra=None
    if model.extra_condition:
        extra=torch.zeros((len(s),2),device=device)
        if condition_budget is not None:extra[:,0]=condition_budget;extra[:,1]=1.
    a=schedule().to(device);ids=np.linspace(len(a)-1,0,steps,dtype=int);trace=[];projection_records=[]
    if device.type=='cuda':torch.cuda.synchronize()
    started=time.perf_counter()
    for j,i in enumerate(ids):
        active=kind is not None and lam>0 and float(a[i])>=.5
        with torch.set_grad_enabled(active):
            z=z.detach().requires_grad_(active)
            tt=torch.full((len(s),),i/(len(a)-1),device=device)
            v=model(z,tt,s,g,extra)
            if extra is not None and condition_budget is not None and cfg_scale!=1.:
                unconditional=model(z,tt,s,g,torch.zeros_like(extra))
                v=unconditional+cfg_scale*(v-unconditional)
            clean=a[i].sqrt()*z-(1-a[i]).sqrt()*v;eps=(1-a[i]).sqrt()*z+a[i].sqrt()*v
            if active:
                e=energy(clean*scale+mean,s,g,p,kind,delta,noise,gamma,cost_weight)
                grad=torch.autograd.grad(e.sum(),z)[0]
                # score=-eps/sqrt(1-a), so s_guided=s-lambda*grad(E).
                guided_eps=eps+lam*(1-a[i]).sqrt()*grad
                guided_clean=(z-(1-a[i]).sqrt()*guided_eps)/a[i].sqrt()
                trace.append({'step':int(i),'energy_mean':float(e.detach().mean()),
                              'gradient_norm':float(grad.detach().flatten(1).norm(dim=1).mean())})
                clean=guided_clean;eps=guided_eps
        if projection and ((final_only and j==len(ids)-1) or (not final_only and j in (len(ids)-9,len(ids)-5,len(ids)-1))):
            from .projection import project_batch
            corrected,records=project_batch(clean.detach()*scale+mean,s,g,p,scale)
            projection_records.extend(records);clean=(corrected-mean)/scale
        z=(a[ids[j+1]].sqrt()*clean+(1-a[ids[j+1]]).sqrt()*eps).detach() if j+1<len(ids) else clean.detach()
    u=z*scale+mean
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    return u,{'seconds_batch':elapsed,'diffusion_steps':steps,'guidance_steps':len(trace),
              'guidance_rollouts_per_plan':K*len(trace) if kind=='cvar' else 0,
              'unique_guidance_rollouts_per_plan':K if kind=='cvar' else 0,'trace':trace,
              'projection_calls':len(projection_records)//len(s),'projection_failed_calls':sum(not r['feasible'] for r in projection_records),
              'projection_attempts':sum(r['attempted'] for r in projection_records),
              'projection_iterations':sum(r['iterations'] for r in projection_records)}
