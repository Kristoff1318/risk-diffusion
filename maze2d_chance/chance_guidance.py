"""Local slides 13–14 direct-action chance guidance, retained for inspection.

EXPERIMENTAL: this model describes direct action execution, not the restored
reference-tracking controller. It is not enabled by nominal_baseline.py and
has not demonstrated budget compliance. See README.md before using it.
"""
import time
import numpy as np
import torch
from runtime import load_policy
from diffuser.models.helpers import apply_conditioning

SIGMA = .005
GAMMA = 10.
STRENGTH = .002

@torch.jit.script
def propagate(actions:torch.Tensor,start:torch.Tensor,sigma:float,a:float,b:float):
    # Semi-implicit Euler with native implicit damping; noise AFTER each step.
    p=start[:,:2];v=start[:,2:];pp=torch.zeros_like(p);pv=pp.clone();vv=pp.clone()
    means=torch.jit.annotate(list[torch.Tensor],[])
    covs=torch.jit.annotate(list[torch.Tensor],[])
    for k in range(actions.shape[1]):
        mask=(v.abs()<5.).to(v.dtype);c=a*mask
        v=a*v.clamp(-5.,5.)+b*actions[:,k].clamp(-1.,1.)
        p=p+.01*v
        pp_new=pp+.02*c*pv+.0001*c*c*vv
        pv=c*pv+.01*c*c*vv
        vv=c*c*vv+sigma*sigma
        pp=pp_new
        means.append(torch.cat([p,v],-1));covs.append(pp)
    return torch.stack(means,1),torch.stack(covs,1)

class DirectActionChanceModel:
    def __init__(self,env,bounds,sigma=SIGMA):
        assert env.frame_skip == 1 and np.isclose(env.dt, .01)
        assert np.allclose(env.model.dof_damping, 1.)
        assert np.allclose(env.model.actuator_gear[:,0], 100.)
        mass=env.model.body_mass[1];self.a=float(mass/(mass+.01));self.b=float(1./(mass+.01));self.sigma=sigma
        self.centers=torch.tensor(np.array([env.model.geom_pos[i,:2]-env.model.body_pos[1,:2] for i in range(env.model.ngeom) if (env.model.geom_id2name(i) or '').startswith('wall_')]),dtype=torch.float32)
        self.oscale=torch.tensor((bounds['observations_maxs']-bounds['observations_mins'])/2)
        self.omiddle=torch.tensor((bounds['observations_maxs']+bounds['observations_mins'])/2)
        self.ascale=torch.tensor((bounds['actions_maxs']-bounds['actions_mins'])/2)
        self.amiddle=torch.tensor((bounds['actions_maxs']+bounds['actions_mins'])/2)
    def physical_actions(self,z): return z[...,:2]*self.ascale+self.amiddle
    def risk(self,z,start,goal):
        actions=self.physical_actions(z[:,:-1])
        means,var=propagate(actions,start,self.sigma,self.a,self.b)
        p=means[...,:2];diff=p[:,:,None,:]-self.centers
        q=diff.abs()-.5;outside=q.clamp(min=0);dist=outside.square().sum(-1).clamp(min=1e-20).sqrt()
        # Exact box SDF, inflated by physical sphere radius + native contact margin.
        m=dist+q.max(-1).values.clamp(max=0)-.102
        outward=outside/dist[...,None]
        interior=torch.nn.functional.one_hot(q.argmax(-1),2).to(q.dtype)
        grad=torch.where((q.max(-1).values>0)[...,None],outward,interior)*diff.sign()
        s=(grad.square()*var[:,:,None,:]).sum(-1).clamp(min=1e-12).sqrt()
        wall=.5*torch.erfc(m/(s*2**.5))
        gd=p[:,-1]-goal;dn=gd.square().sum(-1).clamp(min=1e-20).sqrt();gg=gd/dn[:,None]
        gs=(gg.square()*var[:,-1]).sum(-1).clamp(min=1e-12).sqrt()
        goalrisk=.5*torch.erfc((.5-dn)/(gs*2**.5))
        # Includes terminal obstacles too (stronger than slide k=1..N-1 convention).
        return wall.sum((1,2))+goalrisk,wall.sum((1,2)),goalrisk,means,var
    def energy(self,z,start,goal,budget):
        risk,*_=self.risk(z,start,goal)
        return torch.nn.functional.softplus(GAMMA*(risk-budget))

def sample_direct_actions(diffusion,chance,starts,goals,seeds,budgets,strength=STRENGTH):
    B=len(seeds);gens=[torch.Generator(device='cuda').manual_seed(s) for s in seeds]
    def noise():return torch.stack([torch.randn((384,6),device='cuda',generator=g) for g in gens])
    norm=lambda a:(a-chance.omiddle.numpy())/chance.oscale.numpy()
    cond={0:torch.tensor(norm(starts),device='cuda').float(),383:torch.tensor(norm(np.c_[goals,np.zeros((B,2))]),device='cuda').float()}
    x=apply_conditioning(noise(),cond,2)
    ss=torch.tensor(starts).float();gg=torch.tensor(goals).float();bb=torch.tensor(budgets).float()
    guide_mask=torch.tensor([b>=0 for b in budgets]).float()
    trace=[];beg=time.perf_counter()
    for k in reversed(range(256)):
        t=torch.full((B,),k,device='cuda',dtype=torch.long)
        with torch.no_grad():mean,variance,logvar=diffusion.p_mean_variance(x,cond,t)
        if guide_mask.any():
            # Literal E(u_t) on the CURRENT noisy normalized action coordinates;
            # physical action clipping is part of f and therefore part of autodiff.
            z=x.detach().cpu().requires_grad_(True)
            energies=chance.energy(z,ss,gg,bb)
            grad=torch.autograd.grad((energies*guide_mask).sum(),z)[0]
            mean=mean-strength*variance*grad.to('cuda')
            if k%32==0:trace.append(dict(step=k,energy=energies.detach().tolist(),gradient_norm=grad.flatten(1).norm(dim=1).tolist()))
        with torch.no_grad():x=apply_conditioning(mean+(k>0)*(.5*logvar).exp()*noise(),cond,2)
    torch.cuda.synchronize()
    return x.cpu().numpy(),time.perf_counter()-beg,trace
