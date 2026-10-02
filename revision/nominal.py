"""Risk-agnostic control DDPM: cosine schedule, v prediction, EMA, raw DDIM."""
import copy
import math
import time
import numpy as np
import torch
from torch import nn

H=20
DT=.2

def drift(x, strength=.12):
    return strength * torch.stack((torch.sin(2*x[...,1]),torch.sin(2*x[...,0])),dim=-1)

def rollout(u,start,strength=.12):
    x=start; states=[x]
    for k in range(u.shape[1]):
        x=x+DT*(u[:,k]+drift(x,strength));states.append(x)
    return torch.stack(states,1)

def dataset(n,seed,strength=.12,obstacles=((0.,0.,.43),),height_range=(.58,1.15)):
    if len(obstacles)>1:
        ss,gg,uu=dataset(n*5,seed,strength,height_range=height_range)
        xx=rollout(torch.tensor(uu),torch.tensor(ss),strength).numpy()
        obs=np.array(obstacles)
        valid=(np.linalg.norm(xx[:,:-1,None,:]-obs[None,None,:,:2],axis=-1)>obs[None,None,:,2]+.005).all((1,2))
        chosen=np.flatnonzero(valid)[:n]
        if len(chosen)<n:raise RuntimeError('Insufficient ordinary feasible training paths')
        return ss[chosen],gg[chosen],uu[chosen]
    """Ordinary collision-free routes in both homotopies; no stochastic labels."""
    r=np.random.default_rng(seed)
    s=np.column_stack((-1.5*np.ones(n),r.uniform(-.18,.18,n))).astype('float32')
    g=np.column_stack((1.5*np.ones(n),r.uniform(-.18,.18,n))).astype('float32')
    heights=r.uniform(height_range[0],height_range[1],n)*r.choice([-1.,1.],n)
    knots=np.stack((s,np.column_stack((np.full(n,-.62),heights)),np.column_stack((np.full(n,.62),heights)),g),1)
    paths=[]
    for t in np.linspace(0,3,H+1):
        j=min(int(t),2);a=t-j;a=a*a*(3-2*a)
        paths.append((1-a)*knots[:,j]+a*knots[:,j+1])
    x=torch.tensor(np.stack(paths,1),dtype=torch.float32)
    u=(x[:,1:]-x[:,:-1])/DT-drift(x[:,:-1],strength)
    return s,g,u.numpy()

def schedule(T=128):
    t=torch.arange(T+1,dtype=torch.float64)/T
    ac=torch.cos((t+.008)/1.008*math.pi/2)**2;ac=ac/ac[0]
    beta=(1-ac[1:]/ac[:-1]).clamp(max=.999)
    return torch.cumprod(1-beta,0).float()

class Denoiser(nn.Module):
    def __init__(self,width=256,extra_condition=0):
        super().__init__();self.extra_condition=extra_condition
        self.register_buffer('freqs',torch.exp(torch.linspace(math.log(1),math.log(1000),32)))
        self.net=nn.Sequential(nn.Linear(H*2+4+64+extra_condition,width),nn.SiLU(),nn.Linear(width,width),
                               nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,H*2))
    def forward(self,z,t,s,g,extra=None):
        phase=t[:,None]*self.freqs[None]
        pieces=[z.flatten(1),s,g,phase.sin(),phase.cos()]
        if self.extra_condition:
            if extra is None: raise ValueError('Explicit extra condition and missingness mask required')
            pieces.append(extra)
        return self.net(torch.cat(pieces,1)).reshape_as(z)

@torch.no_grad()
def sample(model,mean,scale,s,g,steps=64,seed=0,extra=None):
    device=mean.device;gen=torch.Generator(device=device);gen.manual_seed(seed)
    s=torch.as_tensor(s,dtype=torch.float32,device=device);g=torch.as_tensor(g,dtype=torch.float32,device=device)
    z=torch.randn((len(s),H,2),device=device,generator=gen);a=schedule().to(device)
    ids=np.linspace(len(a)-1,0,steps,dtype=int)
    for j,i in enumerate(ids):
        v=model(z,torch.full((len(s),),i/(len(a)-1),device=device),s,g,extra)
        clean=a[i].sqrt()*z-(1-a[i]).sqrt()*v
        eps=(1-a[i]).sqrt()*z+a[i].sqrt()*v
        z=(a[ids[j+1]].sqrt()*clean+(1-a[ids[j+1]]).sqrt()*eps) if j+1<len(ids) else clean
    return z*scale+mean

def metrics(u,s,g,radius=.43,strength=.12,obstacles=None):
    x=rollout(u,s,strength)
    obs=torch.tensor(obstacles or ((0.,0.,radius),),device=u.device,dtype=u.dtype)
    clear=((x[:,:-1,None,:]-obs[None,None,:,:2]).norm(dim=-1)-obs[None,None,:,2]).flatten(1).min(1).values
    err=(x[:,-1]-g).abs().max(1).values
    bounds=u.abs().flatten(1).max(1).values
    good=(clear>0)&(err<=.3)&(bounds<=2.5)
    return {'feasible_fraction':good.float().mean().item(),'collision_fraction':(clear<=0).float().mean().item(),
            'goal_fraction':(err<=.3).float().mean().item(),'control_fraction':(bounds<=2.5).float().mean().item(),
            'mean_goal_error':err.mean().item(),'upper_route_fraction':(x[:,H//2,1]>0).float().mean().item(),
            'roughness':(u[:,1:]-u[:,:-1]).square().mean().item()}

def train(seed=0,steps=4000,n=6000,device='cuda',strength=.12,obstacles=((0.,0.,.43),)):
    torch.manual_seed(seed);s,g,u=dataset(n,100+seed,strength,obstacles)
    st=torch.tensor(s,device=device);gt=torch.tensor(g,device=device);ut=torch.tensor(u,device=device)
    mean=ut.mean(0,keepdim=True);scale=ut.std(0,keepdim=True).clamp_min(.1);y=(ut-mean)/scale
    model=Denoiser().to(device);ema=copy.deepcopy(model).eval();opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    a=schedule().to(device);hist=[];started=time.perf_counter()
    for step in range(steps):
        ix=torch.randint(n,(256,),device=device);t=torch.randint(len(a),(256,),device=device)
        eps=torch.randn_like(y[ix]);at=a[t,None,None]
        z=at.sqrt()*y[ix]+(1-at).sqrt()*eps;target=at.sqrt()*eps-(1-at).sqrt()*y[ix]
        loss=(model(z,t.float()/(len(a)-1),st[ix],gt[ix])-target).square().mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        with torch.no_grad():
            for e,p in zip(ema.parameters(),model.parameters()):e.mul_(.995).add_(p,alpha=.005)
        if (step+1)%1000==0:
            hist.append({'step':step+1,'loss':loss.item()});print('nominal train',seed,step+1,loss.item(),flush=True)
    if device=='cuda':torch.cuda.synchronize()
    return ema,mean,scale,{'training_seconds':time.perf_counter()-started,'history':hist,'train_seed':seed,'train_steps':steps,'train_n':n,'strength':strength,'obstacles':obstacles,'torch_version':torch.__version__}
