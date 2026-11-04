"""Slides 13–14 chance energy; native Maze2D action diffusion, no replanning."""
import os,sys,json,pickle,time,argparse,hashlib
from pathlib import Path
sys.path.insert(0,'/home/kristoff/Documents/rb-diffuser')
import numpy as np
import torch
import gym,d4rl
from scipy.stats import beta as beta_distribution
from diffuser.models.helpers import apply_conditioning
ROOT=Path(__file__).resolve().parent
EPOCH=1960000
CHECKPOINT=Path('/home/kristoff/Documents/rb-diffuser/logs/maze2d-large-v1/diffusion/H384_T256')
SIGMA=.005
GAMMA=10.
STRENGTH=.002 # lambda * beta; energy is NOT divided by gamma, horizon, or constraints.
BUDGETS=[.05,.15,.30]
TASKS=[([1.,1.,0.,0.],[7.,9.]),([1.,1.,0.,0.],[3.,1.]),([3.,1.,0.,0.],[5.,1.])]

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

class Chance:
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

def load_model():
    cfg=lambda name:pickle.load(open(CHECKPOINT/(name+'_config.pkl'),'rb'))
    model=cfg('model')();diffusion=cfg('diffusion')(model)
    epoch=EPOCH
    state=torch.load(CHECKPOINT/f'state_{epoch}.pt');diffusion.load_state_dict(state['ema']);diffusion.eval()
    for p in diffusion.parameters():p.requires_grad_(False)
    return diffusion,epoch

def sample(diffusion,chance,starts,goals,seeds,budgets,strength=STRENGTH):
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

def wall_contact(e):
    for c in e.sim.data.contact:
        if c.dist<=.002 and any((e.model.geom_id2name(int(g)) or '').startswith('wall_') for g in [c.geom1,c.geom2]):return True
    return False

def execute(e,actions,start,goal,sigma,seed,save=False):
    e.reset();e.set_state(np.array(start[:2]),np.array(start[2:]));e.set_target(np.array(goal))
    rng=np.random.RandomState(seed);collision=False;hist=[e._get_obs().copy()];cost=0.
    for action in actions[:-1]:
        obs,_,_,_=e.step(action);collision=collision or wall_contact(e)
        # Exact declared disturbance; reset randomness overwritten above.
        velocity=obs[2:]+sigma*rng.randn(2);e.set_state(obs[:2].copy(),velocity);obs=e._get_obs()
        hist.append(obs.copy());cost+=float(np.square(np.clip(action,-1,1)).sum())
    dist=float(np.linalg.norm(obs[:2]-goal));success=dist<=.5
    return dict(failure=bool(collision or not success),collision=bool(collision),goal_failure=not success,distance=dist,cost=cost,path_length=float(np.linalg.norm(np.diff(np.array(hist)[:,:2],axis=0),axis=1).sum()),trajectory=np.array(hist) if save else None)

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,mid-half),min(1.,mid+half)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=3);p.add_argument('--seed-base',type=int,default=21);p.add_argument('--execution-seed-base',type=int,default=1000000);p.add_argument('--executions',type=int,default=200);p.add_argument('--strength',type=float,default=STRENGTH);p.add_argument('--out',default='results');args=p.parse_args()
    out=ROOT/args.out;out.mkdir(exist_ok=True)
    if (out/'results.json').exists():
        raise FileExistsError(f'Choose a fresh --out directory; preserving {out}')
    protocol=dict(sigma=SIGMA,gamma=GAMMA,lambda_beta=args.strength,epoch=EPOCH,tasks=TASKS,budgets=BUDGETS,seeds=args.seeds,seed_base=args.seed_base,executions=args.executions,execution_seed_base=args.execution_seed_base)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    e=gym.make('maze2d-large-v1').unwrapped;e.reset();bounds=np.load(ROOT/'normalization.npz');ch=Chance(e,bounds)
    d,epoch=load_model()
    # Each independently seeded candidate is paired across all budgets/methods.
    rows=[];plans=[]
    for method,budget in [('nominal',-1.)]+[('chance',b) for b in BUDGETS]:
        starts=np.array([t[0] for t in TASKS for s in range(args.seeds)]);goals=np.array([t[1] for t in TASKS for s in range(args.seeds)]);seeds=[args.seed_base+s for t in TASKS for s in range(args.seeds)]
        print('SAMPLING',method,budget,flush=True)
        samples,seconds,trace=sample(d,ch,starts,goals,seeds,[budget]*len(seeds),args.strength)
        np.savez_compressed(out/f'{method}_{budget}_plans.npz',samples=samples,starts=starts,goals=goals,seeds=seeds)
        json.dump(trace,open(out/f'{method}_{budget}_trace.json','w'),indent=2)
        risks,wr,gr,means,cov=ch.risk(torch.tensor(samples),torch.tensor(starts).float(),torch.tensor(goals).float())
        for i,(z,start,goal,seed) in enumerate(zip(samples,starts,goals,seeds)):
            actions=ch.physical_actions(torch.tensor(z)).numpy();nom=execute(e,actions,start,goal,0.,1,True);nompath=nom.pop('trajectory')
            counts={k:0 for k in ['failure','collision','goal_failure']};dist=[];cost=[];length=[];beg=time.perf_counter()
            for j in range(args.executions):
                trial=execute(e,actions,start,goal,SIGMA,args.execution_seed_base+j,save=j==0)
                for k in counts:counts[k]+=int(trial[k])
                dist.append(trial['distance']);cost.append(trial['cost']);length.append(trial['path_length'])
                if j==0:np.savez_compressed(out/f'{method}_{budget}_{i}_rollouts.npz',nominal=nompath,stochastic=trial['trajectory'],mean=means[i].detach().numpy(),position_variance=cov[i].detach().numpy())
            row=dict(method=method,budget=budget,task=i//args.seeds,seed=seed,predicted_risk=float(risks[i]),predicted_wall=float(wr[i]),predicted_goal=float(gr[i]),surrogate_accepted=bool(budget>=0 and risks[i]<=budget),nominal=nom,executions=args.executions,counts=counts,failure_rate=counts['failure']/args.executions,failure_ci95=wilson(counts['failure'],args.executions),goal_distance_mean=float(np.mean(dist)),cost_mean=float(np.mean(cost)),path_length_mean=float(np.mean(length)),sampling_seconds_per_candidate=seconds/len(seeds),sampling_batch_seconds=seconds,execution_seconds=time.perf_counter()-beg)
            k=counts['failure'];n=args.executions
            row['failure_exact_ci95']=[float(beta_distribution.ppf(.025,k,n-k+1)) if k else 0.,float(beta_distribution.ppf(.975,k+1,n-k)) if k<n else 1.]
            row['audit_upper_simultaneous95']=float(beta_distribution.ppf(1-.05/(len(BUDGETS)*len(seeds)),k+1,n-k)) if k<n else 1.
            row['audit_accepted']=bool(budget>=0 and row['audit_upper_simultaneous95']<=budget);row['false_safe_confirmed']=bool(row['surrogate_accepted'] and row['failure_exact_ci95'][0]>budget)
            rows.append(row);json.dump(rows,open(out/'results.json','w'),indent=2);print(row,flush=True)
    json.dump(dict(sigma=SIGMA,gamma=GAMMA,lambda_beta=args.strength,epoch=epoch,checkpoint=str(CHECKPOINT/f'state_{epoch}.pt'),tasks=TASKS,executions=args.executions,seeds=args.seeds,seed_base=args.seed_base,execution_seed_base=args.execution_seed_base,budgets=BUDGETS,noise='iid N(0,sigma^2 I2) velocity impulse AFTER native step; x0 exact',horizon=383),open(out/'manifest.json','w'),indent=2)
if __name__=='__main__':main()
