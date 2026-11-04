"""Local constrained projection in controls, with explicit feasibility checks."""
import numpy as np
import torch
from scipy.optimize import minimize
from .risk import simulate

def constraints(flat,start,goal,p):
    u=flat.reshape(1,p.horizon,2);x=simulate(u,start[None],p)[0,0]
    obs=torch.tensor(p.obstacles,dtype=flat.dtype,device=flat.device)
    safe=(x[:-1,None,:]-obs[None,:,:2]).norm(dim=-1)-obs[None,:,2]
    e=x[-1]-goal
    return torch.cat((safe.flatten(),p.goal_half-e,p.goal_half+e))

def project_batch(u,start,goal,p,scale,maxiter=60,tol=1e-6):
    original=u.detach().cpu().numpy();starts=start.detach().cpu().numpy();goals=goal.detach().cpu().numpy()
    if original.shape!=(len(original),p.horizon,2) or starts.shape!=(len(original),2) or goals.shape!=starts.shape:
        raise ValueError('Projection requires matched controls, start, and goal batches')
    scaling=np.broadcast_to(scale.detach().cpu().numpy(),original.shape)
    solutions=[];diags=[]
    for ref,s,g,weight in zip(original,starts,goals,scaling):
        st=torch.tensor(s,dtype=torch.float64);gt=torch.tensor(g,dtype=torch.float64);flat=ref.ravel().astype(float)
        cache={}
        def c_and_j(x):
            key=x.tobytes()
            if cache.get('key')!=key:
                xx=torch.tensor(x,dtype=torch.float64,requires_grad=True)
                val=constraints(xx,st,gt,p)
                jac=torch.autograd.functional.jacobian(lambda v:constraints(v,st,gt,p),xx)
                cache.update(key=key,value=val.detach().numpy(),jac=jac.detach().numpy())
            return cache['value'],cache['jac']
        initial=constraints(torch.tensor(flat),st,gt,p).numpy()
        if initial.min()>=-tol and np.abs(flat).max()<=p.control_limit:
            solution=flat;iterations=0;attempted=False;optimizer_success=True
        else:
            w=weight.ravel().astype(float)
            ans=minimize(lambda v:.5*np.sum(((v-flat)/w)**2),flat,
                         jac=lambda v:(v-flat)/(w*w),method='SLSQP',bounds=[(-p.control_limit,p.control_limit)]*len(flat),
                         constraints=[{'type':'ineq','fun':lambda v:c_and_j(v)[0],'jac':lambda v:c_and_j(v)[1]}],
                         options={'maxiter':maxiter,'ftol':1e-8})
            solution=ans.x;iterations=int(ans.nit);attempted=True;optimizer_success=bool(ans.success)
        solution=np.clip(solution,-p.control_limit,p.control_limit)
        margin=float(constraints(torch.tensor(solution),st,gt,p).min())
        feasible=margin>=-tol and np.abs(solution).max()<=p.control_limit
        solutions.append(solution.reshape(p.horizon,2))
        diags.append({'attempted':attempted,'optimizer_success':optimizer_success,'feasible':bool(feasible),
                      'min_constraint':margin,'iterations':iterations,'squared_displacement':float(np.sum((solution-flat)**2))})
    return torch.tensor(np.array(solutions),dtype=u.dtype,device=u.device),diags
