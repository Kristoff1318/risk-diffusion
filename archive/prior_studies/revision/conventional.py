"""Conventional chance-constrained optimizer as an additional comparator."""
import json,time,argparse
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from .problem import Problem
from .nominal import dataset
from .risk import chance,nominal_cost
from .guidance import load,sample
from .evaluate import evaluate

def optimize(initial,start,goal,p,delta,maxiter=300,ftol=1e-5):
    st=torch.tensor(start[None],dtype=torch.float64);gt=torch.tensor(goal[None],dtype=torch.float64);cache={}
    def compute(x):
        key=x.tobytes()
        if cache.get('key')!=key:
            z=torch.tensor(x.reshape(1,p.horizon,2),dtype=torch.float64,requires_grad=True)
            cost=nominal_cost(z,st,p)[0];risk=chance(z,st,gt,p)['risk'][0]
            cg=torch.autograd.grad(cost,z,retain_graph=True)[0].detach().numpy().ravel()
            rg=torch.autograd.grad(risk,z)[0].detach().numpy().ravel()
            cache.update(key=key,cost=float(cost),risk=float(risk),cg=cg,rg=rg)
        return cache
    began=time.perf_counter()
    ans=minimize(lambda x:compute(x)['cost'],initial.ravel().astype(float),jac=lambda x:compute(x)['cg'],method='SLSQP',
                 bounds=[(-p.control_limit,p.control_limit)]*initial.size,
                 constraints=[{'type':'ineq','fun':lambda x:delta-compute(x)['risk'],'jac':lambda x:-compute(x)['rg']}],
                 options={'ftol':ftol,'maxiter':maxiter})
    final=compute(ans.x)
    return ans.x.reshape(p.horizon,2),{'optimizer_success':bool(ans.success),'iterations':int(ans.nit),
        'seconds':time.perf_counter()-began,'predicted_risk':final['risk'],'planning_cost':final['cost']}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--margin',type=float,default=1.);parser.add_argument('--maxiter',type=int,default=300);parser.add_argument('--ftol',type=float,default=1e-5);parser.add_argument('--output');args=parser.parse_args()
    torch.set_num_threads(2);out=Path(args.output) if args.output else Path('risk_diffusion/revision/outputs/conventional_refined' if args.margin==1. else 'risk_diffusion/revision/outputs/conventional_margin08');out.mkdir(parents=True,exist_ok=True);rows=[]
    configs=[(Problem(process_sigma=.04,goal_half=.45,name='moderate_noise'),'nominal_seed0'),
             (Problem(strength=.35,process_sigma=.04,goal_half=.55,name='nonlinear'),'nonlinear_seed0')]
    for p,ckpt in configs:
        s,g,waypoints=dataset(32,2901,strength=p.strength)
        model,mean,scale=load(f'risk_diffusion/revision/outputs/{ckpt}/nominal.pt')
        raw,meta=sample(model,mean,scale,s,g,p,seed=40101)
        for label,initial in [('WaypointChanceOptimizer',waypoints),('DiffusionFinalChanceOptimizer',raw.cpu().numpy())]:
            sols=[];infos=[]
            for i in range(12):
                sol,info=optimize(initial[i],s[i],g[i],p,.1*args.margin,maxiter=args.maxiter,ftol=args.ftol);sols.append(sol);infos.append(info)
            validation=evaluate(np.array(sols),s[:12],g[:12],p,4096,75100,.1);audit=evaluate(np.array(sols),s[:12],g[:12],p,8192,198100,.1)
            for i,(info,e,a) in enumerate(zip(infos,validation,audit)):rows.append(dict(method=label,problem=p.name,task=i,delta=.1,
                          audit_risk=a['risk'],**info,**e))
            np.savez_compressed(out/f'{p.name}_{label}.npz',u=np.array(sols),start=s[:12],goal=g[:12])
            print(p.name,label,'mean audit risk',np.mean([x['risk'] for x in audit]),'accepted',sum(x['accepted'] for x in validation),flush=True)
    (out/'results.json').write_text(json.dumps({'rows':rows,'optimizer_risk_margin':args.margin,'maxiter':args.maxiter,'ftol':args.ftol,'note':'12 prespecified first tasks; optimizer seconds exclude proposal generation'},indent=2))
if __name__=='__main__':main()
