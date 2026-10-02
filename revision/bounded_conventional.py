"""Time-bounded version of the existing conventional SLSQP comparator.

Same cost, Gaussian chance constraint, bounds, ftol, and maximum iterations as
conventional.optimize. Extra wall guard returns the last evaluated controls,
records failure, and leaves final admission to the independent validator.
"""
import time
import numpy as np
import torch
from scipy.optimize import minimize
from .risk import chance,nominal_cost

class BudgetExpired(Exception):pass

def optimize(initial,start,goal,problem,delta,seconds=.6,maxiter=300,ftol=1e-5):
    st=torch.tensor(start[None],dtype=torch.float64);gt=torch.tensor(goal[None],dtype=torch.float64)
    cache={};began=time.perf_counter();last=initial.ravel().astype(float);calls=0;gradient_seconds=0.
    def compute(x):
        nonlocal last,calls,gradient_seconds
        if time.perf_counter()-began>seconds:raise BudgetExpired()
        key=x.tobytes()
        if cache.get('key')!=key:
            compute_started=time.perf_counter()
            z=torch.tensor(x.reshape(1,problem.horizon,2),dtype=torch.float64,requires_grad=True)
            cost=nominal_cost(z,st,problem)[0];risk=chance(z,st,gt,problem)['risk'][0]
            cg=torch.autograd.grad(cost,z,retain_graph=True)[0].detach().numpy().ravel()
            rg=torch.autograd.grad(risk,z)[0].detach().numpy().ravel()
            cache.update(key=key,cost=float(cost),risk=float(risk),cg=cg,rg=rg)
            last=x.copy();calls+=1
            gradient_seconds+=time.perf_counter()-compute_started
        return cache
    try:
        result=minimize(lambda x:compute(x)['cost'],last,jac=lambda x:compute(x)['cg'],method='SLSQP',
            bounds=[(-problem.control_limit,problem.control_limit)]*initial.size,
            constraints=[{'type':'ineq','fun':lambda x:delta-compute(x)['risk'],'jac':lambda x:-compute(x)['rg']}],
            options={'ftol':ftol,'maxiter':maxiter})
        controls=result.x.reshape(problem.horizon,2);success=bool(result.success);message=str(result.message);iterations=int(result.nit)
    except BudgetExpired:
        controls=last.reshape(problem.horizon,2);success=False;message='per-plan wall budget expired; last evaluated candidate retained';iterations=None
    with torch.no_grad():
        u=torch.tensor(controls[None],dtype=torch.float64)
        cost=float(nominal_cost(u,st,problem)[0]);risk=float(chance(u,st,gt,problem)['risk'][0])
    return controls,{'optimizer_success':success,'message':message,'iterations':iterations,'optimizer_seconds':time.perf_counter()-began,'gradient_evaluations':calls,'energy_and_backward_seconds':gradient_seconds,'predicted_risk':risk,'nominal_cost':cost,'internal_risk_threshold':delta}
