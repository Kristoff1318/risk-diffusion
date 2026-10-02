"""Matched conventional refinement with original or closure constraints.

The Linux real-time signal timer bounds the SLSQP iteration phase. Final
diagnostics and independent validation are timed separately and included in
the complete request. Timed-out candidates remain in the evaluation bank.
"""
import signal,time
import numpy as np
import torch
from scipy.optimize import minimize
from .risk import chance,nominal_cost
from .sine_moments import chance_sine

class SolverTimeout(Exception):pass

def optimize(initial,start,goal,problem,threshold,kind,seconds=.5,maxiter=300,ftol=1e-5):
    if kind not in ('jacobian','closure'):raise ValueError(kind)
    risk_function=chance if kind=='jacobian' else chance_sine
    st=torch.tensor(start[None],dtype=torch.float64);gt=torch.tensor(goal[None],dtype=torch.float64)
    cache={};last=initial.ravel().astype(float);calls=0;gradient_seconds=0.;began=time.perf_counter()
    def compute(x):
        nonlocal last,calls,gradient_seconds
        key=x.tobytes()
        if cache.get('key')!=key:
            t=time.perf_counter();z=torch.tensor(x.reshape(1,problem.horizon,2),dtype=torch.float64,requires_grad=True)
            cost=nominal_cost(z,st,problem)[0];risk=risk_function(z,st,gt,problem)['risk'][0]
            cg=torch.autograd.grad(cost,z,retain_graph=True)[0].detach().numpy().ravel();rg=torch.autograd.grad(risk,z)[0].detach().numpy().ravel()
            cache.update(key=key,cost=float(cost),risk=float(risk),cg=cg,rg=rg);last=x.copy();calls+=1
            gradient_seconds+=time.perf_counter()-t
        return cache
    def timeout(signum,frame):raise SolverTimeout()
    handler=signal.getsignal(signal.SIGALRM);old_timer=signal.getitimer(signal.ITIMER_REAL)
    if old_timer[0]!=0:raise RuntimeError('Refusing to override an existing timer')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        result=minimize(lambda x:compute(x)['cost'],last,jac=lambda x:compute(x)['cg'],method='SLSQP',
                        bounds=[(-problem.control_limit,problem.control_limit)]*initial.size,
                        constraints=[{'type':'ineq','fun':lambda x:threshold-compute(x)['risk'],'jac':lambda x:-compute(x)['rg']}],
                        options={'maxiter':maxiter,'ftol':ftol})
        controls=result.x.reshape(problem.horizon,2);success=bool(result.success);message=str(result.message);iterations=int(result.nit);timed_out=False
    except SolverTimeout:
        controls=last.reshape(problem.horizon,2);success=False;message='hard SLSQP iteration timer expired; last completed evaluation retained';iterations=None;timed_out=True
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
    solver_seconds=time.perf_counter()-began;t=time.perf_counter()
    with torch.no_grad():
        u=torch.tensor(controls[None],dtype=torch.float64)
        risk=float(risk_function(u,st,gt,problem)['risk'][0]);cost=float(nominal_cost(u,st,problem)[0])
    return controls,{'optimizer_success':success,'timed_out':timed_out,'message':message,'iterations':iterations,'gradient_evaluations':calls,'gradient_seconds':gradient_seconds,'solver_seconds':solver_seconds,'postsolve_diagnostic_seconds':time.perf_counter()-t,'predicted_risk':risk,'nominal_cost':cost,'internal_threshold':threshold,'constraint_violation':max(0.,risk-threshold)}
