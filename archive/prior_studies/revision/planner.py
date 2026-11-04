"""Validated candidate selection; rejected attempts never become returned plans."""
import numpy as np
import time
from .guidance import sample
from .evaluate import evaluate
from .risk import nominal_cost
import torch

def plan(model,mean,scale,start,goal,problem,delta,kind='chance',lam=8,candidates=4,
         seed=0,validation_seed=810000,draws=8192,confidence=.95):
    if not isinstance(candidates,int) or candidates<1:raise ValueError('candidates must be a positive integer')
    if not 0<delta<1:raise ValueError('delta must be in (0,1)')
    planning_start=time.perf_counter()
    starts=np.repeat(np.asarray(start,dtype='float32')[None],candidates,axis=0)
    goals=np.repeat(np.asarray(goal,dtype='float32')[None],candidates,axis=0)
    u,metadata=sample(model,mean,scale,starts,goals,problem,kind,delta,lam,seed=seed)
    # Simultaneous coverage across the fixed candidate batch used for selection.
    per_candidate=1-(1-confidence)/candidates
    validation_start=time.perf_counter()
    estimates=evaluate(u.cpu().numpy(),starts,goals,problem,draws,validation_seed,delta,per_candidate)
    metadata['validation_seconds']=time.perf_counter()-validation_start
    costs=nominal_cost(u,torch.tensor(starts,device=u.device),problem).detach().cpu().numpy()
    metadata['planning_seconds']=time.perf_counter()-planning_start
    valid=[i for i,e in enumerate(estimates) if e['accepted']]
    if not valid:return {'accepted':False,'controls':None,'reason':'No candidate passed the execution-risk confidence bound',
                         'attempts':estimates,'metadata':metadata}
    i=min(valid,key=lambda j:costs[j])
    return {'accepted':True,'controls':u[i].cpu().numpy(),'selected':i,'risk_upper':estimates[i]['risk_upper'],
            'planning_cost':float(costs[i]),'attempts':estimates,'metadata':metadata}
