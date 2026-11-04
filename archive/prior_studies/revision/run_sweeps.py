"""Prespecified sensitivity grid on tuning tasks, separate from reported test tasks."""
import json
from pathlib import Path
import numpy as np
import torch
from .problem import Problem
from .nominal import dataset
from .guidance import load,sample
from .risk import nominal_cost
from .evaluate import evaluate

def main():
    torch.set_num_threads(2);out=Path('risk_diffusion/revision/outputs/sweeps');out.mkdir(parents=True,exist_ok=True)
    model,mean,scale=load('risk_diffusion/revision/outputs/nominal_seed0/nominal.pt');s,g,_=dataset(24,1701)
    p=Problem(process_sigma=.04,goal_half=.45,name='moderate_noise');rows=[]
    configs=[]
    for kind in ['chance','cvar']:
        for delta in [.05,.1,.2]:
            for lam in [0,.5,2,8,32]:configs.append(dict(kind=kind,delta=delta,lam=lam,K=256,gamma=30,cost_weight=0.))
    for K in [64,1024]:configs.append(dict(kind='cvar',delta=.05,lam=8,K=K,gamma=30,cost_weight=0.))
    for gamma in [10,100]:configs.append(dict(kind='chance',delta=.1,lam=8,K=256,gamma=gamma,cost_weight=0.))
    for cost in [.03,.1,.3]:
        for delta in [.05,.1,.2]:configs.append(dict(kind='chance',delta=delta,lam=8,K=256,gamma=30,cost_weight=cost))
    for cfg in configs:
        u,meta=sample(model,mean,scale,s,g,p,seed=31001,**cfg)
        ev=evaluate(u.cpu().numpy(),s,g,p,draws=4096,seed=74001,delta=cfg['delta'])
        cost=nominal_cost(u,torch.tensor(s,device=u.device),p).detach().cpu().numpy()
        row=dict(**cfg,mean_risk=float(np.mean([x['risk'] for x in ev])),accepted=sum(x['accepted'] for x in ev),
                 planning_cost=float(cost.mean()),seconds_batch=meta['seconds_batch'],rollouts=meta['guidance_rollouts_per_plan'])
        rows.append(row)
    (out/'results.json').write_text(json.dumps({'problem':p.as_dict(),'task_seed':1701,'sample_seed':31001,'rows':rows},indent=2));print('sweep cases',len(rows))
if __name__=='__main__':main()
