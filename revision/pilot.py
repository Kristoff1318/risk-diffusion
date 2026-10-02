import json
from pathlib import Path
import numpy as np
import torch
from .problem import Problem
from .nominal import dataset
from .guidance import load,sample
from .risk import chance,nominal_cost
from .evaluate import evaluate

def main():
    torch.set_num_threads(2);out=Path('risk_diffusion/revision/outputs/guidance_pilot');out.mkdir(parents=True,exist_ok=True)
    model,mean,scale=load('risk_diffusion/revision/outputs/nominal_seed0/nominal.pt')
    s,g,_=dataset(24,1701);p=Problem(process_sigma=.04,goal_half=.45,name='circle_moderate_noise')
    rows=[];plans={}
    for kind,weights in [(None,[0]),('deterministic',[1,4]),('chance',[.5,2,8]),('cvar',[.5,2,8])]:
        for lam in weights:
            u,meta=sample(model,mean,scale,s,g,p,kind,.1,lam,seed=31001,K=256)
            ev=evaluate(u.cpu().numpy(),s,g,p,draws=4096,seed=74001,delta=.1)
            pred=chance(u,torch.tensor(s,device='cuda'),torch.tensor(g,device='cuda'),p)['risk'].detach().cpu().numpy()
            cost=nominal_cost(u,torch.tensor(s,device='cuda'),p).detach().cpu().numpy()
            for i,e in enumerate(ev):rows.append(dict(method=str(kind),lam=lam,task=i,predicted_risk=float(pred[i]),planning_cost=float(cost[i]),**e))
            plans[f'{kind}_{lam}']=u.cpu().numpy()
            print(kind,lam,'risk',np.mean([x['risk'] for x in ev]),'accepted',sum(x['accepted'] for x in ev),'time',meta['seconds_batch'],flush=True)
    (out/'results.json').write_text(json.dumps({'problem':p.as_dict(),'rows':rows},indent=2));np.savez_compressed(out/'plans.npz',start=s,goal=g,**plans)
if __name__=='__main__':main()
