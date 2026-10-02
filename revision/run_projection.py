"""Stage 8 after nominal and guidance gates; actual constrained projection."""
import json
from pathlib import Path
import numpy as np
import torch
from .problem import Problem
from .nominal import dataset
from .guidance import load,sample
from .risk import chance,nominal_cost
from .evaluate import evaluate
from .calibration import decompose

def main():
    torch.set_num_threads(2);out=Path('risk_diffusion/revision/outputs/projection');out.mkdir(parents=True,exist_ok=True)
    base='risk_diffusion/revision/outputs/'
    configs=[(Problem(process_sigma=.04,goal_half=.45,name='moderate_noise'),base+'nominal_seed0/nominal.pt'),
             (Problem(process_sigma=.04,goal_half=.45,obstacles=((0.,0.,.43),(-.7,.75,.20)),name='staggered'),base+'staggered_seed0/nominal.pt'),
             (Problem(strength=.35,process_sigma=.04,goal_half=.55,name='nonlinear'),base+'nonlinear_seed0/nominal.pt')]
    s,g,_=dataset(32,2901);rows=[];plans={};cal=[]
    methods=[('B0',None,0,False,False),('B1','deterministic',8,False,False),('B2','chance',8,False,False),
             ('B3','cvar',8,False,False),('B4',None,0,True,False),('B5','chance',8,True,False),('FinalProjection',None,0,True,True)]
    for p,path in configs:
        model,mean,scale=load(path)
        for delta in (.05,.1,.2):
            for name,kind,lam,proj,final in methods:
                u,meta=sample(model,mean,scale,s,g,p,kind,delta,lam,seed=40101,K=256,projection=proj,final_only=final)
                raw=u.cpu().numpy();ev=evaluate(raw,s,g,p,4096,75100,delta);audit=evaluate(raw,s,g,p,8192,198100,delta)
                pred=chance(u,torch.tensor(s,device=u.device),torch.tensor(g,device=u.device),p)['risk'].detach().cpu().numpy()
                cost=nominal_cost(u,torch.tensor(s,device=u.device),p).detach().cpu().numpy();meta.pop('trace')
                for i,(e,a) in enumerate(zip(ev,audit)):rows.append(dict(problem=p.name,method=name,delta=delta,task=i,training_seed=0,
                    predicted_risk=float(pred[i]),planning_cost=float(cost[i]),audit_risk=a['risk'],**meta,**e))
                plans[f'{p.name}_{name}_{int(delta*100)}']=raw
                if delta==.1 and name in ('B0','B2','B3'):
                    for i in range(8):cal.append(dict(problem=p.name,method=name,task=i,**decompose(raw[i],s[i],g[i],p,seed=62100+i)))
                print(p.name,delta,name,'risk',round(np.mean([x['risk'] for x in audit]),4),'accepted',sum(x['accepted'] for x in ev),'proj failures',meta['projection_failed_calls'],flush=True)
    (out/'results.json').write_text(json.dumps({'problems':[p.as_dict() for p,_ in configs],'rows':rows,'calibration':cal},indent=2))
    np.savez_compressed(out/'plans.npz',start=s,goal=g,**plans)
if __name__=='__main__':main()
