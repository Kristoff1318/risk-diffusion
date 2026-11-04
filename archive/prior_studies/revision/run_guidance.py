"""Stages 5–7: paired B0–B3 experiments and risk-error decomposition."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .problem import Problem
from .nominal import dataset
from .guidance import load,sample
from .risk import chance,nominal_cost,sampled_cvar,simulate,hard_loss
from .evaluate import evaluate
from .calibration import decompose

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',default='risk_diffusion/revision/outputs/guidance');args=pa.parse_args()
    torch.set_num_threads(2);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    problems=[Problem(name='low_noise'),Problem(process_sigma=.04,goal_half=.45,name='moderate_noise')]
    rows=[];traces=[];saved={};cal=[]
    s,g,_=dataset(32,2901)
    for trainseed in range(3):
        model,mean,scale=load(f'risk_diffusion/revision/outputs/nominal_seed{trainseed}/nominal.pt')
        for p in problems:
            for delta in (.05,.1,.2):
                for name,kind,lam in [('B0',None,0),('B1','deterministic',8),('B2','chance',8),('B3','cvar',8)]:
                    u,meta=sample(model,mean,scale,s,g,p,kind,delta,lam,seed=40101+trainseed,K=256)
                    raw=u.cpu().numpy();ev=evaluate(raw,s,g,p,draws=4096,seed=75100,delta=delta)
                    # New independent draw batch audits accepted outputs; never used for selection.
                    audit=evaluate(raw,s,g,p,draws=8192,seed=198100,delta=delta)
                    pred=chance(u,torch.tensor(s,device=u.device),torch.tensor(g,device=u.device),p)['risk'].detach().cpu().numpy()
                    cost=nominal_cost(u,torch.tensor(s,device=u.device),p).detach().cpu().numpy()
                    key=f'{p.name}_{trainseed}_{name}_{int(delta*100)}';saved[key]=raw
                    trace=meta.pop('trace');traces.append(dict(key=key,trace=trace))
                    for i,(e,auditrow) in enumerate(zip(ev,audit)):
                        rows.append(dict(method=name,delta=delta,training_seed=trainseed,task=i,problem=p.name,
                                         predicted_risk=float(pred[i]),planning_cost=float(cost[i]),audit_risk=auditrow['risk'],
                                         audit_risk_upper=auditrow['risk_upper'],**meta,**e))
                    if trainseed==0 and delta==.1:
                        for i in range(8):cal.append(dict(method=name,problem=p.name,task=i,**decompose(raw[i],s[i],g[i],p,seed=62100+i)))
                print('finished',trainseed,p.name,delta,flush=True)
    (out/'results.json').write_text(json.dumps({'problems':[p.as_dict() for p in problems],'rows':rows,'calibration':cal,'traces':traces,
        'protocol':{'tasks':32,'training_seeds':3,'diffusion_steps':64,'cvar_K':256,'acceptance_draws':4096,'audit_draws':8192,
                    'acceptance_confidence':'95% one-sided per-plan, not simultaneous','validation_seed':75100,'audit_seed':198100}},indent=2))
    np.savez_compressed(out/'plans.npz',start=s,goal=g,**saved)
if __name__=='__main__':main()
