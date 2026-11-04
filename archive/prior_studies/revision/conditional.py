"""Matched-budget conditioning experiment after inference guidance evaluation."""
import copy,json,time,argparse
from pathlib import Path
import numpy as np
import torch
from .nominal import dataset,Denoiser,schedule
from .problem import Problem
from .evaluate import evaluate
from .guidance import load,sample
from .risk import chance,nominal_cost

def train_pair_data(s,g,u,budgets,conditional,seed,steps=4000):
    torch.manual_seed(seed);dev='cuda';st=torch.tensor(s,device=dev);gt=torch.tensor(g,device=dev);ut=torch.tensor(u,device=dev)
    mean=ut.mean(0,keepdim=True);scale=ut.std(0,keepdim=True).clamp_min(.1);y=(ut-mean)/scale
    model=Denoiser(extra_condition=2).to(dev);ema=copy.deepcopy(model).eval();opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    a=schedule().to(dev);labels=torch.tensor(budgets,dtype=torch.float32,device=dev);n=len(y)
    dropgen=torch.Generator(device=dev);dropgen.manual_seed(30000+seed)
    began=time.perf_counter()
    for step in range(steps):
        ix=torch.randint(n,(256,),device=dev);t=torch.randint(len(a),(256,),device=dev);noise=torch.randn_like(y[ix]);at=a[t,None,None]
        z=at.sqrt()*y[ix]+(1-at).sqrt()*noise;target=at.sqrt()*noise-(1-at).sqrt()*y[ix]
        extra=torch.zeros((len(ix),2),device=dev)
        if conditional:
            present=(torch.rand(len(ix),device=dev,generator=dropgen)>=.15).float();extra[:,0]=labels[ix]*present;extra[:,1]=present
        loss=(model(z,t.float()/(len(a)-1),st[ix],gt[ix],extra)-target).square().mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        with torch.no_grad():
            for e,p in zip(ema.parameters(),model.parameters()):e.mul_(.995).add_(p,alpha=.005)
    torch.cuda.synchronize()
    return ema,mean,scale,{'steps':steps,'train_n':n,'seed':seed,'conditional':conditional,'seconds':time.perf_counter()-began,'last_loss':float(loss)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ordinary-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);out=Path('risk_diffusion/revision/outputs/conditional_ordinary' if args.ordinary_only else 'risk_diffusion/revision/outputs/conditional');out.mkdir(parents=True,exist_ok=True)
    p=Problem(process_sigma=.04,goal_half=.45,name='moderate_noise')
    ordinary=dataset(2000,19001);marginal=dataset(2000,19002,height_range=(.58,1.15) if args.ordinary_only else (.445,.80))
    s,g,u=[np.concatenate((a,b)) for a,b in zip(ordinary,marginal)]
    labeled=evaluate(u,s,g,p,draws=512,seed=120001)
    upper=np.array([x['risk_upper'] for x in labeled]);measured=np.array([x['risk'] for x in labeled])
    # Each example receives a requested budget under which its label is admissible.
    # Label uncertainty is not interpreted as a formal training-time guarantee.
    levels=np.array([.05,.1,.2,.5,.9,1.]);rng=np.random.default_rng(39100)
    budgets=np.array([rng.choice(levels[levels>=r]) for r in upper],dtype='float32')
    np.savez_compressed(out/'labeled_data.npz',start=s,goal=g,u=u,measured_risk=measured,risk_upper=upper,budget=budgets)
    starts,goals,_=dataset(32,2901);rows=[];metadata=[];saved={}
    # Both models have identical dimensions, examples, optimizer steps and seeded batches/noise.
    for seed in range(3):
        for conditional in [False,True]:
            model,mean,scale,meta=train_pair_data(s,g,u,budgets,conditional,seed);metadata.append(meta)
            path=out/f'{"conditional" if conditional else "matched_nominal"}_{seed}.pt'
            torch.save({'model':model.state_dict(),'mean':mean.cpu(),'scale':scale.cpu(),'extra_condition':2,'metadata':meta},path)
            for param in model.parameters():param.requires_grad_(False)
            for delta in [.05,.1,.2]:
                for guided in [False,True]:
                    name=('B7' if guided else 'B6') if conditional else ('MatchedNominalChance' if guided else 'MatchedNominal')
                    proposal,info=sample(model,mean,scale,starts,goals,p,'chance' if guided else None,delta,8,
                                         seed=40101+seed,condition_budget=delta if conditional else None)
                    raw=proposal.cpu().numpy();ev=evaluate(raw,starts,goals,p,4096,75100,delta);audit=evaluate(raw,starts,goals,p,8192,198100,delta)
                    pred=chance(proposal,torch.tensor(starts,device='cuda'),torch.tensor(goals,device='cuda'),p)['risk'].detach().cpu().numpy()
                    cost=nominal_cost(proposal,torch.tensor(starts,device='cuda'),p).detach().cpu().numpy();info.pop('trace')
                    for i,(e,auditrow) in enumerate(zip(ev,audit)):rows.append(dict(method=name,training_seed=seed,task=i,delta=delta,
                        predicted_risk=float(pred[i]),planning_cost=float(cost[i]),audit_risk=auditrow['risk'],**info,**e))
                    saved[f'{name}_{seed}_{int(delta*100)}']=raw
            print('finished training variant',seed,conditional,flush=True)
    report={'data_mode':'ordinary_only' if args.ordinary_only else 'ordinary_plus_marginal','problem':p.as_dict(),'training':metadata,'rows':rows,'label_range':[float(measured.min()),float(measured.max())],
            'ordinary_label_range':[float(measured[:2000].min()),float(measured[:2000].max())],
            'budget_counts':{str(l):int((budgets==l).sum()) for l in levels},
            'conditioning':'requested upper budget chosen from levels >= independent 512-draw risk upper label; explicit mask; CFG scale=1'}
    (out/'results.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'plans.npz',start=starts,goal=goals,**saved)
if __name__=='__main__':main()
