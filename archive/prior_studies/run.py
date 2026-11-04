"""Reproduce training, ablation, calibration, and figures.

Usage: python -m risk_diffusion.run --output risk_diffusion/outputs
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import core
from . import evaluator
from .evaluator import evaluate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='risk_diffusion/outputs')
    p.add_argument('--train-n',type=int,default=2500); p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--draws',type=int,default=2048); p.add_argument('--tasks',type=int,default=12)
    p.add_argument('--steps',type=int,default=25); p.add_argument('--sample-seeds',type=int,default=3)
    p.add_argument('--radius',type=float,default=.43); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--quick',action='store_true')
    args=p.parse_args(); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    core.RADIUS=args.radius; evaluator.RADIUS=args.radius
    torch.set_num_threads(2)
    if args.quick:
        args.train_n=500;args.epochs=6;args.draws=512;args.tasks=4;args.steps=12;args.sample_seeds=1
    starts,goals,controls=core.data(args.train_n,args.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    t=time.perf_counter();model,mean,scale,losses=core.train(starts,goals,controls,epochs=args.epochs,seed=args.seed,device=device)
    torch.save({'model':model.state_dict(),'mean':mean.cpu(),'scale':scale.cpu(),'epochs':args.epochs},out/'nominal.pt')
    training_seconds=time.perf_counter()-t
    rng=np.random.default_rng(123+args.seed)
    s=np.column_stack((-1.5*np.ones(args.tasks),rng.uniform(-.18,.18,args.tasks))).astype('float32')
    g=np.column_stack((1.5*np.ones(args.tasks),rng.uniform(-.18,.18,args.tasks))).astype('float32')
    methods=[('B0',None,0,False,None),('B1','sdf',.7,False,None),
             ('B2','chance',.45,False,None),('B3','cvar',.16,False,None),
             ('B4',None,0,True,None),('B5','chance',.45,True,None)]
    rows=[]; plans={}
    for delta in (.05,.10,.20):
        for sample_seed in range(args.sample_seeds):
            for name,kind,lam,projection,condition in methods:
                u,meta=core.sample(model,mean,scale,s,g,kind,delta,lam,args.steps,projection,condition,
                                   seed=71+sample_seed,K=32,projection_steps=8)
                un=u.cpu().numpy();plans[(name,delta,sample_seed)]=un
                ev=evaluate(un,s,g,args.draws,seed=400+sample_seed,delta=delta)
                pred=core.chance_risk(u,torch.tensor(s,device=device),torch.tensor(g,device=device)).detach().cpu().numpy()
                nominal=core.rollout(u,torch.tensor(s,device=device))[0].detach().cpu().numpy()
                for i,e in enumerate(ev):
                    rows.append(dict(method=name,delta=delta,task=i,sample_seed=sample_seed,
                                     nominal_feasible=bool(np.all(np.linalg.norm(nominal[i,1:],axis=1)>args.radius)
                                                           and np.all(np.abs(nominal[i,-1]-g[i])<=core.GOAL_HALF)),
                                     predicted_risk=float(pred[i]),**meta,**e))
        print('finished delta',delta,flush=True)
    # Calibration: independent nominal and perturbation trajectories, not reused guidance samples.
    cs,cg,cu=core.data(100,891+args.seed,height_range=(.36,1.18))
    cr=evaluate(cu,cs,cg,args.draws,seed=991)
    cp=core.chance_risk(torch.tensor(cu,device=device),torch.tensor(cs,device=device),torch.tensor(cg,device=device)).detach().cpu().numpy()
    calibration=[{'predicted':float(a),'actual':b['risk'],'task':i} for i,(a,b) in enumerate(zip(cp,cr))]
    # Risk labels on ordinary and perturbed paths; train conditional after inference pilots.
    label_n=min(args.train_n,800 if not args.quick else 200)
    ps,pg,pu=core.data(label_n,77+args.seed,height_range=(.38,.9))
    ls=np.concatenate((starts[:label_n],ps));lg=np.concatenate((goals[:label_n],pg))
    lu=np.concatenate((controls[:label_n],pu))
    rr=evaluate(lu,ls,lg,256 if not args.quick else 64,seed=177)
    labels=np.array([e['risk'] for e in rr],dtype='float32')
    conditional,cmean,cscale,closs=core.train(ls,lg,lu,
            conditional=True,risks=labels,epochs=max(8,args.epochs//2),seed=5+args.seed,device=device)
    torch.save({'model':conditional.state_dict(),'mean':cmean.cpu(),'scale':cscale.cpu(),'risk_labels':labels},out/'conditional.pt')
    for delta in (.05,.10,.20):
        for sample_seed in range(args.sample_seeds):
            for name,kind,lam in [('B6',None,0),('B7','chance',.45)]:
                u,meta=core.sample(conditional,cmean,cscale,s,g,kind,delta,lam,args.steps,False,
                        delta,seed=71+sample_seed,K=32)
                plans[(name,delta,sample_seed)]=u.cpu().numpy()
                ev=evaluate(u.cpu().numpy(),s,g,args.draws,seed=400+sample_seed,delta=delta)
                pred=core.chance_risk(u,torch.tensor(s,device=device),torch.tensor(g,device=device)).detach().cpu().numpy()
                nominal=core.rollout(u,torch.tensor(s,device=device))[0].detach().cpu().numpy()
                for i,e in enumerate(ev):rows.append(dict(method=name,delta=delta,task=i,sample_seed=sample_seed,
                                                            nominal_feasible=bool(np.all(np.linalg.norm(nominal[i,1:],axis=1)>args.radius)
                                                                                  and np.all(np.abs(nominal[i,-1]-g[i])<=core.GOAL_HALF)),
                                                            predicted_risk=float(pred[i]),**meta,**e))
    # Guidance strength sweep, held at 0.10 budget.
    sweep=[]
    for kind,weights in [('chance',[0,.01,.03,.1,.3,.9]),('cvar',[0,.06,.16,.32])]:
        for lam in weights:
            u,meta=core.sample(model,mean,scale,s,g,kind,.1,lam,args.steps,False,None,seed=171,K=32)
            ev=evaluate(u.cpu().numpy(),s,g,args.draws,seed=144,delta=.1)
            sweep.append({'kind':kind,'lambda':lam,**meta,'mean_risk':float(np.mean([x['risk'] for x in ev])),
                          'mean_cost':float(np.mean([x['cost'] for x in ev]))})
    sensitivity=[]
    for K in (16,32,128):
        u,meta=core.sample(model,mean,scale,s,g,'cvar',.1,.32,args.steps,seed=171,K=K)
        ev=evaluate(u.cpu().numpy(),s,g,args.draws,seed=144,delta=.1)
        sensitivity.append({'parameter':'K','value':K,'mean_risk':float(np.mean([x['risk'] for x in ev])),**meta})
    for gamma in (5,15,40):
        u,meta=core.sample(model,mean,scale,s,g,'chance',.1,.1,args.steps,seed=171,gamma=gamma)
        ev=evaluate(u.cpu().numpy(),s,g,args.draws,seed=144,delta=.1)
        sensitivity.append({'parameter':'gamma','value':gamma,'mean_risk':float(np.mean([x['risk'] for x in ev])),**meta})
    diversity={}
    for name in [m[0] for m in methods]+['B6','B7']:
        x=np.stack([plans[(name,.1,j)] for j in range(args.sample_seeds)])
        if args.sample_seeds>1:
            pairs=[np.linalg.norm(x[i]-x[j],axis=(1,2)).mean() for i in range(args.sample_seeds) for j in range(i+1,args.sample_seeds)]
            diversity[name]=float(np.mean(pairs))
        else: diversity[name]=None
    result={'config':vars(args),'device':device,'train_seconds':training_seconds,'train_last_loss':losses[-1],
            'conditional_last_loss':closs[-1],'risk_label_range':[float(labels.min()),float(labels.max())],
            'rows':rows,'calibration':calibration,'sweep':sweep,'sensitivity':sensitivity,'diversity':diversity}
    (out/'results.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/'plans.npz',**{f'{name}_{int(delta*100)}_{seed}':u for (name,delta,seed),u in plans.items()})
    from .plots import make_plots
    make_plots(result,out)
    print('saved',out/'results.json',flush=True)

if __name__=='__main__':main()
