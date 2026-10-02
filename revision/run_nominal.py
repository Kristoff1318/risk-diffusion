import argparse,json
from pathlib import Path
import numpy as np
import torch
from .nominal import train,sample,dataset,metrics,schedule

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=4000);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',default='risk_diffusion/revision/outputs/nominal_seed0');p.add_argument('--strength',type=float,default=.12);p.add_argument('--layout',choices=['circle','staggered'],default='circle');args=p.parse_args()
    torch.set_num_threads(2);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    obstacles=((0.,0.,.43),) if args.layout=='circle' else ((0.,0.,.43),(-.7,.75,.20))
    model,mean,scale,meta=train(seed=args.seed,steps=args.steps,strength=args.strength,obstacles=obstacles)
    torch.save({'model':model.state_dict(),'mean':mean.cpu(),'scale':scale.cpu(),'metadata':meta},out/'nominal.pt')
    s,g,oracle=dataset(512,901,args.strength,obstacles);u=sample(model,mean,scale,s,g,seed=10001)
    report=metrics(u,torch.tensor(s,device='cuda'),torch.tensor(g,device='cuda'),strength=args.strength,obstacles=obstacles)
    report.update(meta);report['alpha_terminal']=schedule()[-1].item()
    report['sampling_seed']=10001;report['task_seed']=901;report['sampling_steps']=64
    report['oracle']=metrics(torch.tensor(oracle),torch.tensor(s),torch.tensor(g),strength=args.strength,obstacles=obstacles)
    report['gate_pass']=bool(report['feasible_fraction']>=.95 and abs(report['upper_route_fraction']-report['oracle']['upper_route_fraction'])<=.10)
    (out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'samples.npz',start=s,goal=g,u=u.cpu().numpy())
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
