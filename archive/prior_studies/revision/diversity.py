"""Within-condition diversity: 32 independent diffusion draws, five fixed tasks."""
import json
from pathlib import Path
import numpy as np
import torch
from .nominal import dataset
from .guidance import load,sample
from .problem import Problem
from .risk import simulate

def main():
    torch.set_num_threads(2);root=Path('risk_diffusion/revision/outputs');p=Problem(process_sigma=.04,goal_half=.45)
    s,g,_=dataset(5,23001);ss=np.repeat(s,32,axis=0);gg=np.repeat(g,32,axis=0);rows=[]
    for seed in range(3):
        model,mean,scale=load(str(root/f'nominal_seed{seed}/nominal.pt'))
        for name,kind in [('B0',None),('B2','chance'),('B3','cvar')]:
            u,meta=sample(model,mean,scale,ss,gg,p,kind,.1,8,seed=45101+seed)
            paths=simulate(u,torch.tensor(ss,device=u.device),p)[0].cpu().numpy().reshape(5,32,21,2)
            for i,paths_i in enumerate(paths):
                distance=np.sqrt(np.mean(np.sum((paths_i[:,None]-paths_i[None,:])**2,axis=-1),axis=-1))
                rows.append(dict(method=name,training_seed=seed,task=i,upper_route_fraction=float(np.mean(paths_i[:,10,1]>0)),
                                 pairwise_path_rms=float(distance[np.triu_indices(32,1)].mean())))
    result={'protocol':'Five new fixed start/goal conditions; 32 independent diffusion draws each; three checkpoints; Delta=.1; all proposals; nominal state-path RMS in coordinate units.', 'rows':rows}
    (root/'diversity.json').write_text(json.dumps(result,indent=2))
    for name in ['B0','B2','B3']:
        rr=[x for x in rows if x['method']==name];print(name,'upper',np.mean([x['upper_route_fraction'] for x in rr]),'RMS',np.mean([x['pairwise_path_rms'] for x in rr]))
if __name__=='__main__':main()
