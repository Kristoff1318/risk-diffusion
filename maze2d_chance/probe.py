import sys,os,pickle,json
sys.path.insert(0,'/home/kristoff/Documents/rb-diffuser')
import numpy as np, torch, gym,d4rl
from diffuser.models.helpers import apply_conditioning
from pathlib import Path
ROOT=Path('/home/kristoff/Documents/rb-diffuser/logs/maze2d-large-v1/diffusion/H384_T256')
torch.set_num_threads(4)
def load():
 cfg=lambda n:pickle.load(open(ROOT/(n+'_config.pkl'),'rb'))
 dataset=cfg('dataset')(); model=cfg('model')(); d=cfg('diffusion')(model)
 epoch=max(int(p.stem.split('_')[1]) for p in ROOT.glob('state_*.pt'))
 ck=torch.load(ROOT/f'state_{epoch}.pt'); d.load_state_dict(ck['ema']); d.eval()
 return d,dataset.normalizer,epoch
if __name__=='__main__':
 e=gym.make('maze2d-large-v1').unwrapped; e.reset(); print('mass',e.model.body_mass,'body pos',e.model.body_pos,flush=True)
 d,n,epoch=load(); print('normalizers',n,flush=True)
 np.savez('maze2d_chance/normalization.npz',**{k+'_'+a:getattr(n.normalizers[k],a) for k in ['observations','actions'] for a in ['mins','maxs']})
 starts=[([1.,1.,0.,0.],[7.,9.]),([1.,1.,0.,0.],[3.,1.]),([3.,1.,0.,0.],[5.,1.])]
 out=[]
 for start,goal in starts:
  cond={0:torch.tensor(n.normalize(np.array(start),'observations'),device='cuda').float()[None],383:torch.tensor(n.normalize(np.r_[goal,0,0],'observations'),device='cuda').float()[None]}
  torch.manual_seed(11)
  with torch.no_grad(): sample=d(cond,verbose=False).cpu().numpy()[0]
  obs=n.unnormalize(sample[:,2:],'observations'); actions=n.unnormalize(sample[:,:2],'actions')
  for mode in ['actions','tracking']:
   e.reset();e.set_state(np.array(start[:2]),np.array(start[2:]));e.set_target(np.array(goal)); hist=[e._get_obs()]; contacts=0
   for t in range(383):
    s=e._get_obs();a=actions[t] if mode=='actions' else obs[t+1,:2]-s[:2]+obs[t+1,2:]-s[2:]
    s,_,_,_=e.step(a);hist.append(s);contacts+=sum('wall' in (e.model.geom_id2name(int(c.geom1)) or '') or 'wall' in (e.model.geom_id2name(int(c.geom2)) or '') for c in e.sim.data.contact)
   r=dict(start=start,goal=goal,mode=mode,final=np.linalg.norm(s[:2]-goal),contacts=contacts); print(r,flush=True);out.append(r)
  np.savez('maze2d_chance/probe_'+str(goal)+'.npz',obs=obs,actions=actions)
 json.dump(out,open('maze2d_chance/probe.json','w'),indent=2)
