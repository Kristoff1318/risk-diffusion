"""Common frozen planning representation, whole-arm risk, and physical cost."""
import pathlib,sys,time,json,math,numpy as np,torch
from scipy.interpolate import BSpline
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent/'phase6'));import tracking as tracking
p0=tracking.p0
from torch.utils.checkpoint import checkpoint
DT=.05;H=200;KGAIN=2.;ARMS=['classical','classical_cvar','mpd','mpd_cvar']
class Core(tracking.GPUArm):
 def __init__(self):
  self.arm=p0.Arm();super().__init__(self.arm)
  knots=np.pad(np.linspace(0,1,25),5,'edge');basis=BSpline(knots,np.eye(29),5)(np.linspace(0,1,201));reduced=np.column_stack((basis[:,3:26],basis[:,26:].sum(1)));self.B=self.cast(reduced);self.C=self.cast(basis[:,:3].sum(1));self.vmax=self.cast(self.arm.vmax)
 def nom(self,x,start):return self.B@x+self.C[:,None]*start
 def points(self,poses,kind):
  groups,rr,ii,jj,th=self.models[kind];pts=self.t.empty((len(poses['panda_hand'][1]),len(rr),3),**self.kw)
  for link,idx,xyz in groups:
   R,pos=poses[link];pts[:,idx]=torch.einsum('nij,sj->nsi',R,xyz)+pos[:,None]
  return pts
 def margins_tensor(self,q):
  out=[]
  for kind,(_,rr,ii,jj,th) in enumerate(self.models):
   pts=self.points(self.fk(q,kind),kind);wm=(torch.linalg.vector_norm(pts[:,:,None]-self.centers[None,None],dim=-1)-rr[None,:,None]-self.radii).flatten(1).amin(1)
   sm=torch.full((len(q),),float('inf'),**self.kw)
   for a in range(0,len(ii),64):sm=torch.minimum(sm,(torch.linalg.vector_norm(pts[:,ii[a:a+64]]-pts[:,jj[a:a+64]],dim=-1)-th[a:a+64]).amin(1))
   out.extend((wm,sm))
  # Same order as immutable P0 hard event validator.
  return torch.stack((out[0],out[2],out[1],out[3],torch.minimum(q-self.lo,self.hi-q).amin(1)),1)
 def pose_errors(self,q,target):
  R,pos=self.fk(q,0)['panda_hand'];pe=torch.linalg.vector_norm(pos-target[:,-1],dim=-1);rel=R.transpose(-1,-2)@target[:,:3];c=((rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1);vee=torch.stack((rel[:,2,1]-rel[:,1,2],rel[:,0,2]-rel[:,2,0],rel[:,1,0]-rel[:,0,1]),-1)/2;ang=torch.atan2(torch.linalg.vector_norm(vee,dim=-1),c)
  return pe,ang
 def cost(self,nom):
  vel=nom.diff(dim=-2)/DT;acc=vel.diff(dim=-2)/DT;return DT*vel.square().sum((-1,-2))+.005*DT*acc.square().sum((-1,-2))
 def nominal_objective(self,x,start,target):
  nom=self.nom(x,start);m=self.margins_tensor(nom);pe,ang=self.pose_errors(nom[-1:],target);vel=nom.diff(dim=0)/DT
  collision=torch.relu(.001-m[:,:4]).square().sum()/201/.01**2;joint=torch.relu(-m[:,4]).square().mean()/.02**2;velocity=torch.relu(vel.abs()-self.vmax).square().mean()/.1**2
  return self.cost(nom)+500*(collision+joint+velocity)+100*(pe/.01).square().sum()+10*(ang/math.radians(2)).square().sum()
 @staticmethod
 def tail(loss,delta):
  values=loss.sort(descending=True).values;t=delta*len(loss);m=int(math.floor(t));f=t-m;return (values[:m].sum()+f*values[m])/t if m<len(loss) else values.mean()
 def risk_losses(self,nom,target,errors,checkpointed=True):
  e,velerr,commanderr=errors;sub=10;n=len(e);ff=nom.diff(dim=0)/DT;fractions=torch.arange(sub,**self.kw)/sub
  nd=(nom[:-1,None]+DT*fractions[None,:,None]*ff[:,None]).reshape(-1,7);nd=torch.cat((nd,nom[-1:]),0);loss=[]
  for i in range(n):
   q=nd+e[i];m=checkpoint(self.margins_tensor,q,use_reentrant=False) if checkpointed and q.requires_grad else self.margins_tensor(q)
   geom=torch.cat((-m[:,:4]/.01,-m[:,4:]/.02),1).amax();cv=((ff+commanderr[i]).abs()-self.vmax).amax()/.1;ev=((ff+velerr[i]).abs()-self.vmax).amax()/.1;pe,ang=self.pose_errors(q[-1:],target);goal=torch.maximum(pe/.01-1,ang/math.radians(2)-1).amax();loss.append(torch.stack((geom,cv,ev,goal)).amax())
  return torch.stack(loss)

def errors(n,seed,scale=1.,dense=True):
 draw=p0.noise(n,seed,scale);q,v,u=tracking.track(np.zeros((201,7)),draw);return (p0.dense(q,v,10) if dense else q,v,u)
def audit(core,nom,target,err,chunk_rollouts=8):
 """Hard5ms complete events; independent pose implementation uses SciPy."""
 from scipy.spatial.transform import Rotation
 begin=time.perf_counter();nd=nom.detach().cpu().numpy();ff=np.diff(nd,axis=0)/DT;fractions=np.arange(10)/10;dense_nom=np.concatenate(((nd[:-1,None]+DT*fractions[None,:,None]*ff[:,None]).reshape(-1,7),nd[-1:]),0);e,v,u=err;n=len(e);mins=np.empty((n,5))
 with torch.no_grad():
  for a in range(0,n,chunk_rollouts):
   q=core.cast(dense_nom[None]+e[a:a+chunk_rollouts]);m=core.margins_tensor(q.reshape(-1,7)).reshape(len(q),2001,5).amin(1);mins[a:a+len(q)]=m.cpu().numpy()
  qe=nd[-1]+e[:,-1];R,pos=core.arm.fk(qe)['panda_hand'];tg=target.detach().cpu().numpy();pe=np.linalg.norm(pos-tg[:,3],axis=-1);ang=Rotation.from_matrix(R.transpose(0,2,1)@tg[:,:3]).magnitude();flags=np.column_stack((mins<0,(abs(ff[None]+u)>core.arm.vmax).any((1,2)),(abs(ff[None]+v)>core.arm.vmax).any((1,2)),pe>.01,ang>math.radians(2)));count=int(flags.any(1).sum())
 row={'n':n,'union':count,'risk':count/n,'CI95':p0.interval(count,n),'counts':dict(zip(p0.EVENTS,flags.sum(0).astype(int).tolist())),'minimum_margins':mins.min(0).tolist(),'max_position':float(pe.max()),'max_orientation':float(ang.max()),'seconds':time.perf_counter()-begin,'finite':bool(np.isfinite(nd).all() and np.isfinite(mins).all())}
 return row,flags

def nominal_check(core,nom,target):
 with torch.no_grad():
  nd=nom.detach().cpu().numpy();zeros=(np.zeros((1,2001,7)),np.zeros((1,200,7)),np.zeros((1,200,7)));r,f=audit(core,nom,target,zeros);r['cost']=float(core.cost(nom));return r

def refine(core,x,start,target,*,steps,lr,delta=None,err=None,wallcap=1000):
 x=x.detach().clone().requires_grad_(True);opt=torch.optim.Adam([x],lr=lr);trace=[];beg=time.perf_counter();timedout=False
 for i in range(steps):
  if time.perf_counter()-beg>=wallcap:timedout=True;break
  opt.zero_grad();nom=core.nom(x,start)
  if delta is None:loss=core.nominal_objective(x,start,target);tail=None
  else:
   lossvec=core.risk_losses(nom,target,err);tail=core.tail(lossvec,delta);loss=core.cost(nom)+500*torch.relu(tail+.05).square()
  if not torch.isfinite(loss):trace.append({'iteration':i,'nonfinite_loss':True});break
  loss.backward();norm=float(x.grad.norm());torch.nn.utils.clip_grad_norm_([x],20.);opt.step();torch.cuda.synchronize();trace.append({'iteration':i,'objective':float(loss.detach()),'cost':float(core.cost(nom).detach()),'cvar':float(tail.detach()) if tail is not None else None,'grad_norm':norm,'seconds':time.perf_counter()-beg})
 return x.detach(),{'trace':trace,'seconds':time.perf_counter()-beg,'timeout':timedout,'iterations':len(trace),'gradient_calls':len(trace)}

def classical(core,start,target,seed,steps=80):
 beg=time.perf_counter();gen=torch.Generator(device='cuda').manual_seed(seed);seeds=torch.stack((start,core.cast([0,-1.3,0,-2.5,0,1,0])));ends=[];iktrace=[]
 for seedq in seeds:
  q=seedq.detach().clone().requires_grad_(True);opt=torch.optim.Adam([q],lr=.04)
  for k in range(80):
   opt.zero_grad();pe,ang=core.pose_errors(q[None],target);loss=(pe/.01).square().sum()+(ang/math.radians(2)).square().sum();loss.backward();opt.step()
   with torch.no_grad():q.clamp_(core.lo,core.hi)
  ends.append(q.detach());iktrace.append({'position':float(pe.detach()),'orientation':float(ang.detach()),'steps':80})
 candidates=[];logs=[];phase=torch.linspace(0,1,24,**core.kw)
 for i,end in enumerate(ends):
  x=start[None]+phase[:,None]*(end-start)[None]
  if i:x=x+.25*torch.sin(math.pi*phase)[:,None]*torch.randn((1,7),device='cuda',dtype=torch.float64,generator=gen)
  x,log=refine(core,x,start,target,steps=steps,lr=.025);candidates.append(x);logs.append(log)
 return torch.stack(candidates),{'seconds':time.perf_counter()-beg,'IK':iktrace,'nominal_refinements':logs,'multi_starts':2,'reference_IK_not_used':True}

def load_mpd():
 base=p0.BASE;sys.path[:0]=[str(base),str(base/'scripts')];from adapters.native_runtime import model_and_assets,context
 model,norm,_,meta=model_and_assets('cuda');return model,norm,context,meta

def mpd(model,norm,context,queries,i,seed):
 beg=time.perf_counter();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 with torch.no_grad():x=model.run_inference(context_d=context(queries,i,norm),hard_conds={},horizon=24,n_samples=2,method='ddim',ddim_sampling_timesteps=15,ddim_eta=.5,guide=None);x=norm['control_points'].unnormalize(x).double()
 torch.cuda.synchronize();return x,{'seconds':time.perf_counter()-beg,'denoiser_calls':15,'n_samples':2,'seed':seed,'guide':False}
