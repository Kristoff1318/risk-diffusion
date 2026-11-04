"""Closed-loop P0 follow-up; preserve phase5, same assets/noise/targets."""
import pathlib,sys,time,json,hashlib,resource
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import beta
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent/'phase5'));import panda_p0 as p0
DT=p0.DT;H=p0.H;TAU=10*DT;GAIN=1/TAU;A=1-DT*GAIN

def save(name,v): (P/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def track(nom,draws,gain=GAIN):
 e,b,eta=draws;n=len(e);q=np.empty((n,H+1,7));v=np.empty((n,H,7));u=np.empty_like(v);q[:,0]=nom[0]+e;ff=np.diff(nom,axis=0)/DT
 for k in range(H):
  u[:,k]=ff[k]-gain*(q[:,k]-nom[k]);v[:,k]=u[:,k]+b+eta[:,k];q[:,k+1]=q[:,k]+DT*v[:,k]
 return q,v,u

def covariance(k,l):
 m=min(k,l);c=lambda t:(1-A**t)/GAIN
 return A**(k+l)*.002**2+c(k)*c(l)*.001**2+DT**2*A**abs(l-k)*(1-A**(2*m))/(1-A**2)*.005**2

class GPUArm:
 """Float64 Torch FK and complete distance sets; no extensions or model loads."""
 def __init__(self,arm):
  import torch
  self.t=torch;self.arm=arm;self.kw={'dtype':torch.float64,'device':'cuda'};self.cast=lambda v:torch.as_tensor(v,**self.kw)
  self.trees=[]
  for tree in arm.trees:self.trees.append([(pa,ch,self.cast(R),self.cast(xyz),idx,typ,val,self.cast(ax)) for pa,ch,R,xyz,idx,typ,val,ax in tree])
  self.centers=self.cast(arm.centers);self.radii=self.cast(arm.radii);self.lo=self.cast(arm.lo);self.hi=self.cast(arm.hi)
  self.models=[]
  for kind,spheres in enumerate(arm.spheres):
   links=list(dict.fromkeys(s[0] for s in spheres));groups=[(link,torch.tensor([i for i,s in enumerate(spheres) if s[0]==link],device='cuda'),self.cast([s[1] for s in spheres if s[0]==link])) for link in links]
   pairs=arm.pairs[kind];self.models.append((groups,self.cast([s[2] for s in spheres]),torch.tensor(pairs[:,0],dtype=torch.long,device='cuda'),torch.tensor(pairs[:,1],dtype=torch.long,device='cuda'),self.cast(pairs[:,2])))
 def fk(self,q,kind):
  t=self.t;n=len(q);poses={'panda_link0':(t.eye(3,**self.kw).expand(n,3,3),t.zeros((n,3),**self.kw))};pending=self.trees[kind].copy()
  while pending:
   progress=False
   for j in pending[:]:
    pa,ch,R0,xyz,idx,typ,val,axis=j
    if pa not in poses:continue
    Rp,tp=poses[pa];R=Rp@R0;pos=tp+(Rp@xyz); 
    if idx is not None:
     c=t.cos(q[:,idx]);s=t.sin(q[:,idx]);rz=t.zeros((n,3,3),**self.kw);rz[:,0,0]=c;rz[:,0,1]=-s;rz[:,1,0]=s;rz[:,1,1]=c;rz[:,2,2]=1;R=R@rz
    elif typ=='prismatic':pos=pos+R@(axis*val)
    poses[ch]=(R,pos);pending.remove(j);progress=True
   if not progress:raise RuntimeError('Disconnected')
  return poses
 def margins(self,q):
  t=self.t;q=self.cast(q);n=len(q);out=t.empty((n,5),**self.kw)
  for kind,(groups,rr,ii,jj,th) in enumerate(self.models):
   poses=self.fk(q,kind);pts=t.empty((n,len(rr),3),**self.kw)
   for link,idx,xyz in groups:
    R,pos=poses[link];pts[:,idx]=t.einsum('nij,sj->nsi',R,xyz)+pos[:,None]
   world=t.linalg.vector_norm(pts[:,:,None]-self.centers[None,None],dim=-1)-rr[None,:,None]-self.radii
   out[:,kind]=world.flatten(1).min(1).values;sm=t.full((n,),float('inf'),**self.kw)
   for a in range(0,len(ii),64):sm=t.minimum(sm,(t.linalg.vector_norm(pts[:,ii[a:a+64]]-pts[:,jj[a:a+64]],dim=-1)-th[a:a+64]).min(1).values)
   out[:,2+kind]=sm
  out[:,4]=t.minimum(q-self.lo,self.hi-q).min(1).values
  return out.cpu().numpy()

def evaluate(arm,nom,target,draws,sub=10,backend=None,chunk=512):
 beg=time.perf_counter();q,v,u=track(nom,draws);qd=p0.dense(q,v,sub);n,t,_=qd.shape;mins=np.full((n,5),np.inf);flat=qd.reshape(-1,7)
 for a in range(0,len(flat),chunk):
  m=backend.margins(flat[a:a+chunk]) if backend else arm.margins(flat[a:a+chunk])[0];ix=np.arange(a,min(a+chunk,len(flat)))//t;np.minimum.at(mins,ix,m)
 R,pos=arm.fk(q[:,-1])['panda_hand'];pe=np.linalg.norm(pos-target[:,3],axis=-1);ang=Rotation.from_matrix(R.transpose(0,2,1)@target[:,:3]).magnitude()
 flags=np.column_stack((mins<0,(np.abs(u)>arm.vmax).any((1,2)),(np.abs(v)>arm.vmax).any((1,2)),pe>.01,ang>np.deg2rad(2)))
 def stats(f):
  count=int(f.any(1).sum());return {'n':len(f),'counts':dict(zip(p0.EVENTS,f.sum(0).astype(int).tolist())),'union':count,'risk':count/len(f),'CI95':p0.interval(count,len(f))}
 row=stats(flags);row.update(runtime_seconds=time.perf_counter()-beg,grid_seconds=DT/sub,backend='torch_cuda_float64' if backend else 'numpy_cpu',chunk=chunk,prefix64=stats(flags[:64]),min_margins=dict(zip(p0.EVENTS[:5],mins.min(0).tolist())),max_position_m=float(pe.max()),max_orientation_rad=float(ang.max()),command_velocity_margin=float((arm.vmax-np.abs(u)).min()),execution_velocity_margin=float((arm.vmax-np.abs(v)).min()),saturation='none; requested=applied, both hard-limit events counted',peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 return row,flags,q,v,u

def freeze():
 old=json.load(open(P.parent/'phase5/PROTOCOL.json'));bank=P.parent/'phase5/FROZEN_PATHS.npz'
 spec={'scope':'new disclosed ideal-encoder joint tracking benchmark; no planner advantage','law':'u[k]=u_nom[k]-K*(q[k]-q_nom[k]);q[k+1]=q[k]+dt*(u[k]+b+eta[k])','gain_rule':'tau=10 control periods=.5s; K=1/tau=2s^-1 before outcomes','tau_seconds':TAU,'gain_s_inverse':GAIN,'error_transition':A,'duration_seconds':10,'dt':DT,'H':H,'observation':'perfect actual joint positions at50ms updates; no delay, encoder noise or encoder bias','e0_interpretation':'physical initial tracking offset, same numeric draws as P0; not an unobservable calibration/estimation error. This is an explicit new benchmark interpretation, not evidence old estimation uncertainty vanished in hardware','saturation_policy':'no clipping or saturation; requested velocity is applied, over-limit requested/applied and noisy executed velocities are separate hard failures','goal':{'position_m':.01,'orientation_deg':2,'redundant_joints':'free under hard joint limits'},'noise':old['regimes'],'seeds':old['seeds'],'same_noise_draws':'generate1024 streams, use identical64/256 prefixes from P0','primary_grid':.005,'diagnostic_grid':.0025,'same_path_diagnostics':True,'P0_bank_sha256':p0.sha(bank),'P0_source_sha256':p0.sha(P.parent/'phase5/panda_p0.py'),'geometry_events_unchanged':p0.EVENTS,'covariance_formula':'a^(k+l)Se0 + [(1-a^k)/K][(1-a^l)/K]Sb + dt²*a^abs(k-l)*(1-a^(2min(k,l)))/(1-a²)*Seta','terminal_base_std_rad':float(np.sqrt(covariance(H,H))),'source_sha256':p0.sha(__file__),'artifact_cap':30000000,'command_wall_cap':300,'CPU_profiling':'firstpath/base64, source independent full geometry','GPU_profiling':'same firstpath/base64 plus NumPy agreement; host access required; use faster qualified backend for bank, no native extensions/installs'}
 save('PROTOCOL.json',spec);print(json.dumps(spec),flush=True)

def qualify():
 start=time.perf_counter();arm=p0.Arm();nom=np.load(P.parent/'phase5/FROZEN_PATHS.npz')['q'];draw=p0.noise(20000,2026103101);q,v,u=track(np.zeros((H+1,7)),draw);cov=np.diag([.002**2]*7+[.001**2]*7);F=np.block([[A*np.eye(7),DT*np.eye(7)],[np.zeros((7,7)),np.eye(7)]]);Q=np.diag([DT**2*.005**2]*7+[0]*7);cs=[cov.copy()]
 for _ in range(H):cov=F@cov@F.T+Q;cs.append(cov.copy())
 err=[]
 for k,l in [(0,200),(1,50),(20,70),(100,200),(200,200)]:err.append(np.abs((cs[k]@np.linalg.matrix_power(F.T,l-k))[:7,:7]-covariance(k,l)*np.eye(7)).max())
 assert max(err)<1e-18
 empirical=np.diag(np.cov(q[:,-1].T)).mean();cross=np.diag((q[:,20]-q[:,20].mean(0)).T@(q[:,-1]-q[:,-1].mean(0))/(len(q)-1)).mean();assert abs(empirical/covariance(H,H)-1)<.025;assert abs(cross/covariance(20,H)-1)<.03
 zero=(np.zeros((1,7)),np.zeros((1,7)),np.zeros((1,H,7)));parity=max(abs(track(n,zero)[0][0]-n).max() for n in nom);assert parity<1e-13
 noisy=tuple(x[:2] for x in p0.noise(1024,2026103000));qq,vv,uu=track(nom[0],noisy);assert np.array_equal(p0.dense(qq,vv,10),p0.dense(qq,vv,20)[:,::2])
 # Gainzero reproduces P0 physically and eventsemantics at all200 updates.
 qz,vz,uz=track(nom[0],noisy,0);qo,vo=p0.execute(nom[0,0],np.diff(nom[0],axis=0)/DT,noisy);gainzero=float(abs(qz-qo).max());assert gainzero<1e-13
 # No clipping: deliberately large tracking offset produces commanded overspeed.
 bad=(np.full((1,7),2.),np.zeros((1,7)),np.zeros((1,H,7)));_,_,ub=track(nom[0],bad);assert (abs(ub)>arm.vmax).any()
 save('CHECKS.json',{'pass':True,'seconds':time.perf_counter()-start,'exact_augmented_crosscov_max_abs':float(max(err)),'empirical_terminal_variance':float(empirical),'analytic_terminal_variance':covariance(H,H),'empirical_relative_error':float(empirical/covariance(H,H)-1),'empirical_crosscov':float(cross),'analytic_crosscov':covariance(20,H),'nominal_reconstruction_error':float(parity),'gainzero_P0_parity':gainzero,'same_dense_path_exact':True,'unclipped_command_overspeed':True,'a':A,'stability':'abs(a)=.9<1','terminal_base_std':float(np.sqrt(covariance(H,H)))})
 print(json.dumps(json.load(open(P/'CHECKS.json'))),flush=True)

def profile():
 import torch
 start=time.perf_counter();arm=p0.Arm();bank=np.load(P.parent/'phase5/FROZEN_PATHS.npz');d=tuple(v[:64] for v in p0.noise(1024,2026103000));cpu,fc,_,_,_=evaluate(arm,bank['q'][0],bank['target'][0],d);report={'cpu':cpu,'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count()};backend=None
 if torch.cuda.is_available():
  backend=GPUArm(arm);rng=np.random.default_rng(31);qs=arm.lo+rng.random((512,7))*(arm.hi-arm.lo);mc=arm.margins(qs)[0];mg=backend.margins(qs);err=float(abs(mc-mg).max());assert err<1e-10
  gpu,fg,_,_,_=evaluate(arm,bank['q'][0],bank['target'][0],d,backend=backend,chunk=4096);assert np.array_equal(fc,fg);report.update(gpu=gpu,CPU_GPU_margin_max_abs=err,event_flags_equal=True)
  torch.cuda.synchronize();report['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated();report['gpu_name']=torch.cuda.get_device_name()
 report['chosen_backend']='cuda' if backend and report['gpu']['runtime_seconds']<cpu['runtime_seconds'] else 'cpu';report['profile_seconds']=time.perf_counter()-start;save('PROFILE.json',report);print(json.dumps(report),flush=True)

def run(n,sub):
 start=time.perf_counter();spec=json.load(open(P/'PROTOCOL.json'));assert spec['source_sha256']==p0.sha(__file__);assert spec['P0_source_sha256']==p0.sha(P.parent/'phase5/panda_p0.py');assert spec['P0_bank_sha256']==p0.sha(P.parent/'phase5/FROZEN_PATHS.npz');assert json.load(open(P/'CHECKS.json'))['pass'];profile=json.load(open(P/'PROFILE.json'));arm=p0.Arm();backend=GPUArm(arm) if profile['chosen_backend']=='cuda' else None;bank=np.load(P.parent/'phase5/FROZEN_PATHS.npz');rows=[];flags={};demo={}
 if (P/f'RUN_{n}_{sub}.json').exists():raise RuntimeError('Preserve outputs')
 for i in range(4):
  for regime,scale in [('base',1),('double',2)]:
   draws=tuple(v[:n] for v in p0.noise(1024,spec['seeds'][i],scale));row,f,q,v,u=evaluate(arm,bank['q'][i],bank['target'][i],draws,sub,backend,4096 if backend else 512);row.update(path=i,regime=regime);rows.append(row);key=f'p{i}_{regime}';flags[key]=f;demo[key+'_q']=q[:4];demo[key+'_v']=v[:4];demo[key+'_u']=u[:4];print(json.dumps(row),flush=True)
 report={'rows':rows,'process_seconds':time.perf_counter()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 if backend:report['gpu_peak_allocated_bytes']=backend.t.cuda.max_memory_allocated()
 save(f'RUN_{n}_{sub}.json',report);np.savez_compressed(P/f'FLAGS_{n}_{sub}.npz',**flags);np.savez_compressed(P/f'DEMO_{n}_{sub}.npz',**demo)
if __name__=='__main__':
 action=sys.argv[1]
 if action=='freeze':freeze()
 elif action=='qualify':qualify()
 elif action=='profile':profile()
 elif action=='run':run(int(sys.argv[2]),int(sys.argv[3]))
