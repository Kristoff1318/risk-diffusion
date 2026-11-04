"""Read-only asset reuse; kinematic tracking-error qualification, no planning.
Run: existing .venv-headless/bin/python phase5/panda_p0.py freeze|qualify|run
"""
import os,time,pathlib,json,hashlib,sys,io,pickle,resource
T0=time.perf_counter()
import numpy as np, yaml
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
from scipy.stats import beta
import xml.etree.ElementTree as ET
P=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path('/home/kristoff/Documents/experiments/mpd-seeded-qualification-20261001')
DATA=BASE/'external/mpd-headless/mpd/torch_robotics/torch_robotics/data'
CU=pathlib.Path('/home/kristoff/Documents/experiments/km-mers-diffusion-diagnostics-20260930/external/curobo/src/curobo/content')
DT=.05;H=200;DURATION=10.;EVENTS=['world_mpd','world_curobo','self_mpd','self_curobo','joint','command_velocity','executed_velocity','position','orientation']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,v): (P/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def tensor_numpy(v):return v.detach().cpu().numpy()
class RestrictedCPU(pickle.Unpickler):
 def find_class(self,module,name):
  import torch
  if module=='_codecs' and name=='encode':
   def enc(v,encoding='latin1',errors='strict'):
    assert encoding in ('latin1','latin-1') and errors=='strict';return v.encode(encoding,errors)
   return enc
  if module=='torch.storage' and name=='_load_from_bytes':return lambda b:torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
  allowed={'torch._utils':{'_rebuild_tensor_v2','_rebuild_tensor'},'collections':{'OrderedDict'},'numpy.core.multiarray':{'_reconstruct','scalar'},'numpy':{'ndarray','dtype'},'builtins':{'set','slice'}}
  if name not in allowed.get(module,set()):raise RuntimeError((module,name))
  return super().find_class(module,name)

class Arm:
 def __init__(self):
  self.urdfs=[DATA/'urdf/robots/franka_description/robots/panda_arm_hand_fixed_gripper.urdf',CU/'assets/robot/franka_description/franka_panda.urdf']
  self.trees=[]
  for path in self.urdfs:
   joints=[]
   for e in ET.parse(path).getroot().findall('joint'):
    if e.find('child').get('link')=='panda_link0':continue
    origin=e.find('origin');xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ');rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
    R=Rotation.from_euler('xyz',rpy).as_matrix();name=e.get('name');idx=int(name[-1])-1 if name.startswith('panda_joint') and name[-1] in '1234567' else None
    typ=e.get('type');val=.04 if name.startswith('panda_finger_joint') and typ=='prismatic' else 0.
    axis=np.fromstring(e.find('axis').get('xyz'),sep=' ') if e.find('axis') is not None else np.array([0,0,1.])
    joints.append((e.find('parent').get('link'),e.find('child').get('link'),R,xyz,idx,typ,val,axis))
   self.trees.append(joints)
  mp=yaml.safe_load((DATA/'configs/panda/panda_sphere_config.yaml').read_text());self.mp_cfg=mp
  cfg=yaml.safe_load((CU/'configs/robot/franka.yml').read_text())['robot_cfg']['kinematics'];self.cu_cfg=cfg
  cu=yaml.safe_load((CU/'configs/robot'/cfg['collision_spheres']).read_text())['collision_spheres']
  self.spheres=[];self.pairs=[]
  for kind,d in [('mpd',mp),('curobo',cu)]:
   spheres=[]
   for link,rows in d.items():
    if link=='self_collision':continue
    if kind=='curobo' and link not in cfg['collision_link_names']:continue
    for row in rows:
     xyz=row[:3] if kind=='mpd' else row['center'];rad=row[3] if kind=='mpd' else row['radius']+cfg['collision_sphere_buffer']
     if rad>=0:spheres.append((link,np.array(xyz),rad))
   self.spheres.append(spheres);pairs=[]
   for i,(li,_,ri) in enumerate(spheres):
    for j,(lj,_,rj) in enumerate(spheres):
     if kind=='mpd':
      keep=lj in mp['self_collision'].get(li,[]);margin=ri+rj
     else:
      keep=i<j and li!=lj and lj not in cfg['self_collision_ignore'].get(li,[]) and li not in cfg['self_collision_ignore'].get(lj,[])
      margin=ri+rj+cfg['self_collision_buffer'].get(li,0)+cfg['self_collision_buffer'].get(lj,0)
     if keep:pairs.append((i,j,margin))
   self.pairs.append(np.array(pairs))
  limits=yaml.safe_load((DATA/'configs/panda/joint_limits.yaml').read_text())
  self.lo=np.array([limits[f'panda_joint{i}']['qmin'] for i in range(1,8)]);self.hi=np.array([limits[f'panda_joint{i}']['qmax'] for i in range(1,8)]);self.vmax=np.array([limits[f'panda_joint{i}']['qdot_max'] for i in range(1,8)])
  # Genuine native EnvSpheres3D; AST extraction avoids importing output/cache-producing modules.
  import ast
  source=DATA.parents[0]/'environments/env_spheres_3d.py'
  tree=ast.parse(source.read_text());cl=next(v for v in tree.body if isinstance(v,ast.ClassDef) and v.name=='EnvSpheres3D');init=next(v for v in cl.body if isinstance(v,ast.FunctionDef) and v.name=='__init__')
  assign=next(v for v in init.body if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='centers' for t in v.targets))
  self.centers=np.array(ast.literal_eval(assign.value.args[0]));self.radii=np.full(len(self.centers),.15)
 def fk(self,q,kind=0):
  q=np.asarray(q);n=len(q);poses={'panda_link0':(np.broadcast_to(np.eye(3),(n,3,3)),np.zeros((n,3)))}
  pending=list(self.trees[kind])
  while pending:
   progress=False
   for j in pending[:]:
    parent,child,R0,t0,idx,typ,val,axis=j
    if parent not in poses:continue
    Rp,tp=poses[parent];R=Rp@R0;t=tp+np.einsum('nij,j->ni',Rp,t0)
    if idx is not None:
     assert np.allclose(axis,[0,0,1]);c=np.cos(q[:,idx]);s=np.sin(q[:,idx]);rz=np.zeros((n,3,3));rz[:,0,0]=c;rz[:,0,1]=-s;rz[:,1,0]=s;rz[:,1,1]=c;rz[:,2,2]=1;R=R@rz
    elif typ=='prismatic':t=t+np.einsum('nij,j->ni',R,axis*val)
    poses[child]=(R,t);pending.remove(j);progress=True
   if not progress:raise RuntimeError('Disconnected URDF')
  return poses
 def points(self,poses,kind):
  out=np.empty((len(poses['panda_hand'][1]),len(self.spheres[kind]),3))
  for i,(link,xyz,_) in enumerate(self.spheres[kind]):
   R,t=poses[link];out[:,i]=np.einsum('nij,j->ni',R,xyz)+t
  return out
 def margins(self,q,centers=None,radii=None):
  n=len(q);out=np.empty((n,5));poses=self.fk(q);pose=poses['panda_hand'];centers=self.centers if centers is None else centers;radii=self.radii if radii is None else radii
  for kind in range(2):
   pts=self.points(poses if kind==0 else self.fk(q,1),kind);rr=np.array([s[2] for s in self.spheres[kind]])
   world=np.full(n,np.inf)
   for c,r in zip(centers,radii):world=np.minimum(world,(np.linalg.norm(pts-c,axis=-1)-rr-r).min(1))
   out[:,kind]=world;sm=np.full(n,np.inf)
   pairs=self.pairs[kind]
   # Pair blocks limit temporary distance tensors; every configured pair retained.
   for a in range(0,len(pairs),64):
    block=pairs[a:a+64];i=block[:,0].astype(int);j=block[:,1].astype(int)
    sm=np.minimum(sm,(np.linalg.norm(pts[:,i]-pts[:,j],axis=-1)-block[:,2]).min(1))
   out[:,2+kind]=sm
  out[:,4]=np.minimum(q-self.lo,self.hi-q).min(1)
  return out,pose

def noise(n,seed,scale=1.):
 rng=np.random.default_rng(seed)
 return (rng.normal(size=(n,7))*.002*scale,rng.normal(size=(n,7))*.001*scale,rng.normal(size=(n,H,7))*.005*scale)
def execute(q0,u,draws):
 e,b,eta=draws;v=u[None]+b[:,None]+eta
 q=np.concatenate(((q0+e)[:,None],(q0+e)[:,None]+np.cumsum(DT*v,axis=1)),axis=1)
 return q,v
def dense(q,v,sub):
 # The identical per-step velocity realization is used at every refinement.
 fractions=np.arange(sub)/sub
 a=q[:,:-1,None]+DT*fractions[None,None,:,None]*v[:,:,None]
 return np.concatenate((a.reshape(len(q),H*sub,7),q[:,-1:]),axis=1)
def interval(k,n):
 return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]
def evaluate(arm,q0,u,target,draws,sub=10,chunk=512):
 begin=time.perf_counter();q,v=execute(q0,u,draws);qd=dense(q,v,sub);n,t,_=qd.shape;mins=np.full((n,5),np.inf)
 flat=qd.reshape(-1,7)
 for a in range(0,len(flat),chunk):
  m,_=arm.margins(flat[a:a+chunk]);ix=np.arange(a,min(a+chunk,len(flat)))//t
  np.minimum.at(mins,ix,m)
 R,p=arm.fk(q[:,-1])['panda_hand'];pos=np.linalg.norm(p-target[:,3],axis=-1);angle=Rotation.from_matrix(R.transpose(0,2,1)@target[:,:3]).magnitude()
 flags=np.column_stack((mins<0,np.full(n,(np.abs(u)>arm.vmax).any()),(np.abs(v)>arm.vmax).any((1,2)),pos>.01,angle>np.deg2rad(2)))
 union=flags.any(1);seconds=time.perf_counter()-begin
 row={'n':n,'grid_seconds':DT/sub,'runtime_seconds':seconds,'chunk_states':chunk,'counts':dict(zip(EVENTS,flags.sum(0).astype(int).tolist())),'union':int(union.sum()),'union_probability':float(union.mean()),'union_CI95':interval(int(union.sum()),n),'event_CI95':{k:interval(int(flags[:,i].sum()),n) for i,k in enumerate(EVENTS)},'marginal_count_sum':int(flags.sum()),'min_margins':dict(zip(EVENTS[:5],mins.min(0).tolist())),'position_max_m':float(pos.max()),'orientation_max_rad':float(angle.max()),'command_velocity_margin_rad_s':float((arm.vmax-np.abs(u)).min()),'executed_velocity_margin_rad_s':float((arm.vmax-np.abs(v)).min()),'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 return row,flags,q,v

def freeze():
 if (P/'FROZEN_PATHS.npz').exists():raise RuntimeError('Preserve freeze')
 import torch
 torch.set_num_threads(2)
 selection=json.loads((BASE/'runs/valid-query-selection/selection.json').read_text());indices=selection['selected_indices']['EnvSpheres3D'][:4]
 index=json.loads((BASE/'assets/archive-extraction.json').read_text());db=BASE/'assets'/index['target_dataset_prefix'];f=db/'dataset_merged_doubled_reload-ntasks_-1--bspline-degree_5-n_pts_29-zero_vel_True-zero_acc_True.pickle'
 with open(f,'rb') as stream:fields=RestrictedCPU(stream).load()['fields']
 learned=tensor_numpy(fields['control_points'][indices]).astype(float);q0=tensor_numpy(fields['q_start'][indices]).astype(float);target=tensor_numpy(fields['ee_goal_pose'][indices]).astype(float)
 controls=np.concatenate((np.repeat(q0[:,None],3,1),learned[:,:-1],np.repeat(learned[:,-1:],3,1)),1);assert controls.shape==(4,29,7)
 knots=np.pad(np.linspace(0,1,25),5,'edge');q=BSpline(knots,controls,5,axis=1)(np.linspace(0,1,H+1));u=np.diff(q,axis=1)/DT
 np.savez_compressed(P/'FROZEN_PATHS.npz',q=q,u=u,target=target,control_points=controls,official_indices=indices)
 arm=Arm();assets=[*arm.urdfs,DATA/'configs/panda/panda_sphere_config.yaml',DATA/'configs/panda/joint_limits.yaml',CU/'configs/robot/franka.yml',CU/'configs/robot/spheres/franka_mesh.yml',DATA.parent/'environments/env_spheres_3d.py',BASE/'runs/valid-query-selection/selection.json',BASE/'runs/valid-query-selection/EnvSpheres3D-queries.pt',BASE/'runs/valid-query-selection/reference-path-audit.json',BASE/'external/mpd-splines-public/LICENSE',CU.parents[2]/'LICENSE',CU.parents[2]/'LICENSE_ASSETS',f]
 hashes={str(a):sha(a) for a in assets}
 specification={'scope':'P0 fixed bank kinematic tracking-error qualification; no planner performance comparison','duration_seconds':DURATION,'dt':DT,'H':H,'equation':'q[k+1]=q[k]+dt*(u[k]+b+eta[k]);q0=qstart+e0','regimes':{'base':{'initial_std_rad':.002,'constant_bias_std_rad_s':.001,'iid_step_velocity_std_rad_s':.005},'double':{'initial_std_rad':.004,'constant_bias_std_rad_s':.002,'iid_step_velocity_std_rad_s':.010}},'noise_independence':'all joints independent, e0/b/eta independent; b shared across time; eta constant within each50ms step','terminal_std_rad_base':float(np.sqrt(.002**2+10**2*.001**2+H*DT**2*.005**2)),'covariance':'Sigmae0+k*l*dt^2*Sigmab+min(k,l)*dt^2*Sigmaeta','uncalibrated':True,'pose_tolerance_m':.01,'orientation_tolerance_deg':2,'terminal_joint_target':False,'events':EVENTS,'command_and_executed_velocity_checked':True,'acceleration_or_rigid_body_model':False,'primary_grid_s':.005,'diagnostic_grid_s':.0025,'same_noise_path_grids':True,'contact_rule':'negative signed clearance fails; zero contact nonviolating','geometry':'both complete native sphere sets/configured self pairs, fixed gripper. Native10 sphere scene; no table was present in this asset. No mesh coverage or continuous safety certificate','spheres':[len(s) for s in arm.spheres],'self_pairs':[len(s) for s in arm.pairs],'official_indices':indices,'bank_selection':'first4 existing selected native EnvSpheres3D references, no stochastic outcome selection','seeds':[2026103000+i for i in range(4)],'sampling':'prefix64,256; optional1024 only if measured budget permits; all cases kept','chunk_states':512,'asset_hashes':hashes,'license_boundary':'existing local licensed assets used read-only; no redistribution/copied meshes; see referenced MIT/public and cuRobo LICENSE/ASSETS','source_hash':sha(__file__),'frozen_bank_hash':sha(P/'FROZEN_PATHS.npz')}
 save('PROTOCOL.json',specification);print(json.dumps({'freeze_seconds':time.perf_counter()-T0,'indices':indices,'spheres':specification['spheres'],'pairs':specification['self_pairs'],'RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)

def qualify():
 z=np.load(P/'FROZEN_PATHS.npz');arm=Arm();checks={}
 e,b,eta=noise(2,17);zero=(np.zeros((1,7)),np.zeros((1,7)),np.zeros((1,H,7)))
 checks['nominal_control_state_max_abs']=float(max(np.abs(execute(z['q'][i,0],z['u'][i],zero)[0][0]-z['q'][i]).max() for i in range(4)));assert checks['nominal_control_state_max_abs']<1e-13
 # Independent augmented14d covariance recursion and transition crosscov vs closed form.
 A=np.block([[np.eye(7),DT*np.eye(7)],[np.zeros((7,7)),np.eye(7)]]);Q=np.zeros((14,14));Q[:7,:7]=DT**2*.005**2*np.eye(7);cov=np.diag([.002**2]*7+[.001**2]*7);covs=[cov.copy()]
 for _ in range(H):cov=A@cov@A.T+Q;covs.append(cov.copy())
 errors=[]
 for k,l in [(0,200),(1,199),(50,120),(200,200)]:
  cross=covs[k]@np.linalg.matrix_power(A.T,l-k);expected=(.002**2+k*l*DT**2*.001**2+min(k,l)*DT**2*.005**2)*np.eye(7);errors.append(np.abs(cross[:7,:7]-expected).max())
 checks['augmented_cross_time_covariance_max_abs']=float(max(errors));assert max(errors)<1e-16
 # Bias-only covariance must scale as t*s, not min(t,s).
 d=(np.zeros((2,7)),b,np.zeros_like(eta));q,_=execute(np.zeros(7),np.zeros((H,7)),d);checks['persistent_bias_exact_path_error']=float(np.abs(q-b[:,None]*np.linspace(0,10,H+1)[None,:,None]).max());assert checks['persistent_bias_exact_path_error']<1e-14
 q,v=execute(z['q'][0,0],z['u'][0],(e,b,eta));coarse=dense(q,v,10);fine=dense(q,v,20);checks['paired_grid_max_abs']=float(np.abs(coarse-fine[:,::2]).max());assert checks['paired_grid_max_abs']==0
 # Independent authoritative cached goals, and both URDFs agree.
 targets=z['target'];R,p=arm.fk(z['q'][:,-1])['panda_hand'];R2,p2=arm.fk(z['q'][:,-1],1)['panda_hand'];checks['two_URDF_FK_error']=float(max(np.abs(R-R2).max(),np.abs(p-p2).max()));checks['prescribed_goal_FK_error']=float(max(np.abs(R-targets[:,:,:3]).max(),np.abs(p-targets[:,:,3]).max()));assert checks['prescribed_goal_FK_error']<2e-6 and checks['two_URDF_FK_error']<2e-8
 nominal=[]
 for i in range(4):
  row,flags,_,_=evaluate(arm,z['q'][i,0],z['u'][i],targets[i],zero);assert not flags.any();nominal.append(row)
 # Separate constructed elbow obstacle: freeze collision center at existing link3 sphere, EE far away.
 qfixture=z['q'][0,0][None];poses=arm.fk(qfixture);pts=arm.points(poses,0);ix=next(i for i,s in enumerate(arm.spheres[0]) if s[0]=='panda_link3');center=pts[0,ix];m,pose=arm.margins(qfixture,center[None],np.array([.01]));checks['elbow_fixture']={'world_margin_m':m[0,0],'EE_clearance_m':float(np.linalg.norm(pose[1][0]-center)-.01),'link':arm.spheres[0][ix][0]};assert m[0,0]<0 and checks['elbow_fixture']['EE_clearance_m']>.1
 # Fixed deterministic configuration search for isolated self event, NOT bank/task replacement.
 rng=np.random.default_rng(2026102999);candidate=arm.lo+rng.random((1024,7))*(arm.hi-arm.lo);m,pose=arm.margins(candidate);eligible=np.where((m[:,:2].min(1)>0)&(m[:,2:4].min(1)<0)&(m[:,4]>0))[0];assert len(eligible)>0
 j=int(eligible[0]);qs=candidate[j];checks['self_fixture']={'q':qs.tolist(),'world_margins':m[j,:2].tolist(),'self_margins':m[j,2:4].tolist(),'joint_margin':float(m[j,4]),'search_seed':2026102999,'search_count':1024};assert m[j,2:4].min()<0
 # Orientation-only changed prescribed target, position left exact; no trajectory re-anchoring.
 wrong=targets[0].copy();wrong[:,:3]=wrong[:,:3]@Rotation.from_rotvec([0,0,np.deg2rad(10)]).as_matrix();row,flags,_,_=evaluate(arm,z['q'][0,0],z['u'][0],wrong,zero);assert flags[0,-1] and not flags[0,:-1].any();checks['orientation_only_fixture']=row
 # Exact velocity and joint event checks, no clipping.
 bad_u=z['u'][0].copy();bad_u[0,0]=arm.vmax[0]+.01;assert (np.abs(bad_u)>arm.vmax).any();huge=(np.ones((1,7))*10,np.zeros((1,7)),np.zeros((1,H,7)));row,flags,_,_=evaluate(arm,z['q'][0,0],z['u'][0],targets[0],huge);assert flags[0,4];checks['unclipped_joint_fixture']=row
 checks['pass']=True;checks['nominal']=nominal;checks['seconds']=time.perf_counter()-T0;save('QUALIFICATION.json',checks);print(json.dumps(checks),flush=True)

def run(n,sub):
 protocol=json.loads((P/'PROTOCOL.json').read_text());assert sha(__file__)==protocol['source_hash'];assert sha(P/'FROZEN_PATHS.npz')==protocol['frozen_bank_hash'];assert json.loads((P/'QUALIFICATION.json').read_text())['pass']
 for path,h in protocol['asset_hashes'].items():assert sha(path)==h,path
 arm=Arm();z=np.load(P/'FROZEN_PATHS.npz');rows=[];flags_saved={};traces={};out=P/f'RUN_n{n}_sub{sub}.json'
 if out.exists():raise RuntimeError('Preserve existing outcomes')
 for i in range(4):
  for regime,scale in [('base',1.),('double',2.)]:
   # Generate1024 once so prefixes64/256/1024 share exact draws across sizes/grids.
   draws=tuple(v[:n] for v in noise(1024,protocol['seeds'][i],scale))
   row,flags,q,v=evaluate(arm,z['q'][i,0],z['u'][i],z['target'][i],draws,sub);row.update(path=i,regime=regime);rows.append(row);flags_saved[f'p{i}_{regime}']=flags;traces[f'p{i}_{regime}_q']=q[:4];traces[f'p{i}_{regime}_v']=v[:4]
   print(json.dumps({'path':i,'regime':regime,'n':n,'sub':sub,'seconds':row['runtime_seconds'],'union':row['union'],'events':row['counts']}),flush=True)
 save(out.name,{'rows':rows,'process_seconds':time.perf_counter()-T0,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss});np.savez_compressed(P/f'FLAGS_n{n}_sub{sub}.npz',**flags_saved);np.savez_compressed(P/f'DEMO_n{n}_sub{sub}.npz',**traces)
if __name__=='__main__':
 action=sys.argv[1]
 if action=='freeze':freeze()
 elif action=='qualify':qualify()
 elif action=='run':run(int(sys.argv[2]),int(sys.argv[3]))
 else:raise ValueError(action)
