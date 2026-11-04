"""Representation/physics/validator gates on old fixtures, no heldout scoring."""
import pathlib,sys,time,json,numpy as np,torch
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import core
T=time.perf_counter();c=core.Core();old=np.load(P.parent/'phase5/FROZEN_PATHS.npz');start=c.cast(old['q'][0,0]);target=c.cast(old['target'][0]);controls=old['control_points'][0];x=c.cast(np.concatenate((controls[3:26],controls[-1:]),0));nom=c.nom(x,start);assert abs(nom.cpu().numpy()-old['q'][0]).max()<1e-12
# Complete geometry uses identical margins to immutable prior independent backend.
qs=c.cast(old['q'][0,::10]);m=c.margins_tensor(qs).detach().cpu().numpy();previous=c.arm.margins(qs.cpu().numpy())[0];assert abs(m-previous).max()<1e-10
# Exact fractional tail test, including sub-one-sample tail.
v=c.cast([1,2,3,4]);assert float(c.tail(v,.3))==float((4+.2*3)/1.2);assert float(c.tail(v,.1))==4.
# Independent physical rollout agreement with precomputed error representation.
draw=core.p0.noise(64,2026104900,2.);q,v,u=core.tracking.track(old['q'][0],draw);e,ve,ue=core.errors(64,2026104900,2.);ff=np.diff(old['q'][0],axis=0)/.05;assert abs(core.p0.dense(q,v,10)-(core.p0.dense(old['q'][0:1],ff[None],10)+e)).max()<1e-12
# Common loss event sign agrees with independent validator on this fixture.
err=tuple(c.cast(a[:32]) for a in (e,ve,ue));xp=x.clone().requires_grad_(True);lossvec=c.risk_losses(c.nom(xp,start),target,err);tail=c.tail(lossvec,.05);J=c.cost(c.nom(xp,start));g=torch.autograd.grad(J+500*torch.relu(tail+.05).square(),xp)[0];assert torch.isfinite(g).all()
# Finite difference of common J + physical loss away from event ties.
eps=1e-5;costonly=torch.autograd.grad(c.cost(c.nom(xp,start)),xp)[0];xx=x.clone();xx[10,2]+=eps;yy=x.clone();yy[10,2]-=eps;fd=(c.cost(c.nom(xx,start))-c.cost(c.nom(yy,start)))/(2*eps);assert abs(float(fd-costonly[10,2]))<1e-5
rawrow,flags=core.audit(c,nom,target,(e,ve,ue));assert np.array_equal((lossvec.detach().cpu().numpy()>0),flags[:32].any(1))
# Profile requested large validator sample sizes before freezing main effort.
profile=[]
for n in [1024,4096,8192]:
 beg=time.perf_counter();er=core.errors(n,2026104990,2.);physics=time.perf_counter()-beg;r,f=core.audit(c,nom,target,er,16);profile.append({'n':n,'noise_preparation_seconds':physics,'evaluation_seconds':r['seconds'],'total_seconds':physics+r['seconds']});del er
# Full32sample risk gradient/step throughput at unchanged old fixture.
er=tuple(c.cast(a) for a in core.errors(32,2026104980,2.));_,ref=core.refine(c,x,start,target,steps=3,lr=.01,delta=.05,err=er,wallcap=10)
# Fresh genuine inference model pipeline gate on old context only.
model,norm,ctx,meta=core.load_mpd();base=core.p0.BASE;queries={k:v.to('cuda') for k,v in torch.load(base/'runs/valid-query-selection/EnvSpheres3D-queries.pt',map_location='cpu',weights_only=True).items()};xx,mlog=core.mpd(model,norm,ctx,queries,0,2026104988);assert xx.shape==(2,24,7) and torch.isfinite(xx).all();assert all(not p.requires_grad for p in model.parameters())
report={'pass':True,'nominal_reconstruction_error':float(abs(nom.cpu().numpy()-old['q'][0]).max()),'geometry_max_abs':float(abs(m-previous).max()),'finite_gradient':True,'cost_fd_error':float(abs(fd-costonly[10,2])),'loss_event_sign_equal':True,'physical_error_reparameterization_exact':True,'fractional_tail_pass':True,'validator_profiles':profile,'risk3step_profile':ref,'genuine_inference':mlog,'strict_frozen_model':True,'CUDA_peak_allocated':torch.cuda.max_memory_allocated(),'seconds':time.perf_counter()-T};(P/'QUALIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
