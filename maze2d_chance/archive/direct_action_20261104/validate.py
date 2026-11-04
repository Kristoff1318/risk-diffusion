"""Numerical validation independent of diffusion outcomes."""
from baseline import *
def main():
 torch.set_num_threads(2);e=gym.make('maze2d-large-v1').unwrapped;e.reset();ch=Chance(e,np.load(ROOT/'normalization.npz'))
 rng=np.random.RandomState(47);errors=[]
 for start,goal in TASKS:
  for point in [start[:2],goal]:
   e.set_state(np.array(point),np.zeros(2));e.step(np.zeros(2))
   assert not wall_contact(e), ('invalid task endpoint',point)
 # Native velocity/action clipping and implicit damping away from walls.
 for i in range(100):
  start=np.r_[np.array([1.,1.])+rng.uniform(-.05,.05,2),rng.uniform(-6,6,2)];a=rng.uniform(-2,2,2)
  e.set_state(start[:2],start[2:]);actual=e.step(a)[0]
  predicted=propagate(torch.tensor(a[None,None]).double(),torch.tensor(start[None]).double(),0.,ch.a,ch.b)[0].numpy()[0,0]
  errors.append(float(np.max(abs(actual-predicted))))
 # Covariance compares 100k independent open-loop free executions with vectorized
 # recurrence, not the covariance implementation being checked.
 n=100000;h=120;xs=np.zeros((n,4));xs[:,:2]=[1.,1.]
 for k in range(h):
  xs[:,2:]=ch.a*xs[:,2:];xs[:,:2]+=.01*xs[:,2:];xs[:,2:]+=SIGMA*rng.randn(n,2)
 _,cov=propagate(torch.zeros(1,h,2).double(),torch.tensor([[1.,1.,0.,0.]]).double(),SIGMA,ch.a,ch.b)
 relative=np.abs(xs[:,:2].var(0)-cov.numpy()[0,-1])/cov.numpy()[0,-1]
 # Finite differences of full energy through rollout, CDF, covariance and softplus.
 z=torch.zeros(1,384,6,dtype=torch.float64,requires_grad=True);s=torch.tensor([[1.,1.,0.,0.]],dtype=torch.float64);g=torch.tensor([[1.4,1.]],dtype=torch.float64)
 val=ch.energy(z,s,g,torch.tensor([.15]));grad=torch.autograd.grad(val.sum(),z)[0];idx=(0,120,0);eps=1e-5
 zp=z.detach().clone();zm=zp.clone();zp[idx]+=eps;zm[idx]-=eps
 fd=float((ch.energy(zp,s,g,torch.tensor([.15]))-ch.energy(zm,s,g,torch.tensor([.15])))/(2*eps));ad=float(grad[idx]);rel=abs(fd-ad)/max(abs(fd),abs(ad),1e-8)
 result=dict(native_free_one_step_max_abs_error=max(errors),covariance_relative_error=relative.tolist(),energy_gradient_autodiff=ad,energy_gradient_finite_difference=fd,gradient_relative_error=rel)
 assert max(errors)<1e-10,result
 assert relative.max()<.02,result
 assert rel<.001,result
 json.dump(result,open(ROOT/'validation.json','w'),indent=2);print(result)
if __name__=='__main__':main()
