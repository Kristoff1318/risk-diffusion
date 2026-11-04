import numpy as np
import torch
from .problem import Problem
from .risk import mean_covariance,sampled_cvar,chance
from .nominal import dataset
from .evaluate import _fractional_cvar

def main():
    p=Problem(strength=0.)
    u=torch.zeros((2,20,2),dtype=torch.float64);s=torch.tensor([[-1.5,0.],[-1.5,.1]],dtype=torch.float64)
    mu,P=mean_covariance(u,s,p)
    assert torch.allclose(mu[:,-1],s)
    assert torch.allclose(P[:,-1],torch.eye(2,dtype=u.dtype)[None]*(p.initial_sigma**2+20*p.process_sigma**2))
    loss=torch.tensor([[1.],[2.],[3.],[4.]],requires_grad=True)
    tail=sampled_cvar(loss,.625);tail.backward()
    assert abs(tail.item()-3.2)<1e-6 and abs(loss.grad.sum().item()-1)<1e-6
    assert abs(tail.item()-_fractional_cvar(loss.detach().numpy().ravel(),.625))<1e-6
    a,b,c=dataset(2,44);u=torch.tensor(c,dtype=torch.float64,requires_grad=True)
    s=torch.tensor(a,dtype=torch.float64);g=torch.tensor(b,dtype=torch.float64);p=Problem()
    r=chance(u,s,g,p)['risk'].sum();grad=torch.autograd.grad(r,u)[0]
    step=1e-5;plus=u.detach().clone();minus=plus.clone();plus[0,10,1]+=step;minus[0,10,1]-=step
    finite=(chance(plus,s,g,p)['risk'].sum()-chance(minus,s,g,p)['risk'].sum())/(2*step)
    assert torch.allclose(finite,grad[0,10,1],atol=1e-6,rtol=1e-3),(finite,grad[0,10,1])
    print('test_risk: covariance, fractional CVaR gradients, chance finite difference passed')
if __name__=='__main__':main()
