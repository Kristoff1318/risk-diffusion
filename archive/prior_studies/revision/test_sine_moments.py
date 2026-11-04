"""Numerical checks against independent quadrature and finite differences."""
import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from .problem import Problem
from .risk import chance,mean_covariance
from .sine_moments import gaussian_sine_step,gaussian_sine_moments,chance_from_moments,chance_sine

def run_checks():
    p=Problem(strength=.35,process_sigma=.04)
    nodes,weights=hermgauss(31)
    z=np.stack(np.meshgrid(nodes,nodes,indexing='ij'),-1).reshape(-1,2)*np.sqrt(2)
    w=np.outer(weights,weights).reshape(-1)/np.pi
    errors=[]
    for covariance in (np.array([[.16,.08],[.08,.09]]),np.array([[.16,-.06],[-.06,.09]])):
        mean=np.array([.3,-.5]);control=np.array([.2,-.4])
        points=mean+z@np.linalg.cholesky(covariance).T
        y=points+p.dt*control+p.dt*p.strength*np.sin(2*points[:,::-1])
        qm=(w[:,None]*y).sum(0);d=y-qm
        qc=(d*w[:,None]).T@d+p.process_sigma**2*np.eye(2)
        am,ac=gaussian_sine_step(torch.tensor(mean)[None],torch.tensor(covariance)[None],torch.tensor(control)[None],p)
        em=float(np.max(abs(am.numpy()[0]-qm)));ec=float(np.max(abs(ac.numpy()[0]-qc)))
        assert em<1e-10 and ec<1e-10,(em,ec)
        errors.append({'mean_max_abs':em,'covariance_max_abs':ec,'input_cross_covariance':float(covariance[0,1])})
    gen=torch.Generator().manual_seed(202610020)
    u=torch.randn((64,20,2),generator=gen,dtype=torch.float64)
    starts=torch.randn((64,2),generator=gen,dtype=torch.float64)
    m,c=gaussian_sine_moments(u,starts,p)
    assert torch.isfinite(m).all() and torch.isfinite(c).all()
    eig=float(torch.linalg.eigvalsh(c).min());assert eig>=-1e-12,eig
    linear=Problem(strength=0.)
    old=mean_covariance(u,starts,linear);new=gaussian_sine_moments(u,starts,linear)
    assert torch.allclose(old[0],new[0],atol=1e-12) and torch.allclose(old[1],new[1],atol=1e-12)
    goal=starts+torch.tensor([3.,0.])
    old_risk=chance(u,starts,goal,p)
    same=chance_from_moments(old_risk['mean'],old_risk['covariance'],goal,p)
    assert torch.allclose(old_risk['risk'],same['risk'],atol=1e-12,rtol=0)
    sample=u[:1].clone().requires_grad_(True)
    risk=chance_sine(sample,starts[:1],goal[:1],p)['risk'].sum()
    grad=torch.autograd.grad(risk,sample)[0]
    delta=1e-5;plus=sample.detach().clone();minus=plus.clone()
    plus[0,10,1]+=delta;minus[0,10,1]-=delta
    finite=(chance_sine(plus,starts[:1],goal[:1],p)['risk']-chance_sine(minus,starts[:1],goal[:1],p)['risk'])/(2*delta)
    assert torch.allclose(finite,grad[0,10,1],atol=1e-7,rtol=1e-3)
    zero=Problem(strength=.35,process_sigma=0.,initial_sigma=0.)
    _,zc=gaussian_sine_moments(u,starts,zero)
    assert float(zc.abs().max())<1e-14
    return {'quadrature_cases':errors,'random_multistep_min_eigenvalue':eig,'linear_case':True,'zero_noise_case':True,'unchanged_event_formula':True,'gradient_finite_difference':True,'psd_projection_applied':False}

if __name__=='__main__':print(run_checks())
