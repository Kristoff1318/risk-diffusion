"""Mathematical and end-to-end checks for repaired original-plan components."""
import torch,numpy as np
from .test_risk import main as risk_checks
from .test_evaluate import main as evaluation_checks
from .nominal import sample as nominal_sample,dataset
from .guidance import sample,load
from .problem import Problem
from .planner import plan
from .projection import project_batch

def main():
    torch.set_num_threads(2);risk_checks();evaluation_checks()
    model,mean,scale=load('risk_diffusion/revision/outputs/nominal_seed0/nominal.pt');s,g,_=dataset(4,1800)
    a=nominal_sample(model,mean,scale,s,g,seed=912);b,_=sample(model,mean,scale,s,g,seed=912)
    assert torch.allclose(a,b,atol=1e-6),'B0 and no-guidance sampler disagree'
    yes=plan(model,mean,scale,s[0],g[0],Problem(process_sigma=0.,initial_sigma=0.),.1,candidates=2,draws=256,seed=212)
    no=plan(model,mean,scale,s[0],g[0],Problem(goal_half=.00001),.01,candidates=2,draws=256,seed=212)
    assert yes['accepted'] and yes['controls'] is not None
    assert not no['accepted'] and no['controls'] is None
    u=torch.zeros((1,20,2),dtype=torch.float64);u[:,:,0]=.75
    st=torch.tensor([[-1.5,.1]],dtype=torch.float64);gt=torch.tensor([[1.5,.1]],dtype=torch.float64)
    v,d=project_batch(u,st,gt,Problem(),torch.ones_like(u));_,d2=project_batch(v,st,gt,Problem(),torch.ones_like(u))
    assert d[0]['feasible'] and not d2[0]['attempted']
    print('checks: raw B0 equivalence, validated acceptance/rejection, projection feasibility and idempotence passed')
if __name__=='__main__':main()
