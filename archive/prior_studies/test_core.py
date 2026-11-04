import numpy as np
import torch
from .core import H, chance_risk, cvar, data, rollout, terminal_correct
from .evaluator import evaluate

def test_nominal_data_and_terminal_correction():
    s,g,u=data(8,seed=9)
    st=torch.tensor(s);gt=torch.tensor(g);ut=torch.tensor(u)
    end=rollout(ut,st)[0,:,-1]
    assert torch.max(torch.abs(end-gt)).item()<1e-5
    bad=ut+.1
    fixed=terminal_correct(bad,st,gt)
    assert torch.max(torch.abs(rollout(fixed,st)[0,:,-1]-gt)).item()<2e-4

def test_risk_gradients_and_independent_evaluator():
    s,g,u=data(3,seed=3,height_range=(.4,.8))
    ut=torch.tensor(u,requires_grad=True);st=torch.tensor(s);gt=torch.tensor(g)
    risk=chance_risk(ut,st,gt)
    assert torch.isfinite(risk).all()
    risk.sum().backward();assert torch.isfinite(ut.grad).all()
    tail=cvar(ut.detach().requires_grad_(True),st,gt,.1,K=32)
    assert torch.isfinite(tail).all()
    measured=evaluate(u,s,g,draws=256,seed=4)
    assert all(0<=x['joint_ci'][0]<=x['joint_safety']<=x['joint_ci'][1]<=1 for x in measured)
