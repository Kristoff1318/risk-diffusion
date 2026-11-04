"""Independent NumPy stochastic execution evaluator, separate from guidance."""
from __future__ import annotations
import numpy as np
from scipy.stats import beta as beta_dist

H=20; DT=.20; NOISE=.025; RADIUS=.43; GOAL_HALF=.30

def evaluate(controls,starts,goals,draws=4096,seed=0,delta=.1):
    rng=np.random.default_rng(seed); out=[]
    for u,start,goal in zip(np.asarray(controls),np.asarray(starts),np.asarray(goals)):
        x=np.tile(start,(draws,1)).astype(float)
        violation=np.full(draws,-np.inf)
        collided=np.zeros(draws,dtype=bool)
        path=np.zeros(draws)
        for uk in u:
            prev=x.copy()
            drift=np.column_stack((.12*np.sin(2*x[:,1]),.12*np.sin(2*x[:,0])))
            x=x+DT*(uk+drift)+rng.normal(0,NOISE,(draws,2))
            loss=RADIUS-np.linalg.norm(x,axis=1)
            collided|=loss>=0
            violation=np.maximum(violation,loss)
            path+=np.linalg.norm(x-prev,axis=1)
        goal_loss=np.max(np.abs(x-goal)-GOAL_HALF,axis=1)
        violation=np.maximum(violation,goal_loss)
        reached=goal_loss<=0
        safe=(~collided)&reached
        n=int(safe.sum()); lo=float(beta_dist.ppf(.025,n,draws-n+1)) if n else 0.
        hi=float(beta_dist.ppf(.975,n+1,draws-n)) if n<draws else 1.
        tail=max(1,int(np.ceil(delta*draws)))
        out.append({'joint_safety':float(safe.mean()),'joint_ci':[lo,hi],
                    'collision':float(collided.mean()),'goal':float(reached.mean()),
                    'cost':float(path.mean()),'cvar':float(np.sort(violation)[-tail:].mean()),
                    'risk':float(1-safe.mean())})
    return out
