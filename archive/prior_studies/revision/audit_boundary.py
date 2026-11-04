"""Reproduce the independently audited nonlinear false-safe counterexample."""
import json
from pathlib import Path
import numpy as np
import torch
from .problem import Problem
from .evaluate import evaluate
from .calibration import decompose

def main():
    torch.set_num_threads(2)
    root=Path('risk_diffusion/revision/outputs')
    data=np.load(root/'conventional/nonlinear_WaypointChanceOptimizer.npz')
    i=3;p=Problem(strength=.35,process_sigma=.04,goal_half=.55,name='nonlinear')
    result={'source':'conventional raw Delta=.1; task3; optimizer reported success',
            'evaluation':evaluate(data['u'][i:i+1],data['start'][i:i+1],data['goal'][i:i+1],p,65536,811200,.1)[0],
            'decomposition':decompose(data['u'][i],data['start'][i],data['goal'][i],p,32768,982102)}
    (root/'boundary_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
