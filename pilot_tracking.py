"""Small, disclosed guidance-strength pilot; no held-out risk audits."""
import json,time
from pathlib import Path
import numpy as np
import torch
from runtime import ASSETS,load_policy
from environment import make_environment,ADDED_OBSTACLES
from chance_guidance import TrackingChanceModel,sample_guided
from nominal_baseline import sample
from evaluate_tracking import execute_reference


def main():
    root=Path(__file__).parent;out=root/'results/chance_comparison/pilot'
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(seed=10,budget=.1,noise_std=.005,gamma=10.,guidance_steps=32,
                  strengths=[.0002,.002,.02],execution_steps=800,
                  scenarios={'walls':[],'added_obstacles':ADDED_OBSTACLES},
                  selection='fewest zero-noise union failures, then smallest sum surrogate risk; tie smaller strength',
                  independent_stochastic_executions=0)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    torch.set_num_threads(2);policy=load_policy();rows=[]
    for scenario,obstacles in protocol['scenarios'].items():
        env=make_environment(obstacles);env.seed(10);start=env.reset();goal=np.array(env.get_target(),dtype=float)
        model=TrackingChanceModel(env,np.load(ASSETS/'normalization.npz'),obstacles)
        for strength in [0.]+protocol['strengths']:
            if strength:
                reference,seconds,trace=sample_guided(policy,model,start,goal,10,.1,strength=strength)
            else:
                paths,seconds=sample(policy,start,goal,10);reference=paths.observations[0];trace=[]
            result=execute_reference(env,reference,start,goal,save=True)
            risk=model.risk_reference(torch.tensor(reference[None],dtype=torch.float64),
                torch.tensor(start[None]),torch.tensor(goal[None]))
            trajectory=result.pop('trajectory')
            row=dict(scenario=scenario,strength=strength,surrogate_risk=float(risk.total_risk),
                     generation_seconds=seconds,nominal=result,
                     max_gradient_norm=max([r['gradient_norm'] for r in trace] or [0.]))
            rows.append(row)
            np.savez_compressed(out/f'{scenario}_{strength}.npz',reference=reference,execution=trajectory)
            (out/f'{scenario}_{strength}_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
            (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(row),flush=True)
        env.close()
    scores=[]
    for strength in protocol['strengths']:
        group=[r for r in rows if r['strength']==strength]
        scores.append((sum(r['nominal']['failure'] for r in group),sum(r['surrogate_risk'] for r in group),strength))
    selected=min(scores)[2]
    decision=dict(selected_strength=selected,scores=scores,scope='pilot seed10 only; no held-out audit or noise tuning')
    (out/'DECISION.json').write_text(json.dumps(decision,indent=2)+'\n');print(json.dumps(decision),flush=True)


if __name__=='__main__':main()
