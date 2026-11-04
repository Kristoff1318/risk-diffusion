"""Paired nominal versus slides-13/14 guidance, audited in native MuJoCo."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from runtime import ASSETS,PROJECT,load_policy,sha
from environment import make_environment,ADDED_OBSTACLES
from chance_guidance import TrackingChanceModel,sample_guided
from nominal_baseline import sample
from evaluate_tracking import execute_reference,evaluate

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/chance_comparison'


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Comparison already specified; preserve existing outcomes.')
    decision=json.loads((OUT/'pilot/DECISION.json').read_text())
    sources=[ROOT/n for n in ['chance_guidance.py','tracking_dynamics.py','environment.py',
              'evaluate_tracking.py','compare_tracking.py','runtime.py','nominal_baseline.py']]
    protocol=dict(seeds=[0,1,2],budgets=[.05,.1],noise_std=.005,gamma=10.,guidance_steps=32,
                  strength=decision['selected_strength'],execution_steps=800,
                  validation_draws=256,audit_draws=256,
                  scenarios={'walls':[],'added_obstacles':ADDED_OBSTACLES},
                  validation_seed_base=2026110400,audit_seed_base=2026119400,
                  acceptance='single prespecified proposal; one-sided exact95% upper <= requested budget',
                  risk_event='any native wall/added-obstacle contact OR terminal goal distance>=0.5',
                  goal_metric='terminal goal at800 for risk; ever-reached goal also reported',
                  guidance_candidate='denoiser clean reference; gradient through denoiser and actual feedback model',
                  slide_obstacle_index='x1..x799; goal x800. Independent validator also checks x0 and x800 collisions.',
                  source_sha256={str(p.relative_to(PROJECT)):sha(p) for p in sources},
                  qualification=json.loads((ROOT/'results/tracking_qualification.json').read_text()),
                  original_checkpoint_sha256=sha(ASSETS/'state_1960000.pt'))
    save('protocol.json',protocol)  # Frozen before opening comparison outcomes.
    torch.set_num_threads(2);policy=load_policy();rows=[];generation=[];cache={}
    began=time.perf_counter()
    # First generate all paired proposals and preserve them before stochastic audits.
    for scenario,obstacles in protocol['scenarios'].items():
        env=make_environment(obstacles)
        model=TrackingChanceModel(env,np.load(ASSETS/'normalization.npz'),obstacles,protocol['noise_std'])
        for seed in protocol['seeds']:
            env.seed(seed);start=env.reset();goal=np.asarray(env.get_target(),dtype=float)
            for budget in [None]+protocol['budgets']:
                method='nominal' if budget is None else 'chance'
                name=f'{scenario}_s{seed}_{method}'+('' if budget is None else f'_d{budget}')
                if budget is None:
                    paths,seconds=sample(policy,start,goal,seed);reference=paths.observations[0];trace=[]
                else:
                    reference,seconds,trace=sample_guided(policy,model,start,goal,seed,budget,
                        strength=protocol['strength'],guidance_steps=32,gamma=10.)
                with torch.no_grad():
                    risk=model.risk_reference(torch.tensor(reference[None],dtype=torch.float64),
                                              torch.tensor(start[None]),torch.tensor(goal[None]))
                nominal=execute_reference(env,reference,start,goal,save=True)
                path=nominal.pop('trajectory')
                row=dict(name=name,scenario=scenario,seed=seed,method=method,budget=budget,
                         reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                         start=start.tolist(),goal=goal.tolist(),generation_seconds=seconds,
                         surrogate_risk=float(risk.total_risk),surrogate_wall=float(risk.wall_risk),
                         surrogate_added=float(risk.added_obstacle_risk),surrogate_goal=float(risk.goal_risk),
                         nominal=nominal,trace=trace)
                np.savez_compressed(OUT/f'{name}_paths.npz',reference=reference,nominal=path,
                                    predicted_mean=risk.mean_states[0].numpy(),
                                    predicted_position_variance=risk.position_variances[0].numpy())
                rows.append(row);save('generation.json',rows)
                print('GENERATED',name,'risk',row['surrogate_risk'],'nominal_failure',nominal['failure'],flush=True)
        env.close()
    # Audit streams are independent of optimization, of the pilot, and of one another.
    for scenario_index,(scenario,obstacles) in enumerate(protocol['scenarios'].items()):
        env=make_environment(obstacles)
        for row in [r for r in rows if r['scenario']==scenario]:
            reference=np.load(OUT/f'{row["name"]}_paths.npz')['reference']
            start=np.asarray(row['start']);goal=np.asarray(row['goal'])
            key=(scenario,row['seed'],row['reference_sha256'])
            if key in cache:
                prior=cache[key]
                row.update(validation=prior['validation'],audit=prior['audit'],
                           validation_seconds_charged=0.,audit_seconds_charged=0.,
                           exact_path_audit_reuse=prior['name'])
            else:
                base=1000*scenario_index+100*row['seed']
                validation,vflags,vexample=evaluate(env,reference,start,goal,protocol['noise_std'],
                    protocol['validation_draws'],protocol['validation_seed_base']+base)
                print('VALIDATED',row['name'],validation,flush=True)
                audit,aflags,aexample=evaluate(env,reference,start,goal,protocol['noise_std'],
                    protocol['audit_draws'],protocol['audit_seed_base']+base)
                row.update(validation=validation,audit=audit,
                           validation_seconds_charged=validation['seconds'],audit_seconds_charged=audit['seconds'],
                           exact_path_audit_reuse=None)
                np.savez_compressed(OUT/f'{row["name"]}_executions.npz',validation_flags=vflags,
                                    audit_flags=aflags,validation_example=vexample,audit_example=aexample)
                cache[key]=row
            if row['budget'] is not None:
                row['surrogate_pass']=row['surrogate_risk']<=row['budget']
                row['accepted']=row['validation']['upper95']<=row['budget']
                row['confirmed_false_safe']=row['surrogate_pass'] and row['audit']['CI95'][0]>row['budget']
            else:
                row['acceptance_by_budget']={str(b):row['validation']['upper95']<=b for b in protocol['budgets']}
            save('results.json',rows)
            print('AUDITED',row['name'],row['audit']['failure_rate'],'accepted',row.get('accepted'),
                  'reuse',row['exact_path_audit_reuse'],flush=True)
        env.close()
    save('status.json',dict(complete=True,proposal_count=len(rows),unique_audited_paths=len(cache),
                           native_stochastic_executions=len(cache)*512,
                           process_seconds=time.perf_counter()-began))
    print('COMPLETE',json.loads((OUT/'status.json').read_text()),flush=True)


if __name__=='__main__':main()
