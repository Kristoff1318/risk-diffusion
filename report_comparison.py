"""Tables and all-case plots from the frozen native comparison, no new planning."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
from environment import make_environment

ROOT=Path(__file__).parent
OUT=ROOT/'results/chance_comparison'


def main():
    status=json.loads((OUT/'status.json').read_text())
    assert status['complete']
    protocol=json.loads((OUT/'protocol.json').read_text())
    rows=json.loads((OUT/'results.json').read_text())
    with (OUT/'per_plan.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(['scenario','seed','method','budget','predicted_risk','validation_failures',
                         'validation_upper95','accepted','audit_failures','audit_draws','audit_lower95',
                         'audit_upper95','wall_failures','added_obstacle_failures','goal_failures',
                         'generation_seconds','audit_mean_length','audit_mean_effort'])
        for r in rows:
            writer.writerow([r['scenario'],r['seed'],r['method'],r['budget'],r['surrogate_risk'],
                r['validation']['failures'],r['validation']['upper95'],r.get('accepted'),
                r['audit']['failures'],r['audit']['draws'],*r['audit']['CI95'],
                *[r['audit']['counts'][k] for k in ['wall','added_obstacle','goal_failure']],
                r['generation_seconds'],r['audit']['mean_length'],r['audit']['mean_effort']])
    summary=[]
    for scenario,obstacles in protocol['scenarios'].items():
        for budget in protocol['budgets']:
            for method in ['nominal','chance']:
                group=[r for r in rows if r['scenario']==scenario and r['method']==method
                       and (method=='nominal' or r['budget']==budget)]
                accepted=[r for r in group if r['validation']['upper95']<=budget]
                summary.append(dict(scenario=scenario,budget=budget,method=method,
                    requests=len(group),accepted=len(accepted),
                    all_proposal_audit_risk=float(np.mean([r['audit']['failure_rate'] for r in group])),
                    accepted_audit_risk=float(np.mean([r['audit']['failure_rate'] for r in accepted])) if accepted else None,
                    mean_generation_seconds=float(np.mean([r['generation_seconds'] for r in group])),
                    mean_validation_seconds=float(np.mean([r['validation']['seconds'] for r in group]))))
        env=make_environment(obstacles)
        centers=[env.model.geom_pos[i,:2]-env.model.body_pos[1,:2] for i in range(env.model.ngeom)
                 if (env.model.geom_id2name(i) or '').startswith('wall_')]
        fig,axes=plt.subplots(3,3,figsize=(15,12),sharex=True,sharey=True)
        for i,seed in enumerate(protocol['seeds']):
            for j,budget in enumerate([None]+protocol['budgets']):
                r=next(r for r in rows if r['scenario']==scenario and r['seed']==seed and r['budget']==budget)
                ax=axes[i,j]
                for p in centers:ax.add_patch(Rectangle((p[1]-.5,p[0]-.5),1,1,facecolor='.7'))
                for x,y,radius in obstacles:ax.add_patch(Circle((y,x),radius,color='firebrick',alpha=.65))
                paths=np.load(OUT/f'{r["name"]}_paths.npz')
                execution_name=r['exact_path_audit_reuse'] or r['name']
                execution=np.load(OUT/f'{execution_name}_executions.npz')['audit_example']
                ax.plot(paths['reference'][:,1],paths['reference'][:,0],'--',color='tab:blue',lw=1.1,label='reference')
                ax.plot(execution[:,1],execution[:,0],color='tab:orange',lw=1.4,label='first independent audit execution')
                goal=r['goal'];start=r['start']
                ax.add_patch(Circle((goal[1],goal[0]),.5,color='green',alpha=.15))
                ax.plot(start[1],start[0],'go',ms=4)
                label='Nominal' if budget is None else f'Chance, budget {budget:.0%}'
                accepted=r.get('accepted',r.get('acceptance_by_budget'))
                ax.set_title(f'{label}; seed {seed}\naudit {r["audit"]["failures"]}/{r["audit"]["draws"]}, predicted {r["surrogate_risk"]:.3g}',fontsize=10)
                ax.set_xlim(-.7,11.3);ax.set_ylim(8.3,-.7);ax.set_aspect('equal')
                ax.set_xlabel('qpos y');ax.set_ylabel('qpos x (downward)')
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'{scenario}: same controller, native physical obstacles; all three fixed cases',fontsize=13)
        fig.tight_layout();fig.savefig(OUT/f'{scenario}.png',dpi=140);plt.close(fig);env.close()
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Controller-matched chance-guidance comparison','',
        'This experiment connects the slide-based risk calculation and denoising gradient to the '
        'same state-reference feedback controller used by the verified nominal baseline. '
        'It runs native D4RL Maze2D Large v1 with its actual walls, and separately with three '
        'additional physical MuJoCo obstacles. The checkpoint is unchanged.','',
        '## Main outcomes','',
        'On the wall-only scene, guidance admits 1/3 requests at each budget versus 0/3 for nominal. '
        'The gain is seed 1: nominal has 34/256 independent audit failures (13.28%), while both '
        'guided versions have 0/256. Their separate validation upper limit is 1.16%, below both '
        'budgets; the audit two-sided 95% upper limit is 1.43%. Seed 1 was already nominally '
        'collision-free without noise, so this is an execution-risk improvement. The other two '
        'wall cases retain nominal collisions and fail all audited executions.','',
        'With the three added physical obstacles, no method admits a request. Guidance reduces '
        'some surrogate sums but does not find safe paths in this fixed test. Thus this is one '
        'positive risk-repair case alongside retained failures, not a broadly successful planner.','',
        '| Scene | Budget | Method | Accepted / 3 | Mean audit failure, all proposals | Accepted audit failure | Generation seconds |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        accepted='—' if r['accepted_audit_risk'] is None else f'{r["accepted_audit_risk"]:.2%}'
        lines.append(f'| {r["scenario"]} | {r["budget"]:.0%} | {r["method"]} | {r["accepted"]} | '
                     f'{r["all_proposal_audit_risk"]:.2%} | {accepted} | {r["mean_generation_seconds"]:.3f} |')
    lines+=['','Acceptance uses the independent 256-execution one-sided exact 95% failure upper '
        'bound, not the surrogate or the later audit. One fixed proposal is generated per request; '
        'there is no candidate selection/resampling. Confidence is per proposal, not simultaneous '
        'across the whole study. All-proposal means include every failed case. Nominal is evaluated '
        'against each requested budget without regenerating it.','',
        '## Every fixed case','',
        '| Scene | Seed | Method / budget | Surrogate sum | Validation upper | Accepted | Independent audit risk [95% CI] |',
        '|---|---|---|---:|---:|---|---|']
    for r in rows:
        label=r['method'] if r['budget'] is None else f'chance / {r["budget"]:.0%}'
        accepted=r.get('accepted',r.get('acceptance_by_budget'))
        a=r['audit'];lo,hi=a['CI95']
        lines.append(f'| {r["scenario"]} | {r["seed"]} | {label} | {r["surrogate_risk"]:.6g} | '
                     f'{r["validation"]["upper95"]:.2%} | {accepted} | {a["failure_rate"]:.2%} [{lo:.2%}, {hi:.2%}] |')
    lines+=['','## Mapping to slides 13–14','',
        'The existing checkpoint diffuses joint action/state samples, while execution tracks only '
        'the sampled state reference. Thus the reference parameterizes controller inputs: '
        '`u_physical = clip(p_reference - p + (v_reference - v), -1, 1)`. The stochastic transition '
        'includes this feedback. Guidance differentiates through that same execution chain. '
        'This is an explicit reference-parameterization adapter, not a newly trained control-only prior.','',
        '- Slide 13 steps 1–3: `tracking_dynamics.py` propagates executed means and local covariance; '
        'the Jacobian includes the feedback controller and native action/velocity clipping.\n'
        '- Steps 4–6: wall/circle/goal constraints are linearized at those means; Gaussian CDFs '
        'estimate their violation probabilities. Wall/circle geometry includes agent radius.\n'
        '- Steps 7–8: the untruncated event sum is compared with the requested Delta. '
        'Literal slide indexing uses obstacles at x1..x799 and goal at x800. The independent '
        'validator additionally checks collision at initial and terminal states.\n'
        '- Slide 14 steps 9–10: `softplus(10*(risk_sum-Delta))`, with no clearance penalty or '
        'extra loss.\n'
        '- Steps 11–13: autodiff through the denoiser clean-reference prediction, normalization, '
        'controller, mean/covariance and risk energy modifies the DDPM posterior mean by '
        '`-posterior_variance * strength * gradient`. Guidance is applied in the last 32 of '
        '256 steps, preserving the original random stream and endpoint conditioning.','',
        'The risk candidate is a predicted clean reference rather than a raw noisy reference. '
        'That choice and the 32-stage schedule are disclosed sampling design choices; the '
        'slides do not specify this checkpoint-specific adapter. There is no CVaR, projection, '
        'final trajectory optimizer, new training, obstacle relocation, or task replacement.','',
        '## Execution and independent evidence','',
        'Every path is executed for 800 native MuJoCo steps using the original controller. '
        'The initial state is known and fixed by each native reset seed. After every step, '
        'independent N(0,0.005² I) velocity perturbations are added. Feedback observes the '
        'resulting state exactly. This is a declared synthetic disturbance model, not calibrated '
        'hardware uncertainty. Failure means any wall/added-obstacle contact OR terminal distance '
        'at least 0.5 from the goal. Reaching the goal after touching an obstacle is a failure.','',
        'Scenes: original walls alone, and the same maze plus cylinders of radius 0.35 at '
        '(3,4), (5,8), (7,5) in agent coordinates. These are native collision objects, not '
        'visual overlays. Full sample paths and native executed examples are saved. Only '
        'the first ncon MuJoCo contact entries are inspected.','',
        'A pilot on separate seed 10 compared strengths 0.0002/0.002/0.02, using only deterministic '
        'execution and the surrogate. It selected 0.02; all pilot outcomes are retained. '
        'Noise, obstacles and budgets were not changed after this pilot. The final protocol '
        'and source hashes were saved before the three-seed comparison. Validation and audit '
        'use independent streams; common noise within a case pairs methods. Exact duplicate '
        'path/scene/seed audit reuse, if any, is recorded per row.','',
        f'Completed {status["proposal_count"]} proposals and {status["native_stochastic_executions"]} '
        f'unique stochastic native executions in {status["process_seconds"]:.1f} process seconds. '
        'Each unique path gets 256 admission and 256 independent audit executions. '
        'An audit with 0/256 failures has a two-sided95% upper limit of1.43%, not zero true risk.','',
        '## Qualification and limits','',
        'The displayed surrogate is a sum of event probabilities, so it can exceed 1; it must '
        'not be read as a calibrated whole-trip probability. A small sum is also only an '
        'approximation under nonlinear clipping and geometry. Native validation decides admission.','',
        'Qualification passed native zero-noise trajectory parity (max error2.7e-15), '
        'independent matrix covariance parity, a 30,000-execution covariance check, complete-energy '
        'finite differences, physical added-obstacle collision fixtures, and exact zero-guidance '
        'sampler parity. See ../tracking_qualification.json.','',
        'Gaussian closure and constraint linearization are approximations, especially near '
        'clipping/contact. Summing strongly correlated time events can be conservative. '
        'Deeply colliding candidates can have saturated CDF gradients. These are observed '
        'limits of the tested mechanism; no universal safety or population generalization '
        'claim follows from three fixed starts and one checkpoint. Guidance has more compute '
        'than nominal, so this is not an equal-compute comparison.','',
        '![Wall-only cases](walls.png)','', '![Added-obstacle cases](added_obstacles.png)']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
