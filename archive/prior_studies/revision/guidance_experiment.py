"""Frozen, no-training CPU guidance ablation with bounded conventional planning."""
import argparse,datetime,hashlib,json,os,pathlib,time,platform
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from .problem import Problem
from .guidance import load,sample as historical_sample
from .guidance_closure import sample
from .nominal import dataset
from .risk import chance,nominal_cost
from .sine_moments import chance_sine
from .evaluate import evaluate
from .bounded_conventional import optimize
from .compare_sine import reference

ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def status(out,stage,start,**extra):
    data={'stage':stage,'elapsed_seconds':time.perf_counter()-start,'device':'cpu','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
    write(out/'status.json',data);print(json.dumps(data),flush=True)

def freeze(out):
    began=time.perf_counter();out.mkdir(parents=True,exist_ok=False);arrays={};layouts=[];inputs={}
    for index,(layout,strength,checkpoint) in enumerate([('moderate_noise',.12,'nominal_seed0'),('nonlinear',.35,'nonlinear_seed0')]):
        p=Problem(strength=strength,process_sigma=.04,goal_half=.45 if strength==.12 else .55,name=layout)
        cp=ROOT/f'revision/outputs/{checkpoint}/nominal.pt';inputs[str(cp.relative_to(ROOT))]=digest(cp)
        model,mean,scale=load(str(cp),device='cpu');taskseed=2026101001+index;samplingseed=2026101011+index
        starts,goals,_=dataset(16,taskseed,strength=strength)
        raw,_=sample(model,mean,scale,starts,goals,p,None,.1,8,seed=samplingseed)
        arrays[layout+'_start']=starts;arrays[layout+'_goal']=goals;arrays[layout+'_nominal']=raw.numpy()
        layouts.append({'problem':p.as_dict(),'checkpoint':str(cp.relative_to(ROOT)),'task_seed':taskseed,'sampling_seed':samplingseed,'task_ids':list(range(16)),'validation_seed':2026101100+100*index,'audit_seed':2026101200+100*index,'decomposition_seed':2026101400+100*index})
    np.savez_compressed(out/'frozen_tasks.npz',**arrays)
    files=['revision/guidance.py','revision/risk.py','revision/evaluate.py','revision/sine_moments.py','revision/guidance_closure.py','revision/bounded_conventional.py','revision/guidance_experiment.py','revision/compare_sine.py','revision/conventional.py']
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'layouts':layouts,'task_split':'new final evaluation; disjoint task-generation seeds from prior moment/tuning experiments','methods':{'nominal':None,'chance':'chance','closure_chance':'closure','cvar':'cvar','conventional':'SLSQP from the same frozen nominal diffusion controls'},'delta':[.05,.10],'diffusion_steps':64,'guidance_alpha_min':.5,'guidance_steps_expected':32,'lambda':8.,'gamma':30,'cvar_K':256,'cost_weight':.1,'cost_objective':'nominal path length + .02*dt*sum(u^2) + .01*sum((u[k+1]-u[k])^2); same J for all scoring/optimization','cost_note':'All guided methods use risk energy + .1*J; nominal is an unrefined baseline. Conventional directly minimizes J under buffered chance constraint. No final-task tuning. Same lambda does not equalize gradient scale or CPU work.','validation_draws':4096,'validation_confidence':.95,'audit_draws':8192,'decomposition_draws':2048,'acceptance':'one-sided 95% failure upper bound <= delta and control bounds; one proposal per task/method; no audit-dependent selection','event_semantics':'obstacles x0..x[H-1], goal xH, physical Gaussian initial/process noise, discrete time','conventional':{'internal_risk_margin':.8,'maxiter':300,'ftol':1e-5,'wall_seconds_per_plan':.6,'failure_policy':'last evaluated control candidate retained; independent validator still decides admission; timeout and solver failure reported'},'paired_samples':'all methods use identical initial diffusion RNG seeds and task batches; same validation/audit RNG per task across methods and deltas; independent audit seed; CVaR fixed guidance noise is separate','measurements':['all-proposal collision/goal/combined failure and per-plan CI','accept/reject counts','accepted-only audit risks, CI and J','task-level paired risk/cost differences and common-accepted comparisons','gradient time/norm and denoising/sample counts','end-to-end planning+validation latency, audit/decomposition overhead separately','optimizer failures/timeouts and residual calibration','Gaussian/tangent/union error decomposition'],'resource_cap_seconds':180,'resource_cap_artifact_bytes':25_000_000,'execution_guard_seconds':135,'device':'cpu','threads':2,'frozen_tasks_sha256':digest(out/'frozen_tasks.npz'),'source_sha256':{name:digest(ROOT/name) for name in files},'input_sha256':inputs}
    write(out/'protocol.json',protocol);status(out,'frozen_before_evaluation',began,tasks=32,proposals=320)

def summaries(rows):
    output={}
    for layout in ('moderate_noise','nonlinear'):
        for delta in (.05,.1):
            group={}
            for method in ('nominal','chance','closure_chance','cvar','conventional'):
                r=[a for a in rows if a['layout']==layout and a['delta']==delta and a['method']==method];accepted=[a for a in r if a['validation']['accepted']]
                def avg(items,fun):return float(np.mean([fun(a) for a in items])) if items else None
                group[method]={'proposals':len(r),'accepted':len(accepted),'rejected':len(r)-len(accepted),'all_proposal_audit_failure':avg(r,lambda a:a['audit']['risk']),'all_proposal_audit_collision':avg(r,lambda a:a['audit']['collision']),'all_proposal_audit_goal_success':avg(r,lambda a:a['audit']['goal']),'accepted_audit_failure':avg(accepted,lambda a:a['audit']['risk']),'accepted_audit_collision':avg(accepted,lambda a:a['audit']['collision']),'accepted_audit_goal_success':avg(accepted,lambda a:a['audit']['goal']),'accepted_nominal_J':avg(accepted,lambda a:a['nominal_cost']),'accepted_executed_length':avg(accepted,lambda a:a['audit']['cost']),'accepted_audit_above_budget_point':sum(a['audit']['risk']>delta for a in accepted),'accepted_audit_above_budget_confirmed':sum(a['audit']['risk_ci'][0]>delta for a in accepted),'mean_whole_query_seconds':avg(r,lambda a:a['whole_query_seconds']),'mean_planning_seconds':avg(r,lambda a:a['planning_seconds']),'mean_validation_seconds':avg(r,lambda a:a['validation_seconds']),'mean_gradient_seconds':avg(r,lambda a:a.get('gradient_seconds',0.)),'optimizer_failures':sum(not a['optimizer']['optimizer_success'] for a in r if 'optimizer' in a),'optimizer_timeouts':sum('budget expired' in a['optimizer']['message'] for a in r if 'optimizer' in a),'active_surrogate_false_safe_confirmed':sum(a['active_probability_surrogate'] is not None and a['active_probability_surrogate']<=delta and a['control_feasible'] and a['audit']['risk_ci'][0]>delta for a in r)}
            # Same task pairing, all tasks retained. Cost among common-accepted plans avoids acceptance-composition confounding.
            pairs=[]
            for task in range(16):
                pair={method:next(a for a in rows if a['layout']==layout and a['delta']==delta and a['task']==task and a['method']==method) for method in group}
                b=pair['chance'];c=pair['closure_chance'];pairs.append({'task':task,'chance_accepted':b['validation']['accepted'],'closure_accepted':c['validation']['accepted'],'audit_failure_difference_closure_minus_chance':c['audit']['risk']-b['audit']['risk'],'nominal_J_difference_closure_minus_chance':c['nominal_cost']-b['nominal_cost'],'both_accepted':b['validation']['accepted'] and c['validation']['accepted']})
            common=[a for a in pairs if a['both_accepted']]
            output[f'{layout}/delta_{delta}']={'methods':group,'closure_vs_chance':{'pairs':pairs,'acceptance_gained':sum(not a['chance_accepted'] and a['closure_accepted'] for a in pairs),'acceptance_lost':sum(a['chance_accepted'] and not a['closure_accepted'] for a in pairs),'mean_audit_failure_difference_all_tasks':float(np.mean([a['audit_failure_difference_closure_minus_chance'] for a in pairs])),'common_accepted_tasks':len(common),'mean_J_difference_common_accepted':float(np.mean([a['nominal_J_difference_closure_minus_chance'] for a in common])) if common else None}}
    return output

def run(out):
    began=time.perf_counter();protocol=json.loads((out/'protocol.json').read_text())
    assert digest(out/'frozen_tasks.npz')==protocol['frozen_tasks_sha256']
    for name,value in {**protocol['source_sha256'],**protocol['input_sha256']}.items():assert digest(ROOT/name)==value,('frozen input changed',name)
    if (out/'results.json').exists():raise RuntimeError('Refusing to overwrite completed results')
    bank=np.load(out/'frozen_tasks.npz',allow_pickle=False);rows=[];saved={};checks=[]
    def guard():
        if time.perf_counter()-began>protocol['execution_guard_seconds']:raise RuntimeError('aggregate experiment wall guard reached')
    status(out,'parity_checks',began)
    for layout in protocol['layouts']:
        p=Problem(**layout['problem']);name=p.name;starts=bank[name+'_start'];goals=bank[name+'_goal'];model,mean,scale=load(str(ROOT/layout['checkpoint']),device='cpu')
        # Validate the matched loop against unmodified historical sampling for all old paths.
        for kind in (None,'chance','cvar'):
            a,_=sample(model,mean,scale,starts[:2],goals[:2],p,kind,.1,8,seed=layout['sampling_seed'],cost_weight=.1)
            b,_=historical_sample(model,mean,scale,starts[:2],goals[:2],p,kind,.1,8,seed=layout['sampling_seed'],cost_weight=.1)
            assert torch.allclose(a,b,atol=1e-6,rtol=0),(name,kind,float((a-b).abs().max()))
            checks.append({'layout':name,'kind':kind,'historical_sampler_max_abs_difference':float((a-b).abs().max())})
        for delta in protocol['delta']:
            base_planning=None
            for method,kind in [('nominal',None),('chance','chance'),('closure_chance','closure'),('cvar','cvar'),('conventional','SLSQP')]:
                guard();status(out,'running',began,layout=name,delta=delta,method=method,completed_proposals=len(rows))
                optimizer_infos=[]
                if method=='conventional':
                    controls=[]
                    for i in range(16):
                        guard();u,info=optimize(bank[name+'_nominal'][i],starts[i],goals[i],p,delta*.8,seconds=.6,maxiter=300,ftol=1e-5);controls.append(u);optimizer_infos.append(info)
                    controls=np.array(controls);meta={'gradient_seconds':0.};planning=[info['optimizer_seconds']+base_planning for info in optimizer_infos]
                else:
                    t=time.perf_counter();u,meta=sample(model,mean,scale,starts,goals,p,kind,delta,8,seed=layout['sampling_seed'],cost_weight=.1);elapsed=time.perf_counter()-t
                    controls=u.numpy();planning=[elapsed/16]*16
                    if method=='nominal':
                        assert np.array_equal(controls,bank[name+'_nominal']);base_planning=elapsed/16
                t=time.perf_counter();validation=evaluate(controls,starts,goals,p,4096,layout['validation_seed'],delta);validation_seconds=(time.perf_counter()-t)/16
                t=time.perf_counter();audit=evaluate(controls,starts,goals,p,8192,layout['audit_seed'],delta);audit_seconds=(time.perf_counter()-t)/16
                with torch.no_grad():
                    tu=torch.tensor(controls,dtype=torch.float64);ts=torch.tensor(starts,dtype=torch.float64);tg=torch.tensor(goals,dtype=torch.float64)
                    costs=nominal_cost(tu,ts,p).numpy();jac=chance(tu,ts,tg,p);closure=chance_sine(tu,ts,tg,p)
                saved[f'{name}_{int(delta*100)}_{method}']=controls
                for i in range(16):
                    estimates={label:{k:(float(v[i]) if k=='risk' else v[i].numpy()) for k,v in value.items()} for label,value in [('jacobian',jac),('sine_gaussian_closure',closure)]}
                    t=time.perf_counter();decomposition=reference(controls[i],starts[i],goals[i],p,2048,layout['decomposition_seed']+i,estimates)
                    # Save compact decomposition, not full event arrays from this small diagnostic stream.
                    compact={label:{k:m[k] for k in ('gaussian_sum','tangent_marginal_sum','exact_marginal_sum','gaussian_error','geometry_excess','union_excess','tangent_marginal_mae','terminal_mean_error','terminal_covariance_error','minimum_covariance_eigenvalue','finite')} for label,m in decomposition['methods'].items()}
                    assert all(m['finite'] and m['minimum_covariance_eigenvalue']>=-1e-8 for m in compact.values())
                    active=float(closure['risk'][i]) if method=='closure_chance' else float(jac['risk'][i]) if method in ('chance','conventional') else None
                    row={'layout':name,'task':i,'delta':delta,'method':method,'validation':validation[i],'audit':audit[i],'control_feasible':bool(np.max(abs(controls[i]))<=p.control_limit),'nominal_cost':float(costs[i]),'jacobian_predicted_risk':float(jac['risk'][i]),'closure_predicted_risk':float(closure['risk'][i]),'active_probability_surrogate':active,'planning_seconds':planning[i],'validation_seconds':validation_seconds,'whole_query_seconds':planning[i]+validation_seconds,'audit_seconds_research_only':audit_seconds,'decomposition_seconds_research_only':time.perf_counter()-t,'decomposition':compact,'guidance_steps':meta.get('guidance_steps',0),'diffusion_steps':meta.get('diffusion_steps',64),'guidance_rollouts_per_plan':meta.get('guidance_rollouts_per_plan',0),'gradient_seconds':meta.get('energy_and_backward_seconds_batch',0.)/16,'gradient_trace':meta.get('trace',[])}
                    if optimizer_infos:
                        row['optimizer']=optimizer_infos[i]
                        row['gradient_seconds']=optimizer_infos[i]['energy_and_backward_seconds']
                    rows.append(row)
                write(out/'progress_rows.json',{'rows':rows})
    result={'protocol_sha256':digest(out/'protocol.json'),'parity_checks':checks,'rows':rows,'summary':summaries(rows),'elapsed_seconds':time.perf_counter()-began,'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}}
    write(out/'results.json',result);write(out/'summary.json',result['summary']);np.savez_compressed(out/'plans.npz',**saved)
    (out/'progress_rows.json').unlink()
    status(out,'completed',began,proposals=len(rows),artifact_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);ap.add_argument('--output',required=True);args=ap.parse_args();torch.set_num_threads(2)
    (freeze if args.mode=='freeze' else run)(pathlib.Path(args.output))
if __name__=='__main__':main()
