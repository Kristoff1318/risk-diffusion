"""Scalar-request, time-matched conventional refinement comparison."""
import argparse,datetime,hashlib,json,os,pathlib,time,platform
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from .problem import Problem
from .nominal import dataset
from .guidance import load,sample
from .risk import chance
from .sine_moments import chance_sine
from .refinement_solver import optimize
from .evaluate import evaluate

ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def status(out,stage,start,**extra):
    record={'stage':stage,'elapsed_seconds':time.perf_counter()-start,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'device':'cpu',**extra};write(out/'status.json',record);print(json.dumps(record),flush=True)
def freeze(out):
    began=time.perf_counter();out.mkdir(parents=True,exist_ok=False);p=Problem(strength=.35,process_sigma=.04,goal_half=.55,name='nonlinear')
    start,goal,waypoint=dataset(8,2026102001,strength=p.strength);checkpoint=ROOT/'revision/outputs/nonlinear_seed0/nominal.pt'
    model,mean,scale=load(str(checkpoint),device='cpu');controls=[]
    for i in range(8):
        u,_=sample(model,mean,scale,start[i:i+1],goal[i:i+1],p,seed=2026102010+i);controls.append(u[0].numpy())
    np.savez_compressed(out/'frozen_tasks.npz',start=start,goal=goal,waypoint=waypoint,diffusion=np.array(controls))
    files=['revision/refinement_experiment.py','revision/refinement_solver.py','revision/guidance.py','revision/risk.py','revision/sine_moments.py','revision/evaluate.py','revision/nominal.py','revision/problem.py']
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'problem':p.as_dict(),'task_ids':list(range(8)),'task_seed':2026102001,'task_split':'new evaluation, disjoint generation seeds from phases2/3; no reused diagnostic cases','initializers':['waypoint','diffusion'],'constraints':['jacobian','closure'],'risk_budgets':[.05,.1],'internal_risk_margin':.8,'objective':'identical nominal path length + .02*dt*sum(u^2) + .01*sum((u[k+1]-u[k])^2)','solver':'SLSQP','solver_iteration_seconds_cap':.5,'solver_cap_mechanism':'Linux real-time signal timer; timeouts retained; postsolve diagnostics/validation outside iteration cap and inside request time','maxiter':300,'ftol':1e-5,'diffusion_steps':64,'checkpoint':str(checkpoint.relative_to(ROOT)),'diffusion_seed_base':2026102010,'validation_draws':4096,'validation_seed_base':2026102200,'audit_draws':8192,'audit_seed_base':2026102400,'acceptance':'one-sided 95% failure upper bound <= requested budget AND control limits; solver success not required; all attempts retained','paired_noise':'identical initial controls/start/goal across constraint variants; per-task validation and independent audit noise paired across variants/initializers/budgets','request_latency':'scalar initializer generation (including checkpoint deserialization for diffusion) + capped solver + postsolve diagnostics + scalar independent validation; no batch amortization; research audit excluded; Python import/process startup reported separately','resource_guard_seconds':60,'resource_cap_seconds':120,'resource_cap_artifact_bytes':15_000_000,'tuning':'no new tuning, no omitted task, fixed .8 margin from existing strongest conventional baseline','frozen_tasks_sha256':digest(out/'frozen_tasks.npz'),'source_sha256':{name:digest(ROOT/name) for name in files},'input_sha256':{str(checkpoint.relative_to(ROOT)):digest(checkpoint)},'threads':2,'device':'cpu'}
    write(out/'protocol.json',protocol);status(out,'frozen_before_evaluation',began,tasks=8,requests=64)
def summary(rows):
    groups={}
    for init in ('waypoint','diffusion'):
        for delta in (.05,.1):
            group={}
            for kind in ('jacobian','closure'):
                r=[a for a in rows if a['initializer']==init and a['delta']==delta and a['constraint']==kind];accepted=[a for a in r if a['validation']['accepted']]
                avg=lambda data,key:float(np.mean([a[key] for a in data])) if data else None
                vals=[a['request_seconds'] for a in r];solve=[a['solver']['solver_seconds'] for a in r]
                group[kind]={'attempts':len(r),'accepted':len(accepted),'rejected':len(r)-len(accepted),'solver_successes':sum(a['solver']['optimizer_success'] for a in r),'timeouts':sum(a['solver']['timed_out'] for a in r),'solver_success_but_rejected':sum(a['solver']['optimizer_success'] and not a['validation']['accepted'] for a in r),'all_proposal_audit_failure':float(np.mean([a['audit']['risk'] for a in r])),'all_proposal_audit_collision':float(np.mean([a['audit']['collision'] for a in r])),'all_proposal_goal_success':float(np.mean([a['audit']['goal'] for a in r])),'all_proposal_nominal_J':float(np.mean([a['solver']['nominal_cost'] for a in r])),'accepted_audit_failure':float(np.mean([a['audit']['risk'] for a in accepted])) if accepted else None,'accepted_nominal_J':float(np.mean([a['solver']['nominal_cost'] for a in accepted])) if accepted else None,'accepted_executed_length':float(np.mean([a['audit']['cost'] for a in accepted])) if accepted else None,'internal_constraint_violations':sum(a['solver']['constraint_violation']>1e-6 for a in r),'false_safe_confirmed':sum(a['solver']['predicted_risk']<=delta and a['control_feasible'] and a['audit']['risk_ci'][0]>delta for a in r),'accepted_audit_unsafe_confirmed':sum(a['audit']['risk_ci'][0]>delta for a in accepted),'accepted_audit_unsafe_point':sum(a['audit']['risk']>delta for a in accepted),'request_seconds_mean':float(np.mean(vals)),'request_seconds_min':min(vals),'request_seconds_median':float(np.median(vals)),'request_seconds_p90':float(np.quantile(vals,.9)),'request_seconds_max':max(vals),'solver_seconds_mean':float(np.mean(solve)),'solver_seconds_median':float(np.median(solve)),'solver_seconds_max':max(solve),'gradient_evaluations_mean':float(np.mean([a['solver']['gradient_evaluations'] for a in r]))}
            pairs=[]
            for i in range(8):
                b=next(a for a in rows if a['initializer']==init and a['delta']==delta and a['constraint']=='jacobian' and a['task']==i);c=next(a for a in rows if a['initializer']==init and a['delta']==delta and a['constraint']=='closure' and a['task']==i)
                pairs.append({'task':i,'jacobian_accepted':b['validation']['accepted'],'closure_accepted':c['validation']['accepted'],'both_accepted':b['validation']['accepted'] and c['validation']['accepted'],'nominal_J_difference_closure_minus_jacobian':c['solver']['nominal_cost']-b['solver']['nominal_cost'],'audit_failure_difference_closure_minus_jacobian':c['audit']['risk']-b['audit']['risk'],'request_seconds_difference_closure_minus_jacobian':c['request_seconds']-b['request_seconds']})
            common=[a for a in pairs if a['both_accepted']]
            groups[f'{init}/delta_{delta}']={'methods':group,'pairs':pairs,'paired_summary':{'acceptance_gained':sum(not a['jacobian_accepted'] and a['closure_accepted'] for a in pairs),'acceptance_lost':sum(a['jacobian_accepted'] and not a['closure_accepted'] for a in pairs),'common_accepted':len(common),'common_accepted_J_difference':float(np.mean([a['nominal_J_difference_closure_minus_jacobian'] for a in common])) if common else None,'all_task_risk_difference':float(np.mean([a['audit_failure_difference_closure_minus_jacobian'] for a in pairs]))}}
    return groups
def run(out):
    began=time.perf_counter();protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'frozen_tasks.npz')==protocol['frozen_tasks_sha256']
    for name,value in {**protocol['source_sha256'],**protocol['input_sha256']}.items():assert digest(ROOT/name)==value,name
    if (out/'results.json').exists():raise RuntimeError('Refusing to overwrite results')
    p=Problem(**protocol['problem']);bank=np.load(out/'frozen_tasks.npz',allow_pickle=False);rows=[];plans={}
    for initializer in protocol['initializers']:
        for delta in protocol['risk_budgets']:
            for kind in protocol['constraints']:
                status(out,'running',began,initializer=initializer,delta=delta,constraint=kind,completed_requests=len(rows))
                for task in protocol['task_ids']:
                    if time.perf_counter()-began>protocol['resource_guard_seconds']:raise RuntimeError('global wall resource guard reached')
                    start=bank['start'][task];goal=bank['goal'][task]
                    request_began=time.perf_counter();initial_began=request_began
                    if initializer=='diffusion':
                        model,mean,scale=load(str(ROOT/protocol['checkpoint']),device='cpu');initial,_=sample(model,mean,scale,start[None],goal[None],p,seed=protocol['diffusion_seed_base']+task);initial=initial[0].numpy()
                    else:
                        ss,gg,waypoint=dataset(8,protocol['task_seed'],strength=p.strength);initial=waypoint[task]
                        assert np.array_equal(start,ss[task]) and np.array_equal(goal,gg[task])
                    initial_seconds=time.perf_counter()-initial_began
                    assert np.array_equal(initial,bank[initializer][task]),('initial control parity',initializer,task)
                    controls,info=optimize(initial,start,goal,p,delta*protocol['internal_risk_margin'],kind,seconds=.5,maxiter=300,ftol=1e-5)
                    t=time.perf_counter();validation=evaluate(controls[None],start[None],goal[None],p,4096,protocol['validation_seed_base']+task,delta)[0];validation_seconds=time.perf_counter()-t;request_seconds=time.perf_counter()-request_began
                    t=time.perf_counter();audit=evaluate(controls[None],start[None],goal[None],p,8192,protocol['audit_seed_base']+task,delta)[0];audit_seconds=time.perf_counter()-t
                    with torch.no_grad():
                        u=torch.tensor(controls[None],dtype=torch.float64);st=torch.tensor(start[None],dtype=torch.float64);gt=torch.tensor(goal[None],dtype=torch.float64)
                        jac=float(chance(u,st,gt,p)['risk'][0]);closure=float(chance_sine(u,st,gt,p)['risk'][0])
                    row={'initializer':initializer,'delta':delta,'constraint':kind,'task':task,'solver':info,'validation':validation,'audit':audit,'control_feasible':bool(np.max(abs(controls))<=p.control_limit),'control_violation':float(max(0.,np.max(abs(controls))-p.control_limit)),'jacobian_predicted_risk':jac,'closure_predicted_risk':closure,'initialization_seconds':initial_seconds,'validation_seconds':validation_seconds,'request_seconds':request_seconds,'research_audit_seconds':audit_seconds}
                    rows.append(row);plans[f'{initializer}_{int(delta*100)}_{kind}_{task}']=controls
                write(out/'progress_rows.json',{'rows':rows})
    result={'protocol_sha256':digest(out/'protocol.json'),'rows':rows,'summary':summary(rows),'elapsed_seconds':time.perf_counter()-began,'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},'initialization_parity_all_requests':True}
    write(out/'results.json',result);write(out/'summary.json',result['summary']);np.savez_compressed(out/'plans.npz',**plans);(out/'progress_rows.json').unlink()
    status(out,'completed',began,requests=len(rows),artifact_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);ap.add_argument('--output',required=True);args=ap.parse_args();torch.set_num_threads(2)
    (freeze if args.mode=='freeze' else run)(pathlib.Path(args.output))
if __name__=='__main__':main()
