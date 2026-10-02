"""Freeze and evaluate an opt-in propagation change on fixed controls, CPU only."""
import argparse,datetime,hashlib,json,os,pathlib,time,platform
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from .problem import Problem
from .nominal import dataset
from .guidance import load,sample
from .risk import chance
from .sine_moments import chance_sine
from .test_sine_moments import run_checks
from .evaluate import evaluate

ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def update(out,stage,started,**extra):
    data={'stage':stage,'device':'cpu','elapsed_seconds':time.perf_counter()-started,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
    write(out/'status.json',data);print(json.dumps(data),flush=True)

def freeze(out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=False);cases=[];arrays={};inputs={}
    for layout,strength,checkpoint in [('moderate_noise',.12,'nominal_seed0'),('nonlinear',.35,'nonlinear_seed0')]:
        p=Problem(strength=strength,process_sigma=.04,goal_half=.45 if strength==.12 else .55,name=layout)
        checkpoint_path=ROOT/f'revision/outputs/{checkpoint}/nominal.pt';inputs[str(checkpoint_path.relative_to(ROOT))]=digest(checkpoint_path)
        model,mean,scale=load(str(checkpoint_path),device='cpu')
        for split,n,taskseed,samplingseed in [('tuning',24,2026100301,2026100302),('evaluation',32,2026100401,2026100402)]:
            starts,goals,_=dataset(n,taskseed,strength=strength)
            controls,_=sample(model,mean,scale,starts,goals,p,None,.1,0,seed=samplingseed)
            for i in range(n):
                key=f'{layout}_{split}_{i}';arrays[key]=controls[i].numpy()
                cases.append({'id':key,'split':split,'source':'new frozen B0 CPU sample from existing checkpoint','problem':p.as_dict(),'start':starts[i].tolist(),'goal':goals[i].tolist(),'task_seed':taskseed,'sampling_seed':samplingseed,'reference_seed':2026100500+len(cases),'audit_seed':2026100700+len(cases)})
    source=ROOT/'revision/outputs/conventional/nonlinear_WaypointChanceOptimizer.npz';inputs[str(source.relative_to(ROOT))]=digest(source)
    old=np.load(source,allow_pickle=False);p=Problem(strength=.35,process_sigma=.04,goal_half=.55,name='nonlinear')
    for i in range(len(old['u'])):
        key=f'legacy_optimizer_diagnostic_{i}';arrays[key]=old['u'][i]
        cases.append({'id':key,'split':'diagnostic','source':'September 26 nonlinear conventional optimizer; known counterexample is task 3','problem':p.as_dict(),'start':old['start'][i].tolist(),'goal':old['goal'][i].tolist(),'reference_seed':2026100500+len(cases),'audit_seed':2026100700+len(cases)})
    np.savez_compressed(out/'frozen_controls.npz',**arrays)
    files=[ROOT/'revision/risk.py',ROOT/'revision/evaluate.py',ROOT/'revision/problem.py',ROOT/'revision/sine_moments.py',ROOT/'revision/test_sine_moments.py',pathlib.Path(__file__)]
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':cases,'methods':['jacobian','sine_gaussian_closure'],'reference_draws':8192,'audit_draws':32768,'risk_thresholds':[.05,.10,.20],'tuning_note':'No parameters are fitted. Tuning cases are reported separately and never used to modify this run. Diagnostic cases are not held-out evidence.','event_semantics':'Obstacle at x0..x[H-1], terminal goal at xH, boundary contact nonviolating, independent initial/process Gaussian noise; control feasibility separate.','metrics':['mean trajectory error','full covariance Frobenius error','cross covariance absolute error','exact/tangent marginal MAE','signed Gaussian-minus-empirical-tangent sum','union and geometry excess','fresh audit risk underestimation','confirmed false-safe predictions at each threshold','CPU propagation plus risk runtime','minimum eigenvalue, finiteness, asymmetry and PSD corrections'], 'frozen_controls_sha256':digest(out/'frozen_controls.npz'),'source_sha256':{str(f.relative_to(ROOT)):digest(f) for f in files},'input_sha256':inputs,'device':'cpu','dtype':'float64','threads':2,'runtime_repeats':5,'resource_cap_seconds':180,'resource_cap_artifact_bytes':50_000_000}
    write(out/'protocol.json',protocol);update(out,'frozen_before_evaluation',started,cases=len(cases))

def reference(u,start,goal,p,draws,seed,estimates):
    rng=np.random.default_rng(seed);x=start[None]+p.initial_sigma*rng.normal(size=(draws,2));exact=[];tangent={name:[] for name in estimates};means=[];covariances=[]
    obs=np.asarray(p.obstacles)
    for k in range(p.horizon+1):
        means.append(x.mean(0));covariances.append(np.cov(x.T))
        if k<p.horizon:
            for cx,cy,radius in obs:
                center=np.array([cx,cy]);exact.append(np.linalg.norm(x-center,axis=1)<radius)
                for name,estimate in estimates.items():
                    v=estimate['mean'][k]-center;normal=v/max(np.linalg.norm(v),1e-7)
                    tangent[name].append((x-center)@normal<radius)
            drift=p.strength*np.column_stack((np.sin(2*x[:,1]),np.sin(2*x[:,0])))
            x=x+p.dt*(u[k]+drift)+p.process_sigma*rng.normal(size=(draws,2))
    goal_events=[]
    # Matches the historical goal_parts flatten order: dimension first, sign second.
    for dim in range(2):
        for sign in (-1,1):goal_events.append(sign*(x[:,dim]-goal[dim])>p.goal_half)
    exact+=goal_events
    for name in tangent:tangent[name]+=goal_events
    exact=np.stack(exact,1);true_marginals=exact.mean(0);joint=float(exact.any(1).mean())
    empirical_mean=np.array(means);empirical_cov=np.array(covariances);rows={}
    for name,e in estimates.items():
        tangent_marginals=np.stack(tangent[name],1).mean(0)
        predicted_marginals=np.r_[e['obstacle'].reshape(-1),e['goal'].reshape(-1)]
        mean_error=np.linalg.norm(e['mean']-empirical_mean,axis=1)
        covariance_error=np.linalg.norm(e['covariance']-empirical_cov,axis=(1,2))
        eig=np.linalg.eigvalsh(e['covariance'])
        rows[name]={'mean_error_by_time':mean_error.tolist(),'covariance_error_by_time':covariance_error.tolist(),'mean_error_average':float(mean_error.mean()),'mean_error_max':float(mean_error.max()),'terminal_mean_error':float(mean_error[-1]),'covariance_error_average':float(covariance_error.mean()),'terminal_covariance_error':float(covariance_error[-1]),'cross_covariance_mae':float(np.mean(abs(e['covariance'][:,0,1]-empirical_cov[:,0,1]))),'exact_marginal_mae':float(np.mean(abs(predicted_marginals-true_marginals))),'tangent_marginal_mae':float(np.mean(abs(predicted_marginals-tangent_marginals))),'exact_marginal_sum':float(true_marginals.sum()),'tangent_marginal_sum':float(tangent_marginals.sum()),'gaussian_sum':e['risk'],'union_excess':float(true_marginals.sum()-joint),'geometry_excess':float(tangent_marginals.sum()-true_marginals.sum()),'gaussian_error':float(e['risk']-tangent_marginals.sum()),'minimum_covariance_eigenvalue':float(eig.min()),'asymmetry_max':float(np.max(abs(e['covariance']-e['covariance'].swapaxes(-1,-2)))),'finite':bool(np.isfinite(e['mean']).all() and np.isfinite(e['covariance']).all()),'psd_correction_count':0,'predicted_marginals':predicted_marginals.tolist(),'reference_exact_marginals':true_marginals.tolist(),'reference_tangent_marginals':tangent_marginals.tolist()}
    return {'joint_risk':joint,'means':empirical_mean.tolist(),'covariances':empirical_cov.tolist(),'methods':rows}

def run(out):
    started=time.perf_counter();protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'frozen_controls.npz')==protocol['frozen_controls_sha256']
    for name,value in {**protocol['source_sha256'],**protocol['input_sha256']}.items():assert digest(ROOT/name)==value,('frozen source/input changed',name)
    if (out/'results.json').exists():raise RuntimeError('Refusing to overwrite experiment results')
    update(out,'checks',started);checks=run_checks();bank=np.load(out/'frozen_controls.npz',allow_pickle=False);results=[]
    functions={'jacobian':chance,'sine_gaussian_closure':chance_sine}
    for index,case in enumerate(protocol['cases']):
        if time.perf_counter()-started>150:raise RuntimeError('Experiment time guard reached')
        p=Problem(**case['problem']);u=bank[case['id']];start=np.asarray(case['start']);goal=np.asarray(case['goal']);tu=torch.tensor(u[None],dtype=torch.float64);ts=torch.tensor(start[None],dtype=torch.float64);tg=torch.tensor(goal[None],dtype=torch.float64)
        estimates={};runtimes={}
        for name,function in functions.items():
            with torch.no_grad():
                estimate=function(tu,ts,tg,p);t=time.perf_counter()
                for _ in range(protocol['runtime_repeats']):function(tu,ts,tg,p)
                runtimes[name]=(time.perf_counter()-t)/protocol['runtime_repeats']
            estimates[name]={k:(float(v[0]) if k=='risk' else v[0].numpy()) for k,v in estimate.items()}
        t=time.perf_counter();ref=reference(u,start,goal,p,protocol['reference_draws'],case['reference_seed'],estimates);ref_seconds=time.perf_counter()-t
        t=time.perf_counter();audit=evaluate(u[None],start[None],goal[None],p,protocol['audit_draws'],case['audit_seed'],.1)[0];audit_seconds=time.perf_counter()-t
        control_feasible=bool(np.max(abs(u))<=p.control_limit)
        for name,metrics in ref['methods'].items():
            pred=metrics['gaussian_sum'];metrics['risk_underestimation']=max(0.,audit['risk']-pred);metrics['risk_signed_error']=pred-audit['risk'];metrics['cpu_seconds']=runtimes[name]
            metrics['decisions']={str(d):{'predicted_safe':bool(pred<=d and control_feasible),'audit_confirmed_unsafe':bool(audit['risk_ci'][0]>d),'false_safe_point_estimate':bool(pred<=d and control_feasible and audit['risk']>d),'false_safe_confirmed':bool(pred<=d and control_feasible and audit['risk_ci'][0]>d),'audit_interval_straddles':bool(audit['risk_ci'][0]<=d<=audit['risk_ci'][1])} for d in protocol['risk_thresholds']}
            if not metrics['finite'] or metrics['minimum_covariance_eigenvalue'] < -1e-12:raise AssertionError(('nonfinite/nonPSD',case['id'],name,metrics))
        results.append({'case':case,'control_feasible':control_feasible,'reference':ref,'audit':audit,'reference_seconds':ref_seconds,'audit_seconds':audit_seconds})
        if index%16==0:update(out,'evaluating',started,completed_cases=index+1,total_cases=len(protocol['cases']))
    summary={}
    for split in ('tuning','evaluation','diagnostic'):
        for problem in ('moderate_noise','nonlinear'):
            rows=[r for r in results if r['case']['split']==split and r['case']['problem']['name']==problem]
            if not rows:continue
            group={}
            for name in functions:
                metrics=[r['reference']['methods'][name] for r in rows]
                fields=['mean_error_average','terminal_mean_error','covariance_error_average','terminal_covariance_error','cross_covariance_mae','exact_marginal_mae','tangent_marginal_mae','risk_underestimation','risk_signed_error','cpu_seconds','gaussian_error','union_excess','geometry_excess']
                group[name]={field:float(np.mean([m[field] for m in metrics])) for field in fields}
                group[name]['risk_mae']=float(np.mean([abs(m['risk_signed_error']) for m in metrics]))
                group[name]['worst_risk_underestimation']=float(max(m['risk_underestimation'] for m in metrics))
                group[name]['false_safe_confirmed']={str(d):sum(m['decisions'][str(d)]['false_safe_confirmed'] for m in metrics) for d in protocol['risk_thresholds']}
                group[name]['false_safe_point_estimate']={str(d):sum(m['decisions'][str(d)]['false_safe_point_estimate'] for m in metrics) for d in protocol['risk_thresholds']}
                group[name]['predicted_safe_count']={str(d):sum(m['decisions'][str(d)]['predicted_safe'] for m in metrics) for d in protocol['risk_thresholds']}
                group[name]['minimum_covariance_eigenvalue']=min(m['minimum_covariance_eigenvalue'] for m in metrics)
            summary[f'{split}/{problem}']={'cases':len(rows),'methods':group,'audit_confirmed_unsafe_count':{str(d):sum(r['audit']['risk_ci'][0]>d for r in rows) for d in protocol['risk_thresholds']}}
    report={'protocol_sha256':digest(out/'protocol.json'),'checks':checks,'cases':results,'summary':summary,'elapsed_seconds':time.perf_counter()-started,'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}}
    write(out/'results.json',report);write(out/'summary.json',summary)
    update(out,'completed',started,cases=len(results),artifact_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);ap.add_argument('--output',required=True);args=ap.parse_args();torch.set_num_threads(2)
    (freeze if args.mode=='freeze' else run)(pathlib.Path(args.output))
if __name__=='__main__':main()
