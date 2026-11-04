"""Bounded CPU reproduction; never overwrite September 26 artifacts."""
import argparse,datetime,hashlib,json,os,pathlib,time,platform
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from risk_diffusion.revision.problem import Problem
from risk_diffusion.revision.guidance import load,sample
from risk_diffusion.revision.nominal import dataset,sample as raw_sample
from risk_diffusion.revision.risk import chance,nominal_cost
from risk_diffusion.revision.evaluate import evaluate
from risk_diffusion.revision.calibration import decompose
from risk_diffusion.revision.test_risk import main as test_risk
from risk_diffusion.revision.test_evaluate import main as test_evaluate
from risk_diffusion.revision.planner import plan
from risk_diffusion.revision.projection import project_batch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    root=pathlib.Path(__file__).resolve().parent
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False)
    begin=time.perf_counter();torch.set_num_threads(2)
    def status(stage,**extra):
        payload={'stage':stage,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-begin,'device':'cpu','budget_seconds':600,'budget_bytes':500_000_000,**extra}
        (out/'status.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload),flush=True)
    status('mathematical_checks')
    test_risk();test_evaluate()
    model,mean,scale=load(str(root/'revision/outputs/nominal_seed0/nominal.pt'),device='cpu')
    starts,goals,_=dataset(32,2901);ids=[0,3,11,23];p=Problem(process_sigma=.04,goal_half=.45,name='moderate_noise')
    baseline=json.loads((root/'revision/outputs/guidance/results.json').read_text())
    saved=np.load(root/'revision/outputs/guidance/plans.npz',allow_pickle=False)
    print('Saved plan keys:',saved.files[:8],flush=True)
    status('saved_baseline_replay')
    replay=[]
    for method in ('B0','B2','B3'):
        controls=saved[f'moderate_noise_0_{method}_10']
        for task in ids:
            v=evaluate(controls[task:task+1],starts[task:task+1],goals[task:task+1],p,4096,75100+task,.1)[0]
            a=evaluate(controls[task:task+1],starts[task:task+1],goals[task:task+1],p,8192,198100+task,.1)[0]
            old=next(r for r in baseline['rows'] if r['method']==method and r['problem']=='moderate_noise' and r['training_seed']==0 and r['delta']==.1 and r['task']==task)
            same=v['failures']==old['failures'] and a['risk']==old['audit_risk'] and v['accepted']==old['accepted']
            if not same:raise AssertionError(('saved replay mismatch',method,task,v,old))
            replay.append({'method':method,'task':task,'validation':v,'audit':a,'matches_saved_counts':same})
    status('cpu_existing_checkpoint_smoke')
    s=starts[ids];g=goals[ids];fresh=[];plans={};runtime={}
    for method,kind in [('B0',None),('B2','chance'),('B3','cvar')]:
        t=time.perf_counter();u,meta=sample(model,mean,scale,s,g,p,kind,.1,8,seed=40101,K=256)
        runtime[method]=time.perf_counter()-t;controls=u.numpy();plans[method]=controls
        if method=='B0':
            raw=raw_sample(model,mean,scale,s,g,seed=40101)
            if not torch.allclose(raw,u,atol=1e-6):raise AssertionError('B0 sampler mismatch')
        v=evaluate(controls,s,g,p,4096,202610020,.1)
        a=evaluate(controls,s,g,p,8192,202610120,.1)
        pred=chance(u,torch.tensor(s),torch.tensor(g),p)['risk'].detach().numpy()
        cost=nominal_cost(u,torch.tensor(s),p).detach().numpy()
        for i,task in enumerate(ids):fresh.append({'method':method,'task':task,'predicted_risk':float(pred[i]),'nominal_cost':float(cost[i]),'validation':v[i],'audit':a[i]})
    status('acceptance_and_projection_checks')
    yes=plan(model,mean,scale,s[0],g[0],Problem(process_sigma=0.,initial_sigma=0.),.1,candidates=2,draws=256,seed=212)
    no=plan(model,mean,scale,s[0],g[0],Problem(goal_half=.00001),.01,candidates=2,draws=256,seed=212)
    assert yes['accepted'] and yes['controls'] is not None
    assert not no['accepted'] and no['controls'] is None
    u=torch.zeros((1,20,2),dtype=torch.float64);u[:,:,0]=.75
    st=torch.tensor([[-1.5,.1]],dtype=torch.float64);gt=torch.tensor([[1.5,.1]],dtype=torch.float64)
    projected,diag=project_batch(u,st,gt,Problem(),torch.ones_like(u));_,second=project_batch(projected,st,gt,Problem(),torch.ones_like(u))
    assert diag[0]['feasible'] and not second[0]['attempted']
    status('nonlinear_counterexample_replay')
    b=np.load(root/'revision/outputs/conventional/nonlinear_WaypointChanceOptimizer.npz',allow_pickle=False);i=3
    nonlinear=Problem(strength=.35,process_sigma=.04,goal_half=.55,name='nonlinear')
    boundary={'evaluation':evaluate(b['u'][i:i+1],b['start'][i:i+1],b['goal'][i:i+1],nonlinear,65536,811200,.1)[0],
              'decomposition':decompose(b['u'][i],b['start'][i],b['goal'][i],nonlinear,32768,982102)}
    old=json.loads((root/'revision/outputs/boundary_audit.json').read_text())
    assert boundary['evaluation']['failures']==old['evaluation']['failures']
    assert not boundary['evaluation']['accepted']
    assert abs(boundary['decomposition']['gaussian_sum']-old['decomposition']['gaussian_sum'])<1e-12
    source_hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'revision').glob('*.py')}
    report={'protocol':{'device':'cpu','torch_threads':2,'task_seed':2901,'task_ids':ids,'training_seed':0,'sampling_seed':40101,'delta':.1,'validation_draws':4096,'audit_draws':8192,'fresh_validation_seed':202610020,'fresh_audit_seed':202610120,'note':'Four-task smoke, not a new method-ranking study. CPU RNG need not match historical CUDA generation. Saved controls replay historical evaluation seeds exactly.'},
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},'checks':{'mathematical':True,'b0_equivalence':True,'public_acceptance_rejection':True,'projection_feasibility_idempotence':True},'saved_replay':replay,'fresh_cpu_smoke':fresh,'sampling_seconds':runtime,'boundary_replay':boundary,'source_sha256':source_hashes,'elapsed_seconds':time.perf_counter()-begin}
    (out/'results.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'cpu_plans.npz',start=s,goal=g,**plans)
    status('completed',replay_count=len(replay),new_artifact_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()))
    print('Boundary:',json.dumps(boundary),flush=True)
if __name__=='__main__':main()
