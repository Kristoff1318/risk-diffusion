"""Regenerate research plots and machine-readable summaries from saved outputs."""
import json,time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
from .problem import Problem
from .risk import simulate
from .guidance import load,sample
from .nominal import dataset
from .evaluate import evaluate

ROOT=Path('risk_diffusion/revision/outputs')
def read(name):return json.loads((ROOT/name/'results.json').read_text())
def aggregate(rows):
    valid=[x for x in rows if x['accepted']]
    return {'n':len(rows),'attempt_risk':float(np.mean([x['audit_risk'] for x in rows])),
            'accepted':len(valid),'accepted_audit_risk':float(np.mean([x['audit_risk'] for x in valid])) if valid else None,
            'false_accept_point_estimate':sum(x['audit_risk']>x['delta'] for x in valid),
            'validation_goal':float(np.mean([x['goal'] for x in rows])),
            'validation_executed_length':float(np.mean([x['cost'] for x in rows])),
            'validation_cvar':float(np.mean([x['cvar'] for x in rows])),
            'validation_positive_violation':float(np.mean([x['mean_positive_violation'] for x in rows])),
            'surrogate_false_safe_point':sum(x['predicted_risk']<=x['delta'] and x['audit_risk']>x['delta'] for x in rows),
            'cost_all':float(np.mean([x['planning_cost'] for x in rows])),
            'cost_accepted':float(np.mean([x['planning_cost'] for x in valid])) if valid else None}

def main():
    torch.set_num_threads(2);out=ROOT/'summary';out.mkdir(exist_ok=True)
    guide=read('guidance');proj=read('projection');cond=read('conditional');ordinary=read('conditional_ordinary');sweep=read('sweeps');summ={}
    for label,data in [('guidance',guide),('projection',proj),('conditional',cond),('conditional_ordinary',ordinary)]:
        table=[]
        for m in sorted(set(x['method'] for x in data['rows'])):
            for problem in sorted(set(x.get('problem','moderate_noise') for x in data['rows'])):
                for delta in [.05,.1,.2]:
                    rows=[x for x in data['rows'] if x['method']==m and x['delta']==delta and x.get('problem','moderate_noise')==problem]
                    if rows:table.append(dict(method=m,problem=problem,delta=delta,**aggregate(rows)))
        summ[label]=table
    (out/'summary.json').write_text(json.dumps(summ,indent=2))
    colors={'B0':'#53616e','B1':'#999999','B2':'#087da3','B3':'#cc6c2c','B4':'#539546','B5':'#8760a0'}
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for m in ['B0','B2','B3']:
        rr=[x for x in summ['guidance'] if x['method']==m and x['problem']=='moderate_noise']
        axs[0].plot([x['delta'] for x in rr],[x['attempt_risk'] for x in rr],'o-',label=m,color=colors[m])
        axs[1].plot([x['delta'] for x in rr],[x['accepted']/x['n'] for x in rr],'o-',label=m,color=colors[m])
    axs[0].plot([.05,.2],[.05,.2],'k--',alpha=.45,label='risk bound')
    axs[0].set(xlabel='Requested Δ',ylabel='Independent audit failure rate',title='All proposals, including rejected attempts')
    axs[1].set(xlabel='Requested Δ',ylabel='Fraction passing confidence-bound validation',ylim=(.7,1.02),title='96 proposals per method / budget')
    for ax in axs:ax.legend();ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/'requested_risk.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    cal=proj['calibration']
    for env,c in [('moderate_noise','#087da3'),('staggered','#539546'),('nonlinear','#cc6c2c')]:
        x=[q for q in cal if q['problem']==env];axs[0].scatter([q['gaussian_sum'] for q in x],[q['true_risk'] for q in x],label=env,s=20,color=c)
    boundary=json.loads((ROOT/'boundary_audit.json').read_text());b=boundary['decomposition']
    axs[0].scatter([b['gaussian_sum']],[boundary['evaluation']['risk']],marker='x',s=85,color='red',label='optimized boundary counterexample')
    axs[0].plot([0,.25],[0,.25],'k--',alpha=.4);axs[0].set(xlabel='Gaussian union surrogate',ylabel='Independent execution risk',title='Calibration on planner outputs');axs[0].legend(fontsize=7)
    labels=['Joint event','Sum of exact\nmarginals','Sum of tangent\nmarginals','Local Gaussian\nsum']
    axs[1].bar(labels,[b['true_risk'],b['mc_marginal_sum'],b['mc_tangent_sum'],b['gaussian_sum']],color=['#53616e','#539546','#cc6c2c','#087da3'])
    axs[1].axhline(.1,color='black',ls='--');axs[1].set(ylabel='Estimated risk quantity',title='Stronger nonlinear boundary case')
    fig.tight_layout();fig.savefig(out/'calibration.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for kind,c in [('chance','#087da3'),('cvar','#cc6c2c')]:
        rr=[x for x in sweep['rows'] if x['kind']==kind and x['delta']==.1 and x['cost_weight']==0 and x['K']==256 and x['gamma']==30]
        ax.plot([x['planning_cost'] for x in rr],[x['mean_risk'] for x in rr],'o-',label=kind,color=c)
        for x in rr:ax.annotate(str(x['lam']),(x['planning_cost'],x['mean_risk']),fontsize=8,xytext=(4,3),textcoords='offset points')
    ax.set(xlabel='Explicit nominal cost J',ylabel='Independent failure rate',title='Guidance-strength tradeoff on tuning tasks (Δ=.10)');ax.legend();fig.tight_layout();fig.savefig(out/'safety_cost.png',dpi=170);plt.close(fig)
    # Timing includes independent validation; audit executions are research cost.
    model,mean,scale=load(str(ROOT/'nominal_seed0/nominal.pt'));s,g,_=dataset(32,2901);p=Problem(process_sigma=.04,goal_half=.45)
    timing=[]
    for m,kind,projection in [('B0',None,False),('B2','chance',False),('B3','cvar',False),('B4',None,True),('B5','chance',True)]:
        values=[]
        for rep in range(3):
            u,meta=sample(model,mean,scale,s,g,p,kind,.1,8,seed=40101,projection=projection)
            started=time.perf_counter();evaluate(u.cpu().numpy(),s,g,p,4096,75100,.1);vtime=time.perf_counter()-started
            values.append([meta['seconds_batch'],vtime])
        avg=np.mean(values,axis=0);timing.append(dict(method=m,sampling_seconds_batch=float(avg[0]),validation_seconds_batch=float(avg[1]),batch=32))
    (out/'timing.json').write_text(json.dumps(timing,indent=2))
    fig,ax=plt.subplots(figsize=(7,4));xs=np.arange(len(timing));a=np.array([x['sampling_seconds_batch'] for x in timing]);b=np.array([x['validation_seconds_batch'] for x in timing])
    ax.bar(xs,a,label='sample / guidance / projection');ax.bar(xs,b,bottom=a,label='4,096-draw validation')
    ax.set_xticks(xs);ax.set_xticklabels([x['method'] for x in timing]);ax.set(ylabel='Seconds per 32-plan batch',title='Planning time including independent validation');ax.legend();fig.tight_layout();fig.savefig(out/'runtime.png',dpi=170);plt.close(fig)
    # One explicitly selected high-risk nominal case from each environment.
    saved=np.load(ROOT/'projection/plans.npz');fig,axs=plt.subplots(3,4,figsize=(14,10))
    for row,pdict in enumerate(proj['problems']):
        p=Problem(**pdict);env=p.name
        candidates=[x for x in proj['rows'] if x['problem']==env and x['delta']==.1 and x['method']=='B0']
        task=max(candidates,key=lambda x:x['audit_risk'])['task']
        for col,m in enumerate(['B0','B2','B3','B4']):
            ax=axs[row,col];u=torch.tensor(saved[f'{env}_{m}_10'][task:task+1]);s=torch.tensor(saved['start'][task:task+1]);g=saved['goal'][task]
            rng=np.random.default_rng(28800+row);noise=torch.tensor(rng.normal(size=(30,1,21,2)),dtype=u.dtype)
            nominal=simulate(u,s,p)[0,0].numpy();paths=simulate(u,s,p,noise)[:,0].numpy()
            for path in paths:ax.plot(path[:,0],path[:,1],color=colors[m],lw=.6,alpha=.13)
            for cx,cy,r in p.obstacles:ax.add_patch(Circle((cx,cy),r,color='#45484a'))
            ax.add_patch(Rectangle(g-p.goal_half,2*p.goal_half,2*p.goal_half,fc='#d8e8cd',ec='#588a39',alpha=.6))
            ax.plot(nominal[:,0],nominal[:,1],'-o',color=colors[m],ms=2,lw=2)
            rec=next(x for x in proj['rows'] if x['problem']==env and x['delta']==.1 and x['method']==m and x['task']==task)
            ax.set_title(f"{m}: audit risk {100*rec['audit_risk']:.1f}%\n{'accepted' if rec['accepted'] else 'rejected'}",fontsize=10)
            ax.set_aspect('equal');ax.set_xlim(-1.8,2.);ax.set_ylim(-1.5,1.6);ax.grid(alpha=.12)
            if col==0:ax.set_ylabel(env)
    fig.suptitle('Repaired prior and original guidance ablations · Δ=.10\nLargest nominal-risk case selected in each environment; rejected attempts remain visible',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'trajectories.png',dpi=160);plt.close(fig)
    print('reports and figures saved',out)
if __name__=='__main__':main()
