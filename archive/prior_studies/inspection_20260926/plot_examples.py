from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
from risk_diffusion.evaluator import evaluate

out=Path(__file__).parent
plans=np.load(out.parent/'outputs/final/plans.npz')
results=json.loads((out.parent/'outputs/final/results.json').read_text())
rng=np.random.default_rng(123)
s=np.column_stack((-1.5*np.ones(12),rng.uniform(-.18,.18,12)))
g=np.column_stack((1.5*np.ones(12),rng.uniform(-.18,.18,12)))
cases=[(0,1,'Already safe'),(0,3,'Guidance repairs collision'),(2,1,'Chance guidance still fails')]
methods=[('B0','Nominal'),('B2','Chance guidance'),('B3','CVaR guidance'),('B4','Deterministic refinement')]
def paths_for(u,start,n,seed,noise):
 rng=np.random.default_rng(seed);x=np.tile(start,(n,1));states=[x.copy()]
 for uk in u:
  d=np.column_stack((.12*np.sin(2*x[:,1]),.12*np.sin(2*x[:,0])))
  x=x+.2*(uk+d)+noise*rng.standard_normal(x.shape);states.append(x.copy())
 return np.stack(states,1)
fig,axes=plt.subplots(3,4,figsize=(16,10),sharex=True,sharey=True)
records=[];bundle={}
for row,(seed,task,title) in enumerate(cases):
 for col,(m,name) in enumerate(methods):
  ax=axes[row,col];u=plans[f'{m}_10_{seed}'][task]
  nom=paths_for(u,s[task],1,0,0)[0];cloud=paths_for(u,s[task],40,20260926+row,.025)
  ev=evaluate(u[None],s[task][None],g[task][None],draws=8192,seed=8200+row,delta=.1)[0]
  saved=next(x for x in results['rows'] if x['method']==m and x['delta']==.1 and x['sample_seed']==seed and x['task']==task)
  records.append(dict(case=title,task=task,sample_seed=seed,method=m,**ev,predicted_risk=saved['predicted_risk']))
  bundle[f'{row}_{m}_controls']=u;bundle[f'{row}_{m}_nominal']=nom;bundle[f'{row}_{m}_executions']=cloud
  for path in cloud: ax.plot(path[:,0],path[:,1],color='#92b5ce',lw=.65,alpha=.32)
  ax.add_patch(Circle((0,0),.43,facecolor='#42484c',edgecolor='black',zorder=4))
  ax.add_patch(Rectangle(g[task]-.3,.6,.6,facecolor='#e7f3df',edgecolor='#4d7637',alpha=.8,zorder=0))
  ax.plot(nom[:,0],nom[:,1],'-o',color='#075e94',lw=2,ms=2.4,zorder=5)
  ax.scatter(*s[task],s=35,color='#075e94',zorder=6);ax.scatter(*g[task],s=65,marker='*',color='#4d7637',zorder=6)
  risk=ev['risk'];lo,hi=1-ev['joint_ci'][1],1-ev['joint_ci'][0]
  ax.set_title(f'{name}\nFailure {100*risk:.1f}% (95% CI {100*lo:.1f}–{100*hi:.1f}%)',fontsize=10)
  if col==0:ax.set_ylabel(title+'\ny position',fontsize=10)
  if row==2:ax.set_xlabel('x position')
  ax.set_aspect('equal');ax.set_xlim(-1.8,2.0);ax.set_ylim(-1.5,2.0);ax.grid(alpha=.16)
fig.suptitle('Same task and initial diffusion noise within each row · requested failure bound 10%',fontsize=15)
fig.text(.5,.025,'Dark blue: nominal rollout and discrete states  |  Pale blue: 40 fresh stochastic executions  |  Green: terminal goal region\nExamples deliberately show an easy case, a repair, and a remaining failure; rates use 8,192 fresh executions per panel.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.065,1,.95]);fig.savefig(out/'trajectory_examples.png',dpi=170);fig.savefig(out/'trajectory_examples.pdf');plt.close(fig)
(out/'example_metrics.json').write_text(json.dumps(records,indent=2));np.savez_compressed(out/'example_trajectories.npz',**bundle)
for x in records:print(x['case'],x['method'],round(x['risk'],4),'collision',round(x['collision'],4),'goal fail',round(1-x['goal'],4),'surrogate',round(x['predicted_risk'],4))
