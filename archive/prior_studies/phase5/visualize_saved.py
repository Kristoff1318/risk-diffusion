"""Saved evidence visualization only; no new noise, planner, or FK qualification."""
import pathlib,sys,json,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parent));import panda_p0 as p
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
z=np.load(p.P/'FROZEN_PATHS.npz');arm=p.Arm();demo=np.load(p.P/'DEMO_n64_sub10.npz');fig=plt.figure(figsize=(12,8));ax=fig.add_subplot(221,projection='3d')
for i in range(4):
 q=z['q'][i];R,ee=arm.fk(q)['panda_hand'];ax.plot(*ee.T,label=f'Fixed path{i}')
 for step in [0,100,200]:
  poses=arm.fk(q[step:step+1]);chain=np.array([poses[f'panda_link{k}'][1][0] for k in range(8)]+[poses['panda_hand'][1][0]]);ax.plot(*chain.T,alpha=.35,lw=1)
 ax.scatter(*z['target'][i,:,3],s=25)
ax.scatter(*arm.centers.T,s=60,c='gray',alpha=.5,label='Obstacle centers (r=.15m)');ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='Nominal whole-arm motion; selected joint frames');ax.legend(fontsize=7)
ax=fig.add_subplot(222);t=np.arange(201)*p.DT
for regime,color in [('base','tab:blue'),('double','tab:orange')]:
 for q in demo[f'p0_{regime}_q']:ax.plot(t,(q[:,0]-z['q'][0,:,0])*1000,c=color,alpha=.5)
ax.set(xlabel='Time (s)',ylabel='Joint1 execution error (mrad)',title='Four saved executions per regime, path0')
ax=fig.add_subplot(223);v=json.load(open(p.P/'RUN_n256_sub10.json'));x=np.arange(4)
for regime,offset,color in [('base',-.15,'tab:blue'),('double',.15,'tab:orange')]:
 rows=[r for r in v['rows'] if r['regime']==regime];y=np.array([r['union_probability'] for r in rows]);ci=np.array([r['union_CI95'] for r in rows]);ax.errorbar(x+offset,y,yerr=np.stack((y-ci[:,0],ci[:,1]-y)),fmt='o',capsize=3,label=regime,color=color)
ax.set(xticks=x,xlabel='Fixed reference path',ylabel='Union failure probability',ylim=(0,1),title='256 executions; exact two-sided95% intervals');ax.legend()
ax=fig.add_subplot(224);ax.axis('off');ax.text(0,.95,'KINEMATIC TRACKING-ERROR STUDY\n\n10s; 200 velocity commands at50ms\nGaussian initial + persistent bias + step noise\nFull-arm world/self, joints, velocities, pose\n5ms primary geometry grid\n\nSynthetic scales: no hardware calibration\nConfigured spheres: no mesh coverage proof\nDiscrete grids: no continuous safety proof\nFixed paths: no planner advantage claim',va='top',fontsize=11)
fig.tight_layout();fig.savefig(p.P/'DEMO.png',dpi=130);plt.close(fig)
