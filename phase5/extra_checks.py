import sys,json,time,pathlib,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parent));import panda_p0 as p
T=time.perf_counter();arm=p.Arm();z=np.load(p.P/'FROZEN_PATHS.npz');check={}
# Native hard limits agree with URDF metadata for all seven active joints.
import xml.etree.ElementTree as ET
for tree in arm.urdfs:
 vals=[]
 for i in range(1,8):
  el=next(j for j in ET.parse(tree).getroot().findall('joint') if j.get('name')==f'panda_joint{i}').find('limit');vals.append([float(el.get(k)) for k in ['lower','upper','velocity']])
 assert np.array_equal(np.array(vals),np.stack((arm.lo,arm.hi,arm.vmax),1))
check['seven_joint_limits_match_both_URDFs']=True
# Velocity errors are actually executed, independent of commanded velocity.
u=np.zeros((p.H,7));u[0,0]=arm.vmax[0]-.001;draw=(np.zeros((1,7)),np.zeros((1,7)),np.zeros((1,p.H,7)));draw[2][0,0,0]=.01
q,v=p.execute(np.zeros(7),u,draw);assert not (np.abs(u)>arm.vmax).any() and (np.abs(v)>arm.vmax).any();assert q[0,1,0]==p.DT*v[0,0,0]
check['executed_velocity_violation_unclipped']={'command':u[0,0],'executed':v[0,0,0],'limit':arm.vmax[0],'next_joint':q[0,1,0]}
# Empirical covariance check independent samples, no FK; bias correlation across time retained.
d=p.noise(20000,2026102988);qe,_=p.execute(np.zeros(7),np.zeros((p.H,7)),d)
expected=.002**2+10**2*.001**2+p.H*p.DT**2*.005**2;emp=np.cov(qe[:,-1].T);cross=(qe[:,50]-qe[:,50].mean(0)).T@(qe[:,-1]-qe[:,-1].mean(0))/(len(qe)-1);cross_expected=.002**2+50*200*p.DT**2*.001**2+50*p.DT**2*.005**2
check['empirical_terminal_variance_mean']=float(np.diag(emp).mean());check['exact_terminal_variance']=expected;check['empirical_cross_time_diagonal_mean']=float(np.diag(cross).mean());check['exact_cross_time_covariance']=cross_expected;check['relative_variance_error']=float(abs(np.diag(emp).mean()/expected-1));assert check['relative_variance_error']<.025
# Native historical deterministic margin replay at same2001 grid; sourceaudit used509.
audit=json.load(open(p.BASE/'runs/valid-query-selection/reference-path-audit.json'))['rows'];names=['MPD_world_margin_m','curobo_world_margin_m','MPD_self_margin_m','curobo_self_margin_m','joint_margin_rad'];nom=json.load(open(p.P/'QUALIFICATION.json'))['nominal'];diff=[]
for i in range(4):
 old=next(r for r in audit if r['scene']=='EnvSpheres3D' and r['query_position']==i)['checks'];diff.append({k:nom[i]['min_margins'][p.EVENTS[j]]-old[k] for j,k in enumerate(names)})
check['historical_margin_difference_different_grids']=diff;assert max(abs(v) for d in diff for v in d.values())<.001
check['seconds']=time.perf_counter()-T;check['pass']=True;p.save('EXTRA_CHECKS.json',check);print(json.dumps(check),flush=True)
