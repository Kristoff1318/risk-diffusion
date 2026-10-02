"""Frozen deterministic repair of existing MPD seeds; no RNG/inference/IK/risk."""
import pathlib,sys,time,json,hashlib,csv,math,resource
import numpy as np,torch
from scipy.spatial.transform import Rotation
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent/'phase7'));import core
T=time.perf_counter();cfg=json.loads((P/'PROTOCOL.json').read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(P/'repair.py')==cfg['repair_source_sha256']
for name,h in cfg['phase7_source_hashes'].items():assert sha(P.parent/'phase7'/name)==h
for name,h in json.loads((P/'PHASE7_MANIFEST.json').read_text()).items():assert sha(P.parent/'phase7'/name)==h
OUT=P/'outcomes';OUT.mkdir(exist_ok=False);saved=np.load(P.parent/'phase7/outcomes/PLANS.npz');tasks=np.load(P.parent/'phase7/TASKS.npz');old=json.loads((P.parent/'phase7/MAIN_RESULTS.json').read_text());oldnom=json.loads((P.parent/'phase7/NOMINAL_DIAGNOSTICS.json').read_text());c=core.Core();torch.cuda.synchronize();initialization=time.perf_counter()-T
def cpu_validate(q,target):
 beg=time.perf_counter();vel=np.diff(q,axis=0)/.05;dense=core.p0.dense(q[None],vel[None],10)[0];m,pose=c.arm.margins(dense);R,pos=pose;pe=float(np.linalg.norm(pos[-1]-target[:,3]));angle=float(Rotation.from_matrix(R[-1].T@target[:,:3]).magnitude());margin=m.min(0);over=float((np.abs(vel)-c.arm.vmax).max());flags=dict(zip(core.p0.EVENTS,[bool(v) for v in margin<0]+[over>0,over>0,pe>.01,angle>math.radians(2)]));return {'feasible':bool(np.isfinite(q).all() and np.isfinite(m).all() and not any(flags.values())),'flags':flags,'minimum_margins':margin.tolist(),'position_m':pe,'orientation_rad':angle,'velocity_limit_excess_rad_s':over,'dense_states':len(dense),'seconds':time.perf_counter()-beg}
rows=[];paths={};incomplete=False
for task in range(8):
 st=c.cast(tasks['q_start'][task]);target_np=tasks['ee_goal_pose'][task];target=c.cast(target_np)
 historical=next(r for r in old['rows'] if r['task']==task and r['arm']=='classical' and r['delta']==.05)
 for slot in range(2):
  if time.perf_counter()-T>cfg['process_guard_seconds']:incomplete=True;break
  key=f't{task}_mpd_d0.05_c{slot}';x=c.cast(saved[key+'_x']);before=c.nom(x,st);assert np.max(np.abs(before.cpu().numpy()-saved[key]))<1e-12
  raw=cpu_validate(saved[key],target_np);beg=time.perf_counter();new,trace=core.refine(c,x,st,target,steps=80,lr=.025,wallcap=15);torch.cuda.synchronize();elapsed=time.perf_counter()-beg;q=c.nom(new,st);qnp=q.detach().cpu().numpy();after=cpu_validate(qnp,target_np);gpu=core.nominal_check(c,q,target)
  assert bool(gpu['union']==0)==after['feasible'];assert np.max(np.abs(np.array(gpu['minimum_margins'])-np.array(after['minimum_margins'])))<1e-10
  rawstored=oldnom[key];assert raw['feasible']==(rawstored['union']==0)
  clas=oldnom[f't{task}_classical_d0.05_c{slot}'];row={'task':task,'official_ID':int(tasks['official_ids'][task]),'slot':slot,'raw':raw,'repaired':after,'raw_cost':float(c.cost(before)),'repaired_cost':float(c.cost(q)),'refinement_seconds':elapsed,'refinement_trace':trace,'independent_gpu_nominal_check':gpu,'saved_classical_nominal':clas,'saved_classical_proposal_total_seconds':historical['proposal_correction_seconds'],'saved_classical_slot_refinement_seconds':historical['generation']['nominal_refinements'][slot]['seconds'],'seed_was_not_replaced':True,'IK_calls':0,'projection_calls':0}
  rows.append(row);paths[key+'_raw_q']=saved[key];paths[key+'_raw_x']=saved[key+'_x'];paths[key+'_repaired_q']=qnp;paths[key+'_repaired_x']=new.cpu().numpy();np.savez_compressed(OUT/'PATHS.npz',**paths);(OUT/f't{task}_c{slot}.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps({'task':task,'slot':slot,'feasible':after['feasible'],'flags':after['flags'],'seconds':elapsed,'process_seconds':time.perf_counter()-T}),flush=True)
 if incomplete:break
report={'complete':not incomplete and len(rows)==16,'scope':'deterministic nominal health only; no inference, IK, projection, physical noise or CVaR','rows':rows,'raw_feasible':sum(r['raw']['feasible'] for r in rows),'repaired_feasible':sum(r['repaired']['feasible'] for r in rows),'repaired_task_coverage':sum(any(r['task']==t and r['repaired']['feasible'] for r in rows) for t in range(8)),'initialization_seconds':initialization,'process_seconds':time.perf_counter()-T,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(P/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
flat=[]
for r in rows:
 flat.append({'task':r['task'],'official_ID':r['official_ID'],'slot':r['slot'],'raw_feasible':r['raw']['feasible'],'repaired_feasible':r['repaired']['feasible'],'repaired_failures':';'.join(k for k,v in r['repaired']['flags'].items() if v),'raw_cost':r['raw_cost'],'repaired_cost':r['repaired_cost'],'position_mm':1000*r['repaired']['position_m'],'orientation_deg':180/math.pi*r['repaired']['orientation_rad'],'world_mpd_mm':1000*r['repaired']['minimum_margins'][0],'world_curobo_mm':1000*r['repaired']['minimum_margins'][1],'self_mpd_mm':1000*r['repaired']['minimum_margins'][2],'self_curobo_mm':1000*r['repaired']['minimum_margins'][3],'joint_margin_rad':r['repaired']['minimum_margins'][4],'velocity_limit_excess':r['repaired']['velocity_limit_excess_rad_s'],'refinement_seconds':r['refinement_seconds'],'iterations':r['refinement_trace']['iterations'],'saved_classical_feasible':r['saved_classical_nominal']['union']==0,'saved_classical_cost':r['saved_classical_nominal']['cost'],'saved_classical_slot_refinement_seconds':r['saved_classical_slot_refinement_seconds'],'saved_classical_two_seed_IK_and_nominal_seconds':r['saved_classical_proposal_total_seconds']})
with (P/'PER_PROPOSAL.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
for name,h in json.loads((P/'PHASE7_MANIFEST.json').read_text()).items():assert sha(P.parent/'phase7'/name)==h
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
