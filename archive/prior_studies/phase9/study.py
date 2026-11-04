"""Frozen diagnostic replay: equal nominal preparation, none/dense/CVaR correction."""
import pathlib,sys,time,json,hashlib,traceback,resource,os
import numpy as np,torch
from scipy.stats import beta
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent/'phase7'));import core
T=time.perf_counter();cfg=json.loads((P/'PROTOCOL.json').read_text())
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def dump(f,v):f.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
assert sha(P/'study.py')==cfg['source_sha256']
for name,h in json.loads((P/'PREVIOUS_MANIFEST.json').read_text()).items():assert sha(P.parent/name)==h
OUT=P/'outcomes';OUT.mkdir(exist_ok=False);c=core.Core();torch.cuda.synchronize();init=time.perf_counter()-T
p7=P.parent/'phase7';p8=P.parent/'phase8';old=json.loads((p7/'MAIN_RESULTS.json').read_text());repair=json.loads((p8/'RESULTS.json').read_text());tasks=np.load(p7/'TASKS.npz');a=np.load(p7/'outcomes/PLANS.npz');b=np.load(p8/'outcomes/PATHS.npz')
assert json.loads((p8/'PROTOCOL.json').read_text())['phase7_source_hashes']==json.loads((p7/'PROTOCOL.json').read_text())['source_hashes']
for r in repair['rows']:assert r['refinement_trace']['iterations']==80 and not r['refinement_trace']['timeout'] and r['IK_calls']==0 and r['projection_calls']==0
arms=['classical_none','classical_dense','classical_cvar','mpd_none','mpd_dense','mpd_cvar'];rows=[];paths={};cache={};prep=[];abort=False
def packed(f,flags):np.savez_compressed(f,flags=np.packbits(flags,axis=0),n=len(flags))
for task in range(8):
 st=c.cast(tasks['q_start'][task]);target=c.cast(tasks['ee_goal_pose'][task]);bases={'classical':np.stack([a[f't{task}_classical_d0.05_c{s}_x'] for s in range(2)]),'mpd':np.stack([b[f't{task}_mpd_d0.05_c{s}_repaired_x'] for s in range(2)])}
 for s in range(2):assert np.array_equal(a[f't{task}_classical_d0.05_c{s}_x'],a[f't{task}_classical_d0.1_c{s}_x'])
 histc=next(r for r in old['rows'] if r['task']==task and r['arm']=='classical' and r['delta']==.05);histm=next(r for r in old['rows'] if r['task']==task and r['arm']=='mpd' and r['delta']==.05)
 before=time.perf_counter();opt=tuple(c.cast(v) for v in core.errors(32,2026105100+task,2));aderr=core.errors(1024,2026105200+task,2);auerr=core.errors(2048,2026105300+task,2);zero=(c.cast(np.zeros((1,2001,7))),c.cast(np.zeros((1,200,7))),c.cast(np.zeros((1,200,7))));prep.append({'task':task,'seconds':time.perf_counter()-before,'historical_classical_generation_IK_nominal_seconds':histc['proposal_correction_seconds'],'historical_classical_nominal_slot_seconds':[v['seconds'] for v in histc['generation']['nominal_refinements']],'historical_MPD_generation_seconds':histm['generation']['seconds'],'historical_MPD_nominal_preparation_seconds':sum(r['refinement_seconds'] for r in repair['rows'] if r['task']==task),'fresh_generation_IK_nominal_seconds':0})
 for delta in [.05,.1]:
  order=arms[task%6:]+arms[:task%6]
  for arm in order:
   if time.perf_counter()-T>cfg['main_guard_seconds']:abort=True;break
   family,variant=arm.split('_');key=f't{task}_{arm}_d{delta}';beg=time.perf_counter();r={'task':task,'official_ID':int(tasks['official_ids'][task]),'arm':arm,'family':family,'variant':variant,'delta':delta,'key':key,'accepted':False,'selected':None,'status':'started','candidates':[],'corrections':[]}
   try:
    x=c.cast(bases[family]);nominal_input=[]
    for s in range(2):nominal_input.append(core.nominal_check(c,c.nom(x[s],st),target))
    r['prepared_nominal_checks']=nominal_input;cbeg=time.perf_counter()
    if variant!='none':
     xx=[]
     for s in range(2):
      value,log=core.refine(c,x[s],st,target,steps=8,lr=.003,delta=1. if variant=='dense' else delta,err=zero if variant=='dense' else opt,wallcap=15)
      xx.append(value);r['corrections'].append(log)
     x=torch.stack(xx)
    r['correction_seconds']=time.perf_counter()-cbeg;paths_current=[];abeg=time.perf_counter();admitted=[]
    for s in range(2):
     q=c.nom(x[s],st);nc=core.nominal_check(c,q,target);ad,flags=core.audit(c,q,target,aderr,16);k=ad['union'];n=ad['n'];ad['upper_selection']=float(beta.ppf(.975,k+1,n-k)) if k<n else 1.;ad['nominal_command_valid']=bool((q.diff(dim=0).abs()/.05<=c.vmax).all());ad['cost']=float(c.cost(q));ad['admitted']=bool(ad['finite'] and ad['nominal_command_valid'] and ad['upper_selection']<=delta);h=hashlib.sha256(q.detach().cpu().numpy().tobytes()).hexdigest();ad['path_hash']=h;r['candidates'].append({'slot':s,'nominal':nc,'admission':ad});packed(OUT/f'{key}_c{s}_admission.npz',flags);paths_current.append(q)
     if ad['admitted']:admitted.append(s)
    r['admission_seconds']=time.perf_counter()-abeg;r['selected']=min(admitted,key=lambda s:r['candidates'][s]['admission']['cost']) if admitted else None;r['accepted']=bool(admitted);r['incremental_request_seconds']=time.perf_counter()-beg;r['historical_preparation']=prep[-1];before=time.perf_counter();hits=0
    for s,q in enumerate(paths_current):
     n=2048 if s==r['selected'] else 1024;ck=(task,r['candidates'][s]['admission']['path_hash'],n)
     if ck in cache:au,flags=cache[ck];au=dict(au);au['cache_reused']=True;hits+=1
     else:
      au,flags=core.audit(c,q,target,tuple(v[:n] for v in auerr),16);au['cache_reused']=False;cache[ck]=(dict(au),flags.copy())
     au['cost']=r['candidates'][s]['admission']['cost'];r['candidates'][s]['audit']=au;packed(OUT/f'{key}_c{s}_audit.npz',flags);paths[key+f'_c{s}_q']=q.detach().cpu().numpy();paths[key+f'_c{s}_x']=x[s].cpu().numpy()
    r['audit_seconds']=time.perf_counter()-before;r['audit_cache_hits']=hits;r['timeout']=any(v['timeout'] for v in r['corrections']);r['status']='completed'
   except Exception as e:r.update(status='exception',error=repr(e),traceback=traceback.format_exc())
   rows.append(r);dump(OUT/(key+'.json'),r);np.savez_compressed(OUT/'PATHS.npz',**paths);dump(OUT/'INDEX.json',{'rows':len(rows),'accepted':sum(v['accepted'] for v in rows),'seconds':time.perf_counter()-T});print(json.dumps({'task':task,'arm':arm,'delta':delta,'status':r['status'],'accepted':r['accepted'],'selected':r['selected'],'seconds':time.perf_counter()-T}),flush=True)
  if abort:break
 if abort:break
if abort:
 present={(r['task'],r['arm'],r['delta']) for r in rows}
 for task in range(8):
  for arm in arms:
   for delta in [.05,.1]:
    if (task,arm,delta) not in present:rows.append({'task':task,'arm':arm,'delta':delta,'status':'resource_not_run','accepted':False,'selected':None,'candidates':[]})
report={'terminal':True,'complete':not abort and len(rows)==96 and all(r['status']=='completed' for r in rows),'rows':rows,'initialization_seconds':init,'process_seconds':time.perf_counter()-T,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'CPU_process_times':list(os.times()),'audit_cache_entries':len(cache),'task_preparation':prep};dump(P/'MAIN_RESULTS.json',report)
for name,h in json.loads((P/'PREVIOUS_MANIFEST.json').read_text()).items():assert sha(P.parent/name)==h
print(json.dumps({k:v for k,v in report.items() if k not in ['rows','task_preparation']}),flush=True)

