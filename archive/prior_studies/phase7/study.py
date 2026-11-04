"""One frozen cohort/fourarms/tworiskbudgets; terminal outcomes only."""
import pathlib,sys,time,json,hashlib,resource,traceback,os,numpy as np,torch
from scipy.stats import beta
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import core
T=time.perf_counter();OUT=P/'outcomes';OUT.mkdir(exist_ok=False)
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def upper(k,n,alpha=.025):return float(beta.ppf(1-alpha,k+1,n-k)) if k<n else 1.
def statflags(flags):
 k=int(flags.any(1).sum());return {'n':len(flags),'union':k,'risk':k/len(flags),'CI95':core.p0.interval(k,len(flags)),'counts':dict(zip(core.p0.EVENTS,flags.sum(0).astype(int).tolist()))}
def packed(path,flags):np.savez_compressed(path,flags=np.packbits(flags,axis=0),n=len(flags))
protocol=json.load(open(P/'PROTOCOL.json'));assert json.load(open(P/'QUALIFICATION.json'))['pass'] and json.load(open(P/'BASELINE_QUALIFICATION.json'))['pass']
for name,h in protocol['source_hashes'].items():assert core.p0.sha(P/name)==h,name
before=time.perf_counter();c=core.Core();torch.cuda.synchronize();core_init=time.perf_counter()-before
before=time.perf_counter();model,norm,context,meta=core.load_mpd();model_init=time.perf_counter()-before
queries={k:v.to('cuda') for k,v in torch.load(P/'TASKS.pt',map_location='cpu',weights_only=True).items()};audit_cache={};rows=[];taskprep=[];allplans={};resource_abort=False
for task in range(8):
 if time.perf_counter()-T>protocol['main_process_guard_seconds']:resource_abort=True;break
 st=c.cast(queries['q_start'][task].cpu().numpy());target=c.cast(queries['ee_goal_pose'][task].cpu().numpy());before=time.perf_counter();optimerr=tuple(c.cast(a) for a in core.errors(protocol['risk_samples'],2026104100+task,2));valerr=core.errors(protocol['admission_samples'],2026104200+task,2);auditerr=core.errors(protocol['selected_audit_samples'],2026104300+task,2);taskprep.append({'task':task,'seconds':time.perf_counter()-before,'optimizer_samples':protocol['risk_samples'],'admission_samples':protocol['admission_samples'],'audit_stream_samples':protocol['selected_audit_samples']})
 for delta in [.05,.1]:
  order=core.ARMS[task%4:]+core.ARMS[:task%4]
  for arm in order:
   key=f't{task}_{arm}_d{delta}';beg=time.perf_counter();result={'task':task,'official_ID':protocol['selected_ID_order'][task],'arm':arm,'delta':delta,'regime':'double','key':key,'status':'started','accepted':False,'selected':None,'candidates':[]}
   try:
    if arm.startswith('classical'):x,generation=core.classical(c,st,target,2026104000+task*100)
    else:x,generation=core.mpd(model,norm,context,queries,task,2026104000+task*100)
    result['generation']=generation;result['risk_refinements']=[];proposal_wall=time.perf_counter()-beg
    if arm.endswith('cvar'):
     corrected=[]
     for slot in range(2):
      remaining=max(.001,protocol['proposal_correction_wallcap_seconds']-(time.perf_counter()-beg));cap=remaining/(2-slot);xx,log=core.refine(c,x[slot],st,target,steps=protocol['risk_iterations'],lr=protocol['risk_lr'],delta=delta,err=optimerr,wallcap=cap);corrected.append(xx);result['risk_refinements'].append(log)
     x=torch.stack(corrected)
    result['proposal_correction_seconds']=time.perf_counter()-beg;admissionbeg=time.perf_counter();admitted=[];paths=[]
    for slot in range(2):
     nom=c.nom(x[slot],st);paths.append(nom);vc,flags=core.audit(c,nom,target,valerr,16);vc['upper95_selection_corrected']=upper(vc['union'],vc['n']);ff=nom.diff(dim=0)/.05;vc['nominal_command_limit_valid']=bool((ff.abs()<=c.vmax).all());vc['cost']=float(c.cost(nom));vc['admitted']=bool(vc['finite'] and vc['nominal_command_limit_valid'] and vc['upper95_selection_corrected']<=delta);vc['slot']=slot;vc['path_hash']=hashlib.sha256(nom.detach().cpu().numpy().tobytes()).hexdigest();result['candidates'].append({'admission':vc});packed(OUT/f'{key}_c{slot}_admission.npz',flags)
     if vc['admitted']:admitted.append(slot)
    result['admission_seconds']=time.perf_counter()-admissionbeg;result['selected']=min(admitted,key=lambda slot:result['candidates'][slot]['admission']['cost']) if admitted else None;result['accepted']=bool(admitted);result['request_seconds']=time.perf_counter()-beg;result['shared_initialization_charge_seconds']=core_init/64+(model_init/32 if arm.startswith('mpd') else 0);result['latency_with_amortized_init']=result['request_seconds']+result['shared_initialization_charge_seconds'];result['cold_equivalent_request_seconds']=result['request_seconds']+core_init+(model_init if arm.startswith('mpd') else 0)
    auditbeg=time.perf_counter();cache_hits=0
    for slot,nom in enumerate(paths):
     n=protocol['selected_audit_samples'] if slot==result['selected'] else protocol['all_proposal_audit_samples'];cachekey=(task,result['candidates'][slot]['admission']['path_hash'],n)
     if cachekey in audit_cache:ac,af=audit_cache[cachekey];ac=dict(ac);ac['cache_reused']=True;cache_hits+=1
     else:
      er=tuple(v[:n] for v in auditerr);ac,af=core.audit(c,nom,target,er,16);ac['cache_reused']=False;audit_cache[cachekey]=(dict(ac),af.copy())
     ac['cost']=float(c.cost(nom));ac['nominal_control_count']=201;result['candidates'][slot]['audit']=ac;packed(OUT/f'{key}_c{slot}_audit.npz',af)
     if slot==result['selected']:
      ac['point_above_budget']=ac['risk']>delta;ac['lower_CI_above_budget']=ac['CI95'][0]>delta
     allplans[key+f'_c{slot}']=nom.detach().cpu().numpy();allplans[key+f'_c{slot}_x']=x[slot].detach().cpu().numpy()
    result['audit_seconds']=time.perf_counter()-auditbeg;result['audit_cache_hits']=cache_hits;result['status']='completed';result['timeout']=any(v.get('timeout',False) for v in result['risk_refinements']);result['end_to_end_with_research_audit_seconds']=time.perf_counter()-beg
   except Exception as e:
    result.update(status='exception',exception=repr(e),traceback=traceback.format_exc(),request_seconds=time.perf_counter()-beg)
   rows.append(result);dump(OUT/(key+'.json'),result);np.savez_compressed(OUT/'PLANS.npz',**allplans);dump(OUT/'INDEX.json',{'completed_requests':len(rows),'accepted':sum(r['accepted'] for r in rows),'process_seconds':time.perf_counter()-T,'rows':[r['key'] for r in rows]});print(json.dumps({'task':task,'arm':arm,'delta':delta,'status':result['status'],'accepted':result['accepted'],'slot':result['selected'],'request_seconds':result.get('request_seconds'),'audit_seconds':result.get('audit_seconds'),'process_seconds':time.perf_counter()-T}),flush=True)
   if time.perf_counter()-T>protocol['main_process_guard_seconds']:resource_abort=True;break
  if resource_abort:break
 if resource_abort:break
 dump(OUT/'TASK_PREPARATION.json',taskprep)
# Retain explicitly incomplete requests if guard prevents completion.
if resource_abort:
 present={(r['task'],r['arm'],r['delta']) for r in rows}
 for task in range(8):
  for delta in [.05,.1]:
   for arm in core.ARMS:
    if (task,arm,delta) not in present:rows.append({'task':task,'official_ID':protocol['selected_ID_order'][task],'arm':arm,'delta':delta,'status':'resource_not_run','accepted':False,'candidates':[]})
report={'terminal':True,'complete':not resource_abort and len(rows)==64,'rows':rows,'process_seconds':time.perf_counter()-T,'core_initialization_seconds':core_init,'MPD_model_initialization_seconds':model_init,'task_preparation':taskprep,'GPU_peak_allocated':torch.cuda.max_memory_allocated(),'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'CPU_process_times':list(os.times()),'audit_cache_exact_hash_entries':len(audit_cache)};dump(P/'MAIN_RESULTS.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['rows','task_preparation']}),flush=True)
