"""One two-request physical parity smoke; no new planning or research audit."""
import pathlib,sys,time,json,hashlib,resource
import numpy as np
import torch
from fractions import Fraction
from scipy.stats import beta
P=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(P.parent),str(P.parent/'phase7')]
import core
from validation_adapter import Request
from validation_adapter.physical_bridge import freeze_candidate, PhysicalBackend, validate_current
START=time.perf_counter()
cfg=json.loads((P/'PROTOCOL.json').read_text())
result={'complete':False,'new_planning':0,'new_optimization':0,'new_inference':0,
        'independent_research_audits':0,'saved_seed_equivalence_only':True,'rows':[]}


def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def save():
 result['process_seconds']=time.perf_counter()-START
 result['peak_GPU_allocated_bytes']=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
 result['peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (P/'GPU_SMOKE.json').write_text(json.dumps(result,indent=2)+'\n')


def run():
 torch.set_num_threads(2)
 for f,h in cfg['hashes'].items():assert sha(P.parent/f)==h,f
 c=core.Core();torch.cuda.synchronize()
 tasks=np.load(P.parent/'phase7/TASKS.npz');paths=np.load(P.parent/'phase9/outcomes/PATHS.npz')
 for key in cfg['requests']:
  original=json.loads((P.parent/'phase9/outcomes'/f'{key}.json').read_text())
  t=original['task']; target=tasks['ee_goal_pose'][t];saved=[];candidates=[]
  for s,a in enumerate(original['candidates']):
   q=paths[f'{key}_c{s}_q']
   cost=float(c.cost(c.cast(q)))
   assert abs(cost-a['admission']['cost'])<1e-10
   eligible=bool(a['admission']['finite'] and a['admission']['nominal_command_valid'])
   assert eligible==bool(np.isfinite(q).all() and (np.abs(np.diff(q,axis=0))/.05<=c.arm.vmax).all())
   candidates.append(freeze_candidate(s,q,cost=cost,eligible=eligible,expected_path_hash=a['admission']['path_hash']))
   z=np.load(P.parent/'phase9/outcomes'/f'{key}_c{s}_admission.npz')
   saved.append(np.unpackbits(z['flags'],axis=0)[:int(z['n'])].astype(bool))
  request=Request(key,cfg['model_id'],2026105200+t,Fraction(str(original['delta'])))
  rows=[]
  for mode in ('fixed','lazy'):
   if time.perf_counter()-START>35:raise TimeoutError('35s smoke guard')
   cursors={0:0,1:0}; observed=[]
   def observe(s,flags):
    observed.append((s,cursors[s],flags));cursors[s]+=len(flags)
   torch.cuda.synchronize();begin=time.perf_counter()
   backend=PhysicalBackend(core,c,target,model_id=cfg['model_id'],synchronize=torch.cuda.synchronize,
                           kernel_rollouts=16,partial_feedback=True,observe=observe)
   actual=validate_current(candidates,request,backend,mode=mode,chunk_size=16)
   torch.cuda.synchronize();elapsed=time.perf_counter()-begin
   assert actual['executed_counts_exact']
   assert actual['selected']==original['selected'],(key,mode,actual)
   for s,start,flags in observed:np.testing.assert_array_equal(flags,saved[s][start:start+len(flags)])
   if mode=='fixed':
    assert actual['executed']==2048
    for s,a in enumerate(actual['candidates']):
     assert a['failures']==original['candidates'][s]['admission']['union']
     k=a['failures'];upper=float(beta.ppf(.975,k+1,1024-k)) if k<1024 else 1.
     assert bool(a['status']=='certified')==bool(candidates[s].eligible and upper<=float(request.epsilon))
   if actual['certificate']:assert actual['certificate']['n']==1024
   rows.append({'mode':mode,'end_to_end_validation_seconds':elapsed,'result':actual,
                'regenerated_all_acquired_event_rows_match':True,
                'actually_regenerated_event_rows':sum(len(a[2]) for a in observed)})
  assert rows[0]['result']['physical_backend']['full_error_bank_sha256']==rows[1]['result']['physical_backend']['full_error_bank_sha256']
  result['rows'].append({'request':key,'original_selected':original['selected'],
                       'candidate_costs':[a.cost for a in candidates],'methods':rows})
  save()
 result['complete']=True


try:
 run()
except Exception as error:
 import traceback
 result.update(error=repr(error),traceback=traceback.format_exc())
finally:save()
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
sys.exit(0 if result['complete'] else 1)
