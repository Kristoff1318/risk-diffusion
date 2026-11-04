"""Fresh nominal-only cohort; never evaluates stochastic paths."""
import sys,pathlib,json,time,types,torch,numpy as np
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));from core import Core,p0
T=time.perf_counter();core=Core();base=p0.BASE;idx=json.load(open(base/'assets/archive-extraction.json'));mb=base/'assets'/idx['target_model_prefix'];db=base/'assets'/idx['target_dataset_prefix'];pm=types.SimpleNamespace(Unpickler=p0.RestrictedCPU,__name__='restricted')
indices=torch.load(mb/'val_subset_indices.pt',map_location='cpu',pickle_module=pm);window=[int(x) for x in indices[128:384]]
with open(db/'dataset_merged_doubled_reload-ntasks_-1--bspline-degree_5-n_pts_29-zero_vel_True-zero_acc_True.pickle','rb') as f:fields=p0.RestrictedCPU(f).load()['fields']
keys=['q_start','q_goal','qs','ee_goal_pose','ee_goal_orientation','ee_goal_position','control_points'];data={k:fields[k][window].clone().cpu() for k in keys};del fields
excluded=set(json.load(open(base/'runs/valid-query-selection/selection.json'))['source_indices']);excluded.update(json.load(open(P.parent/'phase5/PROTOCOL.json'))['official_indices']);rows=[];valid=[]
for i,official in enumerate(window):
 start=core.cast(data['q_start'][i]);target=core.cast(data['ee_goal_pose'][i]);x=core.cast(data['control_points'][i]);nom=core.nom(x,start)
 r=__import__('core').nominal_check(core,nom,target);endpoint=core.margins_tensor(core.cast(torch.stack((data['q_start'][i],data['q_goal'][i])).numpy())).detach().cpu().numpy();clear=min(r['minimum_margins'][:4]);reasons=[]
 if official in excluded:reasons.append('previous_source_ID')
 if r['union']:reasons.append('reference_not_nominally_valid')
 if endpoint[:,:4].min()<.001 or endpoint[:,4].min()<0:reasons.append('invalid_or_under1mm_endpoint')
 if float(torch.linalg.vector_norm(data['q_goal'][i]-data['q_start'][i]))<.3:reasons.append('joint_displacement_under.3')
 row={'window_position':i,'official_ID':official,'reference_clearance_m':float(clear),'nominal_checks':r,'endpoint_minimum_margin':float(endpoint.min()),'exclusions':reasons,'eligible':not reasons};rows.append(row)
 if not reasons:valid.append(i)
ranked=sorted(valid,key=lambda i:(rows[i]['reference_clearance_m'],i));fractions=[0,.02,.1,.25,.4,.55,.75,.95];chosen=[]
for f in fractions:
 if len(ranked)<8:break
 rank=int(np.floor((len(ranked)-1)*f))
 while ranked[rank] in chosen:rank=(rank+1)%len(ranked)
 chosen.append(ranked[rank])
assert len(chosen)==8,'Frozen source window contains fewer than8 eligible tasks; no automatic cohort expansion'
selected={k:v[chosen] for k,v in data.items()};torch.save(selected,P/'TASKS.pt');np.savez_compressed(P/'TASKS.npz',**{k:v.numpy() for k,v in selected.items()},official_ids=np.array([window[i] for i in chosen]),window_positions=np.array(chosen))
result={'rule':'fixed official val positions128:384; eligible nominal full-arm/start/referencepose/velocity; choose clearance ranks0,.02,.1,.25,.4,.55,.75,.95 with upward unique tie rule; no stochastic filtering','candidate_count':len(window),'eligible_count':len(valid),'selected_count':len(chosen),'rank_fractions':fractions,'source_order':window,'selected_source_ID_order':[window[i] for i in chosen],'selected_window_positions':chosen,'all_rows':rows,'selected_clearances_m':[rows[i]['reference_clearance_m'] for i in chosen],'selected_references_used_only_as_task_descriptors':True,'seconds':time.perf_counter()-T,'GPU_peak_allocated':torch.cuda.max_memory_allocated()};(P/'TASK_SELECTION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['all_rows','source_order']}),flush=True)
