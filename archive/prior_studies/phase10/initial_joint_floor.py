"""Report only the immutable t=0 joint-event probability under independent Gaussians.
Usage: python3 initial_joint_floor.py INPUT.json > REPORT.json
Input keys: tasks [{official_ID,q_start}], joint_lower, joint_upper,
initial_error_std_rad (scalar or seven values), risk_budgets, optional
initial_error_mean_rad (scalar or seven values). No motion planning or sampling.
"""
import json,math,sys
def expand(value):return [float(value)]*7 if isinstance(value,(int,float)) else [float(x) for x in value]
def diagnose(spec):
 lo=expand(spec['joint_lower']);hi=expand(spec['joint_upper']);sd=expand(spec['initial_error_std_rad']);bias=expand(spec.get('initial_error_mean_rad',0));budgets=spec['risk_budgets']
 assert len(lo)==len(hi)==len(sd)==len(bias)==7
 assert all(math.isfinite(x) for x in lo+hi+sd+bias) and all(a<b for a,b in zip(lo,hi)) and all(x>=0 for x in sd)
 assert all(0<d<1 for d in budgets)
 rows=[]
 for task in spec['tasks']:
  q=expand(task['q_start']);assert len(q)==7 and all(math.isfinite(v) for v in q);per=[];log_inside=0.;certain=False
  for j in range(7):
   mu=q[j]+bias[j]
   if sd[j]==0:p=float(mu<lo[j] or mu>hi[j])
   else:p=.5*math.erfc((mu-lo[j])/sd[j]/math.sqrt(2))+.5*math.erfc((hi[j]-mu)/sd[j]/math.sqrt(2))
   p=min(1.,max(0.,p));per.append({'joint':j+1,'nominal_mean_rad':mu,'lower_margin_rad':mu-lo[j],'upper_margin_rad':hi[j]-mu,'std_rad':sd[j],'failure_probability':p})
   if p==1:certain=True
   else:log_inside+=math.log1p(-p)
  floor=1. if certain else -math.expm1(log_inside)
  rows.append({'official_ID':task['official_ID'],'initial_joint_event_probability':floor,'per_joint':per,'budget_diagnostics':[{'delta':d,'ruled_out_by_initial_joint_event':floor>d,'status':'target for this fixed initial-distribution full-trajectory event is ruled out' if floor>d else 'not ruled out by this initial joint event; overall achievable risk remains unknown'} for d in budgets]})
 return {'event':'any initial physical joint outside its limits at t=0, before any command','assumptions':['fixed initial mean/start','independent Gaussian joint errors','given uncalibrated distribution parameters treated as exact','initial joint violation is included in unchanged trajectory-union failure','no safe-start conditioning, abort or repositioning changes this event'],'probability_type':'analytic conditional-model probability; no sampling confidence interval','numeric_note':'stable Gaussian tails and log-product; finite precision may underflow extremely tiny tails, which must not be described as exact zero risk','scope':'lower bound on current full-trajectory union risk, not a global statement about task reachability, alternate starts, other policies or hardware','rows':rows}
if __name__=='__main__':
 print(json.dumps(diagnose(json.load(open(sys.argv[1]))),indent=2,allow_nan=False))

