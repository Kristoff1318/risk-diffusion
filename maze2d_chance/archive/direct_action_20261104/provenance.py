from baseline import *
import platform,csv
out=ROOT/'results';rows=json.load(open(out/'results.json'))
for row in rows:
 n=row['executions'];row['outcome_exact_ci95']={}
 for name,k in row['counts'].items():
  row['outcome_exact_ci95'][name]=[float(beta_distribution.ppf(.025,k,n-k+1)) if k else 0.,float(beta_distribution.ppf(.975,k+1,n-k)) if k<n else 1.]
json.dump(rows,open(out/'results.json','w'),indent=2)
with open(out/'calibration.csv','w') as f:
 w=csv.writer(f);w.writerow(['method','budget','task','seed','union_surrogate','capped_surrogate','observed_failure','exact_ci95_lower','exact_ci95_upper','surrogate_accepted','audit_accepted','false_safe_confirmed'])
 for r in rows:w.writerow([r['method'],r['budget'],r['task'],r['seed'],r['predicted_risk'],min(1.,r['predicted_risk']),r['failure_rate'],*r['failure_exact_ci95'],r['surrogate_accepted'],r['audit_accepted'],r['false_safe_confirmed']])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
provenance=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,gym=gym.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),native_environment='maze2d-large-v1',checkpoint_sha256=sha(CHECKPOINT/f'state_{EPOCH}.pt'),normalization_sha256=sha(ROOT/'normalization.npz'),slides_sha256=sha('/home/kristoff/Documents/risk_diffusion/main.pptx'),source_sha256=sha(ROOT/'baseline.py'))
json.dump(provenance,open(out/'provenance.json','w'),indent=2)
print(provenance)
