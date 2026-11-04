"""Check recorded source snapshots, all native event counts, and admissions."""
import json
from pathlib import Path
import numpy as np
from runtime import PROJECT,sha
from evaluate_tracking import summarize


def main():
    out=PROJECT/'results/chance_comparison'
    protocol=json.loads((out/'protocol.json').read_text())
    snapshots=json.loads((out/'source_snapshot.json').read_text())
    for path,expected in protocol['source_sha256'].items():
        source=out/snapshots[path] if path in snapshots else PROJECT/path
        assert sha(source)==expected,path
    rows=json.loads((out/'results.json').read_text())
    checked=0
    for row in rows:
        owner=row['exact_path_audit_reuse'] or row['name']
        flags=np.load(out/f'{owner}_executions.npz')
        for stream in ['validation','audit']:
            observed=summarize(flags[stream+'_flags'])
            for key in ['draws','failures','counts','CI95','upper95','failure_rate']:
                assert observed[key]==row[stream][key],(row['name'],stream,key)
            checked+=1
        reference=np.load(out/f'{row["name"]}_paths.npz')['reference']
        import hashlib
        assert hashlib.sha256(reference.tobytes()).hexdigest()==row['reference_sha256']
        if row['budget'] is not None:
            assert row['accepted']==(row['validation']['upper95']<=row['budget'])
            assert row['surrogate_pass']==(row['surrogate_risk']<=row['budget'])
    status=json.loads((out/'status.json').read_text())
    assert status['complete'] and len(rows)==18 and status['native_stochastic_executions']==9216
    report=dict(passed=True,source_snapshots_verified=True,reference_hashes_verified=True,
                event_streams_recounted=checked,all_confidence_bounds_exact=True,
                all_admissions_recomputed=True,proposal_count=len(rows))
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
