"""Reconstruct completed screen decisions from audited paired episode outcomes."""
import hashlib
import json
import math
from pathlib import Path
from analyze_results import mean_ci

ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'HUDM/reports/research/k192_sepopt_published_parity_20260924'
SEEDS=(0,1,2,42,100)
OUTPUT=ROOT/'screen_gate_validation_receipt.json'
checks={}
for environment in ('tworoom','reacher','pusht','ogb_cube'):
    path=REPORT/f'screen_{environment}_summary.json'
    if not path.is_file():
        continue
    summary=json.loads(path.read_text())
    assert summary['seeds']==list(SEEDS)
    assert summary['episodes_per_seed_per_model']==50
    assert summary['qualification_margin_pp']==-5.0
    published=[];retrained=[];deltas=[];inputs=[]
    for seed in SEEDS:
        auditpath=REPORT/'screen_n50'/environment/f'seed_{seed}'/'paired_audit.json'
        audit=json.loads(auditpath.read_text());episodes=audit['paired_episodes']
        assert audit['verified'] is True and audit['seed']==seed and len(episodes)==50
        assert all(type(e['published_success']) is bool and type(e['retrained_success']) is bool for e in episodes)
        p=100*sum(e['published_success'] for e in episodes)/len(episodes)
        r=100*sum(e['retrained_success'] for e in episodes)/len(episodes)
        published.append(p);retrained.append(r);deltas.append(r-p)
        inputs.append({'seed':seed,'paired_audit_sha256':hashlib.sha256(auditpath.read_bytes()).hexdigest()})
    for name,values in [('published',published),('retrained',retrained),('paired_delta_pp',deltas)]:
        for field,value in mean_ci(values).items():
            assert math.isclose(value,summary[name][field],abs_tol=1e-9),(environment,name,field)
    qualified=mean_ci(deltas)['mean']>=-5
    assert summary['qualified_for_n100'] is qualified
    checks[environment]={'verified':True,'episodes_per_model':250,'paired_delta_pp':mean_ci(deltas),
                         'qualified_for_n100':qualified,'source_summary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'inputs':inputs}
OUTPUT.write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({e:{k:v for k,v in r.items() if k!='inputs'} for e,r in checks.items()},indent=2))
