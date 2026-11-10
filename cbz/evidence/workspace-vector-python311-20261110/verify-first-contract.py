import hashlib
import json
from pathlib import Path
import re
import zipfile

root = Path(__file__).resolve().parent
p = root / 'contract-artifact-13749492960.zip'
raw = p.read_bytes()
expected = '7eae08a4c6b6b58039f15fc40a7245b1f6ededd1bc71fea5276df7a162b92edb'
if hashlib.sha256(raw).hexdigest() != expected:
    raise ValueError('artifact digest mismatch')
with zipfile.ZipFile(p) as z:
    names = z.namelist()
    if len(names) != len(set(names)) or len(names) != 106 or z.testzip() is not None:
        raise ValueError('artifact entry or CRC mismatch')
    summaries = [name for name in names if name.endswith('/summary.json')]
    if len(summaries) != 1:
        raise ValueError('unexpected summary population')
    folder = summaries[0].rsplit('/', 1)[0]
    summary = json.loads(z.read(summaries[0]))
    if len(summary) != 17:
        raise ValueError('unexpected partial summary')
    events = [name for name in names if name.startswith(folder + '/')
              and name.endswith('.json') and name != summaries[0]]
    if len(events) != 26:
        raise ValueError('unexpected retained event count')
    nonces, rows = set(), []
    for name in events:
        event = json.loads(z.read(name))
        base = name[:-5]
        source = z.read(base + '.py')
        stderr = z.read(base + '.stderr')
        z.read(base + '.stdout')
        digest = hashlib.sha256(source).hexdigest()
        if event['source_before'] != digest or event['source_after'] != digest:
            raise ValueError('source binding mismatch: ' + name)
        if event['tests_run'] != 1 or any(event[k] != 0 for k in ('errors','skipped','expected_failures','unexpected_successes')):
            raise ValueError('invalid child event: ' + name)
        nonce = event['nonce']
        if not re.fullmatch('[0-9a-f]{32}', nonce) or nonce in nonces:
            raise ValueError('invalid nonce')
        nonces.add(nonce)
        rows.append({'name':name,'source_sha256':digest,'stderr_sha256':hashlib.sha256(stderr).hexdigest(),
                     'failures':event['failures']})
    baselines = [r for r in rows if r['name'].endswith('-baseline.json')]
    mutants = [r for r in rows if r['name'].endswith('-mutant.json')]
    admitted = [r for r in mutants if len(r['failures']) == 1 and r['failures'][0]['selected_body'] is True]
    refused = [r for r in mutants if len(r['failures']) == 1 and r['failures'][0]['selected_body'] is False]
    if len(baselines) != 13 or any(r['failures'] for r in baselines) or len(mutants) != 13 or len(admitted) != 4 or len(refused) != 9:
        raise ValueError('unexpected retained outcomes')
    result = {'artifact':13749492960,'sha256':expected,'unique_members':len(names),'crc':'valid',
              'summary_rows':17,'baseline_events':len(baselines),'admitted_mutants':len(admitted),
              'refused_mutants':len(refused),'event_sources':'matched','unique_nonces':len(nonces),
              'optimized':'not present','scope':'retained failed CPython 3.11 run; no qualification'}
(root/'artifact-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
