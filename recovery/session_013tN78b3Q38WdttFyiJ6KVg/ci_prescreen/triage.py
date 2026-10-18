"""Classify pre-screen results against the main baseline. Output triage.json and a table."""
import glob, json, os, sys
OUT = sys.argv[1]
ENV_TESTS = ('test_missing_promisor', 'test_complete_promisor_objects_remain_valid')
ENV_MARKERS = ('invalid run ID',)
def key(r): return (r['workflow'], r['job'], json.dumps(r.get('matrix') or {}, sort_keys=True))
def failed_step(r): return next((s for s in r['steps'] if s.get('exit') and not s.get('continue_on_error')), None)
base = json.load(open(os.path.join(OUT, 'target-0.json')))
B = {key(r): r for r in base['runs']}
rows, cands = [], []
for f in sorted(glob.glob(os.path.join(OUT, 'target-*.json')), key=lambda p: int(p.split('-')[-1][:-5])):
    d = json.load(open(f))
    if d['pr'] == 0: continue
    counts = {'PASS': 0, 'INHERITED': 0, 'ENV': 0, 'UNSUPPORTED': 0, 'SKIPPED': 0, 'CANDIDATE': 0}
    for r in d['runs']:
        if r['status'] in ('PASS', 'UNSUPPORTED', 'SKIPPED'): counts[r['status']] += 1; continue
        s = failed_step(r); b = B.get(key(r)); bs = failed_step(b) if b else None
        ft = set(s.get('failed_tests') or []) if s else set()
        env_only = ft and all(any(e in t for e in ENV_TESTS) for t in ft)
        marker_env = s and any(m in (s.get('tail') or '') for m in ENV_MARKERS) and not ft
        if b and b['status'] == 'FAIL' and bs and bs['name'] == s['name'] and (set(bs.get('failed_tests') or []) >= ft) \
                and (ft or (bs.get('markers') == s.get('markers'))):
            cls = 'INHERITED'
        elif env_only or marker_env:
            cls = 'ENV'
        else:
            cls = 'CANDIDATE'
            new = sorted(ft - set((bs or {}).get('failed_tests') or [])) if b else sorted(ft)
            cands.append({'pr': d['pr'], 'head': d['head'], 'commit': d['commit'], 'workflow': r['workflow'], 'job': r['job'],
                          'matrix': r.get('matrix'), 'step': s['name'] if s else None, 'exit': s['exit'] if s else None,
                          'baseline': (b or {}).get('status', 'ABSENT'), 'new_failed_tests': new[:60],
                          'markers': (s or {}).get('markers'), 'tail': (s or {}).get('tail', '')[-2000:]})
        counts[cls] += 1
    rows.append({'pr': d['pr'], 'head': d['head'][:8], 'status': d['status'], 'selected': len(d.get('selected', [])), **counts})
json.dump({'rows': rows, 'candidates': cands,
           'baseline': {'commit': base['commit'], 'runs': [{k: r[k] for k in ('workflow', 'job', 'status')} | {'reason': r.get('reason', ''),
                        'failed_tests': len((failed_step(r) or {}).get('failed_tests') or [])} for r in base['runs']]}},
          open(os.path.join(OUT, 'triage.json'), 'w'), indent=1)
print('pr head selected pass inherited env unsupported skipped CANDIDATE')
for r in rows: print(r['pr'], r['head'], r['selected'], r['PASS'], r['INHERITED'], r['ENV'], r['UNSUPPORTED'], r['SKIPPED'], r['CANDIDATE'])
print('baseline failing jobs:', [(r['workflow'], r['job']) for r in base['runs'] if r['status'] == 'FAIL'])
print('candidates:', len(cands))
