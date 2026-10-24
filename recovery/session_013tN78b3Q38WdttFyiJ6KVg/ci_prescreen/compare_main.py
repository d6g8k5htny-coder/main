import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
base = json.loads((d / 'mt-0.json').read_text())
bf = {k: set(v.get('failed_tests') or []) for k, v in base['checks'].items()}
for f in sorted(d.glob('mt-*.json'), key=lambda p: int(p.stem[3:])):
    r = json.loads(f.read_text())
    if r['pr'] == 0:
        continue
    if r['status'] == 'CONFLICT':
        print(r['pr'], 'CONFLICT'); continue
    out = []
    for k, v in r['checks'].items():
        new = set(v.get('failed_tests') or []) - bf.get(k, set())
        if v['exit'] != 0 or new:
            out.append(f"{k}: exit {v['exit']} new_fail {len(new)} {v.get('summary')}")
    print(r['pr'], r['status'], r['commit'][:8], '; '.join(out) or 'same as baseline')
