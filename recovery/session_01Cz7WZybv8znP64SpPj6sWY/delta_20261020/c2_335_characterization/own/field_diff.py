"""For one seeded mutant, run transfer.compute() and list which RESULTS.json fields differ (read-only)."""
import json, os, sys
pkt = sys.argv[1]; mutant = sys.argv[2]
sys.path.insert(0, pkt)
import transfer as T
T.MUTANT = mutant
res = T.compute()
stored = json.load(open(os.path.join(pkt, 'RESULTS.json')))
diffs = []
def walk(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            walk(a.get(k, '<absent>'), b.get(k, '<absent>'), path + [k])
    elif a != b:
        diffs.append({'path': '/'.join(map(str, path)), 'stored': a, 'mutant': b})
walk(stored, res, [])
fields = sorted({d['path'].split('/')[-1] for d in diffs})
blocks = sorted({'/'.join(d['path'].split('/')[:2]) for d in diffs})
print(json.dumps({'mutant': mutant, 'n_diffs': len(diffs), 'leaf_fields': fields, 'blocks_touched': len(blocks), 'reference_checks_equal': stored.get('reference_checks') == res.get('reference_checks'), 'sample': diffs[:3]}))
