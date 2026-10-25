# Independent implementation (Claude 01Cz) of the UL v15 obligation-fingerprint recipe,
# written from the published description of model.py lines 11-14 and 32-50; the published
# source itself is not imported or executed.
import json, hashlib, sys
def canon(o): return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
def h(o): return hashlib.sha256(canon(o)).hexdigest()
def fingerprints(d):
    rec = {r['id']: r for r in d['records']}
    obl = {o['conclusion']: o for o in d['obligations']}
    imp = {x['id']: x for x in d['imports']}
    cor = {x['id']: x for x in d['corrections']}
    loc = {x['id']: x for x in d['local_evidence']}
    memo, stack = {}, set()
    def fp(n):
        if n in memo: return memo[n]
        if n in stack: raise SystemExit('cycle at ' + n)
        stack.add(n); o = obl[n]; ps = sorted(o['premises'])
        blob = {'record': rec[n],
                'obligation': {k: v for k, v in o.items() if k != 'fingerprint'},
                'premise_fingerprints': [[p, fp(p)] for p in ps],
                'import_descriptors': [[i, imp[i]] for i in sorted(o['external_theorem_imports'])],
                'source_correction_descriptors': [[i, cor[i]] for i in sorted(o['source_corrections'])],
                'local_check_descriptors': [[i, loc[i]] for i in sorted(o['local_evidence_refs'])]}
        memo[n] = h(blob); stack.discard(n); return memo[n]
    for n in sorted(rec): fp(n)
    return memo
if __name__ == '__main__':
    d = json.load(open(sys.argv[1], encoding='utf-8'))
    f = fingerprints(d); obl = {o['conclusion']: o for o in d['obligations']}
    match = sum(obl[n]['fingerprint'] == v for n, v in f.items())
    print(f'{sys.argv[1]}: {match}/{len(f)} stored fingerprints reproduced; links={sum(len(o["premises"]) for o in d["obligations"])}')
