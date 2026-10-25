# Builds the isolated successor-dataset PROPOSAL from the original carrier, the released
# P15 patch (86e94e13) and the independent fingerprint recipe. Original files are untouched.
import json, copy, sys, hashlib
sys.path.insert(0, '.')
from fp_independent import fingerprints, canon
import hashlib as _h
RAW_ORIG = open("dataset_original.json","rb").read(); RAW_PATCH = open("P15_APPLY_READY.patch.json","rb").read()
assert _h.sha256(RAW_ORIG).hexdigest() == "4cfae309b9da989ca6c95881f207c0a097a4dde836f45eacbe0e78d7faf9e4d7", "original input pin"
assert _h.sha256(RAW_PATCH).hexdigest() == "9779bbb850f829dc5ecda3f2796159b22c1fde8f27c195f350e6f63b019d11fc", "patch input pin"
orig = json.loads(RAW_ORIG.decode("utf-8"))
ops = json.load(open('P15_APPLY_READY.patch.json', encoding='utf-8'))
d = copy.deepcopy(orig)
def ptr(doc, path):
    parts = path.split('/')[1:]
    for p in parts[:-1]: doc = doc[int(p)] if isinstance(doc, list) else doc[p]
    k = parts[-1]; return doc, (int(k) if isinstance(doc, list) else k)
for op in ops:
    o, k = ptr(d, op['path'])
    if op['op'] == 'test': assert o[k] == op['value'], op
    else: o[k] = op['value']
# reverse-dependency closure of the three corrected records on the PATCHED graph
obl = {o['conclusion']: o for o in d['obligations']}
consumers = {n: set() for n in obl}
for n, o in obl.items():
    for p in o['premises']: consumers[p].add(n)
seed = {'P15-T1', 'P15-T2', 'P15-T3'}; closure = set(seed); todo = list(seed)
while todo:
    x = todo.pop()
    for c in consumers[x]:
        if c not in closure: closure.add(c); todo.append(c)
print('reverse-dependency closure:', sorted(closure))
assert closure == {'P15-T1', 'P15-T2', 'P15-T3'}, 'closure must be exactly the three corrected records'
old_fp = fingerprints(orig); new_fp = fingerprints(d)
changed = sorted(n for n in new_fp if new_fp[n] != old_fp[n])
print('fingerprints that change:', changed)
assert set(changed) == closure, 'unexpected fingerprint change outside closure'
for n in changed:
    print(f'  {n}: {old_fp[n][:16]}… -> {new_fp[n][:16]}…'); obl[n]['fingerprint'] = new_fp[n]
d['metadata'].setdefault('documented_link_migrations', []).append({
    'id': 'MIG-P15-T1-PREMISE-T2-20261025',
    'obligation': 'OBL:P15-T1', 'added_premise': 'P15-T2',
    'link_count_before': 108, 'link_count_after': 109,
    'basis': 'Math- 79cb314d frontiers/p15_tail_load_20260928/PROOF.md (blob 8187908d) line 222: Equations (T3)-(T4) now prove (T1); finding LOW-P15-T1-PREMISE',
    'proposal': 'main#605 comment 6404165875 (Claude 01Cz); patch 86e94e13',
    'status': 'PROPOSED_NOT_ADOPTED'})
links = sum(len(o['premises']) for o in d['obligations'])
again = fingerprints(d)
stale = [n for n, o in obl.items() if o['fingerprint'] != again[n]]
print('links:', links, '| stale stored fingerprints after rebuild:', stale)
print('records', len(d['records']), 'obligations', len(d['obligations']), 'original_known_required_links', d['metadata']['original_known_required_links'])
json.dump(d, open('dataset_successor_PROPOSAL.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('successor canonical sha256:', hashlib.sha256(canon(d)).hexdigest())
print('original  canonical sha256:', hashlib.sha256(canon(orig)).hexdigest())
