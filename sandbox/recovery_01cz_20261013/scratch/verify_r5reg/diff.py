import json, sys
old = json.load(open(sys.argv[1]))
new = json.load(open(sys.argv[2]))
def walk(o, n, path, out):
    if type(o) != type(n):
        out.append(('TYPE', path, o, n)); return
    if isinstance(o, dict):
        ko, kn = list(o.keys()), list(n.keys())
        if ko != kn:
            out.append(('KEYS', path, ko, kn))
        for k in ko:
            if k in n:
                walk(o[k], n[k], path + [k], out)
            else:
                out.append(('DEL', path+[k], o[k], None))
        for k in kn:
            if k not in o:
                out.append(('ADD', path+[k], None, n[k]))
    elif isinstance(o, list):
        if len(o) != len(n):
            out.append(('LEN', path, len(o), len(n)))
        for i in range(min(len(o), len(n))):
            walk(o[i], n[i], path+[i], out)
        for i in range(len(o), len(n)):
            out.append(('ADD', path+[i], None, n[i]))
        for i in range(len(n), len(o)):
            out.append(('DEL', path+[i], o[i], None))
    else:
        if o != n:
            out.append(('CHG', path, o, n))
out = []
walk(old, new, [], out)
def label(path):
    s = []
    cur = new
    for p in path:
        s.append(str(p))
    return '/'.join(s)
for kind, path, o, n in out:
    # identify claim id
    cid = ''
    if len(path) >= 2 and path[0] == 'claims':
        cid = new['claims'][path[1]].get('id') or new['claims'][path[1]].get('claim_id')
    print(kind, cid, label(path))
print('TOTAL', len(out))
json.dump([[k, p, o, n] for k,p,o,n in out], open(sys.argv[3], 'w'), indent=1)
