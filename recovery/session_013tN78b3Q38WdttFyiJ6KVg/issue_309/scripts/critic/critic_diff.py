import sys, json
o = json.load(open(sys.argv[1]))
a, r1, r2 = [x['_rowsAP'] for x in o]
def diff(p, q):
    keys = sorted(set(p) | set(q), key=int); n = 0; rows = set()
    for k in keys:
        pa = p.get(k, [None]*16); qa = q.get(k, [None]*16)
        for i in range(16):
            if (pa[i] or '') != (qa[i] or ''): n += 1; rows.add(int(k))
    return n, (min(rows), max(rows), len(rows)) if rows else None
print('0918->R1', diff(a, r1)); print('R1->R2', diff(r1, r2))
