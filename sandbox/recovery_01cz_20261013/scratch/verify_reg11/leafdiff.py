import json, sys
def leaves(o, p=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, p + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, p + (i,))
    else:
        yield p, o
def structure(o):
    if isinstance(o, dict):
        return {k: structure(v) for k, v in o.items()}
    if isinstance(o, list):
        return [structure(v) for v in o]
    return type(o).__name__
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
print("structure equal (keys, order, types):", structure(a) == structure(b))
print("key order equal:", list(leaves(a)) and [p for p,_ in leaves(a)] == [p for p,_ in leaves(b)])
la = dict(leaves(a)); lb = dict(leaves(b))
print("only in old:", [k for k in la if k not in lb])
print("only in new:", [k for k in lb if k not in la])
ch = [k for k in la if k in lb and la[k] != lb[k]]
print("changed leaves:", len(ch))
for k in ch:
    print("  ", k, type(la[k]).__name__, type(lb[k]).__name__)
json.dump([[list(k), la[k], lb[k]] for k in ch], open(sys.argv[3], "w"), indent=1)
