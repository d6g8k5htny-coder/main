import json,sys
old=json.load(open(sys.argv[1])); new=json.load(open(sys.argv[2]))
def leaves(o,p=()):
    if isinstance(o,dict):
        for k,v in o.items(): yield from leaves(v,p+(k,))
    elif isinstance(o,list):
        for i,v in enumerate(o): yield from leaves(v,p+(i,))
    else: yield p,o
L1=dict(leaves(old)); L2=dict(leaves(new))
def fmt(p): return '/'.join(map(str,p))
added=[p for p in L2 if p not in L1]; removed=[p for p in L1 if p not in L2]
changed=[p for p in L1 if p in L2 and L1[p]!=L2[p]]
print("added",len(added)); [print(" +",fmt(p)) for p in added]
print("removed",len(removed)); [print(" -",fmt(p)) for p in removed]
print("changed",len(changed)); [print(" ~",fmt(p)) for p in changed]
# keyorder
