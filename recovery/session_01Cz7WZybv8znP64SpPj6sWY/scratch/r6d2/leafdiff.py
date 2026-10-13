import json,sys
from collections import OrderedDict
a=json.load(open(sys.argv[1]),object_pairs_hook=OrderedDict)
b=json.load(open(sys.argv[2]),object_pairs_hook=OrderedDict)
def flat(x,p='',out=None,keys=None):
    if out is None: out=OrderedDict(); keys={}
    if isinstance(x,dict):
        keys[p]=list(x.keys())
        for k,v in x.items(): flat(v,p+'/'+k,out,keys)
    elif isinstance(x,list):
        keys[p]=len(x)
        for i,v in enumerate(x): flat(v,p+'[%d]'%i,out,keys)
    else: out[p]=x
    return out,keys
fa,ka=flat(a); fb,kb=flat(b)
changed=[p for p in fa if p in fb and fa[p]!=fb[p]]
added=[p for p in fb if p not in fa]
removed=[p for p in fa if p not in fb]
print('changed',len(changed),'added',len(added),'removed',len(removed))
# label claim ids
def cid(p,tree):
    import re
    m=re.match(r'/claims\[(\d+)\]',p)
    if m: return tree['claims'][int(m.group(1))].get('id')
    return None
for p in changed: print('C',p,cid(p,b))
for p in added: print('A',p, repr(fb[p])[:80])
for p in removed: print('R',p)
ko=[p for p in ka if p in kb and isinstance(ka[p],list) and ka[p]!=kb[p]]
print('dict key-order/set changes:',ko)
lc=[p for p in ka if p in kb and isinstance(ka[p],int) and ka[p]!=kb[p]]
print('list length changes:',[(p,ka[p],kb[p]) for p in lc])
# serialization check
raw=open(sys.argv[2],encoding='utf-8').read()
print('serialization indent2 ensure_ascii False + newline:', raw==json.dumps(b,indent=2,ensure_ascii=False)+'\n')
rawa=open(sys.argv[1],encoding='utf-8').read()
print('base serialization same style:', rawa==json.dumps(a,indent=2,ensure_ascii=False)+'\n')
