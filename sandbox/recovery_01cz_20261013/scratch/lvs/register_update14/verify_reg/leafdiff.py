"""NON-CERTIFYING (no floats; text diff only). Leaf-by-leaf diff of REGISTER.json: committed vs worktree."""
import json, sys
D='<SCRATCH>/lvs/register_update14/verify_reg/'
a=json.load(open(D+'REGISTER.committed.json'))
b=json.load(open(D+'REGISTER.new.json'))
def leaves(x,p=()):
    if isinstance(x,dict):
        yield p+('{keys}',), tuple(x.keys())
        for k,v in x.items(): yield from leaves(v,p+(k,))
    elif isinstance(x,list):
        yield p+('[len]',), len(x)
        for i,v in enumerate(x): yield from leaves(v,p+(i,))
    else:
        yield p, x
la=dict(leaves(a)); lb=dict(leaves(b))
assert set(la)==set(lb), (set(la)^set(lb))
ch=[k for k in la if la[k]!=lb[k]]
print('changed', len(ch))
out=[]
for k in ch:
    assert isinstance(la[k],str) and isinstance(lb[k],str), k
    # describe path with row id
    p=list(k)
    if p[0]=='claims':
        rid=a['claims'][p[1]].get('id')
        p[1]=f"{p[1]}:{rid}"
    out.append({'path':p,'old':la[k],'new':lb[k]})
    print(p)
json.dump(out,open(D+'changed_leaves.json','w'),indent=1,ensure_ascii=False)
