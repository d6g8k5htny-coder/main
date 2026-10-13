import json,re,sys,os
reg=json.load(open(sys.argv[1]))
base=sys.argv[2]
def docfor(module,decl):
    src=open(os.path.join(base,module)).read()
    name=decl.split('.')[-1]
    # find docstring immediately preceding 'theorem name'
    m=re.search(r'/--((?:(?!-/).)*?)-/\s*theorem\s+'+re.escape(name)+r'\b',src,re.S)
    return m.group(1) if m else None
res=[]
for ci,c in enumerate(reg['claims']):
    for si,s in enumerate(c.get('statements',[])):
        if s.get('kind')=='theorem' and 'lean_statement_proofs_20261011' in s.get('package',''):
            d=docfor(s['module'],s['declaration'])
            if d is None: res.append((ci,si,s['declaration'],'NODOC')); continue
            m=re.search(r'Does not establish:(.*)$',d,re.S)
            if not m: res.append((ci,si,s['declaration'],'NOFIELD')); continue
            dn=' '.join(m.group(1).split())
            dnc=s['does_not_claim']
            res.append((ci,si,s['declaration'],'EXACT' if dn==dnc else ('PREFIX' if dn.startswith(dnc) else ('CONTAINS' if dnc in dn else 'DIFF'))))
from collections import Counter
print(Counter(r[3] for r in res))
for r in res:
    if r[3]!='EXACT': print(r)
for r in res:
    if 'Side24Covariance' in r[2] or 'Side24ConeMoment' in r[2]: print(r)
