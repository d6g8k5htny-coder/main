"""NON-CERTIFYING text check (no floats). C422-E1 per string leaf: each run id leaf names its PR head and actual tested merge."""
import json, re, collections
D='<SCRATCH>/lvs/register_update14/verify_reg/'
MAP={ # run -> (head prefix, merge prefix, base prefix or None)
 '38116277597':('00d2ac6e','c4c66bb8',None),
 '38121713840':('910809d6','5b198699',None),
 '38122386459':('d9b8dd33','866136cc',None),
 '38150363063':('12e072d6','86a6e8c8',None),
 '38180705952':('48bfcbae','9833d6d9',None),
 '38195810463':('c6ce0256','1320c5b2',None),
 '38277458954':('9e305ea5','1376175c','a0323b94'),
}
def leaves(x,p=()):
    if isinstance(x,dict):
        for k,v in x.items(): yield from leaves(v,p+(k,))
    elif isinstance(x,list):
        for i,v in enumerate(x): yield from leaves(v,p+(i,))
    elif isinstance(x,str): yield p,x
for fn in ['REGISTER.committed.json','REGISTER.new.json']:
    d=json.load(open(D+fn))
    ids=collections.Counter(); bad=[]
    for p,s in leaves(d):
        for rid in set(re.findall(r'(?<!\d)38\d{9}(?!\d)',s)):
            ids[rid]+=1
            if rid not in MAP: bad.append((p,rid,'UNKNOWN')); continue
            h,m,b=MAP[rid]
            if h not in s or m not in s or (b and b not in s): bad.append((p,rid,h in s,m in s,b in s if b else None))
    print(fn, dict(ids)); print(' failures', len(bad))
    for x in bad[:20]: print('  ',x)
# Finer: in new register, for each occurrence of 38277458954, check head & merge within the same parenthetical window (next 400 chars or prev 400)
d=json.load(open(D+'REGISTER.new.json'))
n=0; fails=[]
for p,s in leaves(d):
    for mt in re.finditer('38277458954',s):
        n+=1
        win=s[max(0,mt.start()-300):mt.end()+400]
        if not ('9e305ea5' in win and '1376175c' in win): fails.append((p,s[max(0,mt.start()-120):mt.end()+200]))
print('occurrences of 38277458954:',n,' window failures:',len(fails))
for f in fails: print(f)
