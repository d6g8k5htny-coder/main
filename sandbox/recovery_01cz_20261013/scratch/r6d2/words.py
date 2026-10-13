import json,re,sys,difflib
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
def flat(x,p='',out=None):
    if out is None: out={}
    if isinstance(x,dict):
        for k,v in x.items(): flat(v,p+'/'+k,out)
    elif isinstance(x,list):
        for i,v in enumerate(x): flat(v,p+'[%d]'%i,out)
    elif isinstance(x,str): out[p]=x
    return out
fa=flat(a); fb=flat(b)
pat=re.compile(r'kernel-check|kernel check|\bproved\b|\bproven\b|review|accept|verified|certif|receipt|evidence|closes|discharg|promot|reclassif|establish',re.I)
for p,y in fb.items():
    x=fa.get(p)
    if x==y: continue
    if x is None: ins=[y]
    else:
        sm=difflib.SequenceMatcher(None,re.findall(r'\S+|\s+',x),re.findall(r'\S+|\s+',y),autojunk=False)
        yt=re.findall(r'\S+|\s+',y)
        ins=[''.join(yt[j1:j2]) for op,i1,i2,j1,j2 in sm.get_opcodes() if op in('insert','replace')]
    for s in ins:
        for m in pat.finditer(s):
            print(p, '|', m.group(0), '|', s[max(0,m.start()-90):m.end()+90].replace('\n',' '))
