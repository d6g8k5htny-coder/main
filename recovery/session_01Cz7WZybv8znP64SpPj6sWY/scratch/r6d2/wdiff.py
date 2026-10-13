import json,sys,difflib,re
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
def get(t,p):
    for part in re.findall(r'/([^/\[]+)|\[(\d+)\]',p):
        t=t[part[0]] if part[0] else t[int(part[1])]
    return t
for p in sys.argv[3:]:
    x=get(a,p); y=get(b,p)
    xa=re.findall(r'\S+|\s+',x); ya=re.findall(r'\S+|\s+',y)
    sm=difflib.SequenceMatcher(None,xa,ya,autojunk=False)
    print('=====',p)
    for op,i1,i2,j1,j2 in sm.get_opcodes():
        if op=='equal': continue
        ctx_before=''.join(xa[max(0,i1-12):i1])
        print(f'--[{op}] ...{ctx_before[-120:]}')
        print('   OLD:',''.join(xa[i1:i2]))
        print('   NEW:',''.join(ya[j1:j2]))
