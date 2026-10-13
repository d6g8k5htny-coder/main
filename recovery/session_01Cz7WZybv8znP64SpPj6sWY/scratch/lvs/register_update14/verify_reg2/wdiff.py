import json, difflib, re, sys
d=json.load(open(sys.argv[1]))
sel = sys.argv[2:] 
for e in d:
    p=e['path']
    key='/'.join(map(str,p))
    if sel and not any(s==key for s in sel): continue
    a=re.findall(r'\S+|\s+', e['old']); b=re.findall(r'\S+|\s+', e['new'])
    sm=difflib.SequenceMatcher(None,a,b,autojunk=False)
    print('=====',key, len(e['old']),'->',len(e['new']))
    for op,i1,i2,j1,j2 in sm.get_opcodes():
        if op=='equal': continue
        print(f'  [{op}] -{"".join(a[i1:i2])!r}\n         +{"".join(b[j1:j2])!r}')
