import difflib, sys, re
a = open(sys.argv[1], encoding='utf-8').read().splitlines()
b = open(sys.argv[2], encoding='utf-8').read().splitlines()
def tok(s): return re.findall(r'\s+|[^\s]+', s)
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
segs=[]
for op,i1,i2,j1,j2 in sm.get_opcodes():
    if op=='equal': continue
    A='\n'.join(a[i1:i2]); B='\n'.join(b[j1:j2])
    ta,tb=tok(A),tok(B)
    s2=difflib.SequenceMatcher(None,ta,tb,autojunk=False)
    for o,x1,x2,y1,y2 in s2.get_opcodes():
        if o=='equal': continue
        segs.append((j1+1,''.join(ta[x1:x2]),''.join(tb[y1:y2])))
for k,(ln,o,n) in enumerate(segs):
    print(f"#{k+1} L{ln}\n  OLD: {o!r}\n  NEW: {n!r}")
