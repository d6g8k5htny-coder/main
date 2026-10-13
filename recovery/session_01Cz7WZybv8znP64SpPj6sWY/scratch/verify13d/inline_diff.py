import difflib, sys, re
a = open(sys.argv[1], encoding='utf-8').read().splitlines()
b = open(sys.argv[2], encoding='utf-8').read().splitlines()
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
def tok(s): return re.findall(r'\s+|[^\s]+', s)
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op == 'equal': continue
    print(f"=== {op} old {i1+1}-{i2} new {j1+1}-{j2}")
    if op == 'replace' and (i2-i1) == (j2-j1):
        pairs = zip(range(i1,i2), range(j1,j2))
        for i,j in pairs:
            ta, tb = tok(a[i]), tok(b[j])
            s2 = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
            out=[]
            for o, x1,x2,y1,y2 in s2.get_opcodes():
                if o=='equal':
                    seg=''.join(ta[x1:x2])
                    if len(seg)>80: seg=seg[:35]+' ... '+seg[-35:]
                    out.append(seg)
                else:
                    out.append('[-'+''.join(ta[x1:x2])+'-]{+'+''.join(tb[y1:y2])+'+}')
            print(f"  L{i+1}->L{j+1}: "+''.join(out))
    else:
        for i in range(i1,i2): print(f"  -L{i+1}: {a[i]}")
        for j in range(j1,j2): print(f"  +L{j+1}: {b[j]}")
