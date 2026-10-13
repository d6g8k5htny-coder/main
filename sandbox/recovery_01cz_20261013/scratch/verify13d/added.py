import difflib, sys, re
a = open(sys.argv[1], encoding='utf-8').read()
b = open(sys.argv[2], encoding='utf-8').read()
def tok(s): return re.findall(r'\s+|[^\s]+', s)
ta,tb=tok(a),tok(b)
sm=difflib.SequenceMatcher(None,ta,tb,autojunk=False)
n=0
for o,x1,x2,y1,y2 in sm.get_opcodes():
    if o=='equal': continue
    n+=1
    print(f"#{n} {o}\n  OLD: {''.join(ta[x1:x2])[:3000]!r}\n  NEW: {''.join(tb[y1:y2])[:6000]!r}")
