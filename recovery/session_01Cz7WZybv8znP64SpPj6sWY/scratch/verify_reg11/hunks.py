import json, sys, difflib, re
def get(d, path):
    for p in path: d = d[p]
    return d
bp=json.load(open("be_parent.json")); be=json.load(open("be.json")); old=json.load(open("old.json")); new=json.load(open("../repos/math_r5/specifications/lean_statements_20261010/REGISTER.json"))
def toks(s): return re.findall(r"[^\s;.,()]+|[\s;.,()]", s)
def hunks(a, b):
    ta, tb = toks(a), toks(b)
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    out=[]
    for op,i1,i2,j1,j2 in sm.get_opcodes():
        if op!="equal": out.append((op,''.join(ta[i1:i2]),''.join(tb[j1:j2])))
    return out
paths=[p for p,_,_ in json.load(open("changed_r5.json"))]
for p in paths:
    if sys.argv[1:] and "/".join(map(str,p)) not in sys.argv[1:]: continue
    print("=====", "/".join(map(str,p)), "id=", new["claims"][p[1]].get("id"))
    print(" R4:")
    for h in hunks(get(bp,p), get(be,p)): print("   ",h[0],"\n      -",repr(h[1]),"\n      +",repr(h[2]))
    print(" R5:")
    for h in hunks(get(old,p), get(new,p)): print("   ",h[0],"\n      -",repr(h[1]),"\n      +",repr(h[2]))
