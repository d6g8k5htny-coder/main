import json, sys, difflib, re
ch = json.load(open(sys.argv[1]))
sel = sys.argv[2:] 
def toks(s): return re.findall(r"\S+|\s+", s)
for path, a, b in ch:
    key = "/".join(map(str, path))
    if sel and not any(s == key for s in sel): continue
    ta, tb = toks(a), toks(b)
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    print("=====", key, "len", len(a), "->", len(b))
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal": continue
        print(f"  [{op}] -{''.join(ta[i1:i2])!r}")
        print(f"         +{''.join(tb[j1:j2])!r}")
