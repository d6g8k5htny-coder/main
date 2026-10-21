import sys, hashlib, re
targets = {
 "026": (11725, "e4d8094e", "BEGIN_TBG4_FROZEN_BODY", "END_TBG4_FROZEN_BODY"),
 "022": (21932, "e5a61739", "BEGIN_TBG3_FROZEN_BODY", "END_TBG3_FROZEN_BODY"),
 "032": (12106, "3531e6e5", "BEGIN_TBG2_UNIFORM_FROZEN_BODY", "END_TBG2_UNIFORM_FROZEN_BODY"),
 "036v11": (16856, "3416ddee", "BEGIN_TBG3B_OUTWARD_ESCAPE_V11_FROZEN_BODY", "END_TBG3B_OUTWARD_ESCAPE_V11_FROZEN_BODY"),
 "038": (10410, "5fd2413c", "BEGIN_TBG3_SYNTHESIS_V2_FROZEN_BODY", "END_TBG3_SYNTHESIS_V2_FROZEN_BODY"),
 "036v10": (19633, "4ee99e81", "BEGIN_TBG3B_OUTWARD_ESCAPE_FROZEN_BODY", "END_TBG3B_OUTWARD_ESCAPE_FROZEN_BODY"),
 "021": (11924, "3369ddd5", "BEGIN_TBG2_FROZEN_BODY", "END_TBG2_FROZEN_BODY"),
}
d = sys.argv[1]

def span(text, b, e):
    # text already LF-normalized, BOM stripped
    lines = text.split("\n")
    bi = [i for i,l in enumerate(lines) if l == b]
    ei = [i for i,l in enumerate(lines) if l == e]
    if len(bi)!=1 or len(ei)!=1: return None, (len(bi), len(ei))
    return "\n".join(lines[bi[0]+1:ei[0]]), (bi[0]+1, ei[0]+1)

def finish(s):
    return s.rstrip("\n") + "\n"

def brh_halve(s):
    # halve each run of empty lines (between non-empty lines): run of n empty lines -> n//2
    lines = s.split("\n")
    out=[]; run=0
    for l in lines:
        if l == "":
            run += 1
        else:
            out.extend([""]*(run//2)); run=0; out.append(l)
    out.extend([""]*(run//2))
    return "\n".join(out)

def brh_halve_ceil(s):
    lines = s.split("\n"); out=[]; run=0
    for l in lines:
        if l == "": run += 1
        else:
            out.extend([""]*((run+1)//2)); run=0; out.append(l)
    out.extend([""]*((run+1)//2))
    return "\n".join(out)

def collapse1(s):
    return re.sub(r"\n{3,}", "\n\n", s)

def remove_blank(s):
    return "\n".join(l for l in s.split("\n") if l != "")

rules = {
 "R1_literal": lambda s: finish(s),
 "R2_halve_floor": lambda s: finish(brh_halve(s)),
 "R3_halve_ceil": lambda s: finish(brh_halve_ceil(s)),
 "R4_collapse_to_one_blank": lambda s: finish(collapse1(s)),
 "R5_no_blank_lines": lambda s: finish(remove_blank(s)),
}
for name,(sz,pre,b,e) in targets.items():
    raw = open(f"{d}/{name}.txt","rb").read()
    bom = raw.startswith(b"\xef\xbb\xbf")
    t = raw.decode("utf-8")
    if t.startswith("﻿"): t = t[1:]
    t = t.replace("\r\n","\n").replace("\r","\n")
    s, info = span(t,b,e)
    print(f"== {name} target {sz}/{pre} bom={bom} markers(lines)={info}")
    if s is None: continue
    for rn,f in rules.items():
        body = f(s).encode("utf-8")
        h = hashlib.sha256(body).hexdigest()
        ok = (len(body)==sz and h.startswith(pre))
        print(f"   {rn:28s} {len(body):6d} {h} {'MATCH' if ok else ''}")
