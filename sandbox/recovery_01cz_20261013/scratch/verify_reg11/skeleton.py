import json, difflib, re
def get(d, path):
    for p in path: d = d[p]
    return d
bp=json.load(open("be_parent.json")); be=json.load(open("be.json")); old=json.load(open("old.json")); new=json.load(open("../repos/math_r5/specifications/lean_statements_20261010/REGISTER.json"))
W4=[("87309efad9b804c53c543b60fdb466de1f797971417363cc2e8fa1e10eac26bd","<SHA>"),
    ("9833d6d9f9facc335e1a56e243c3144adefe6225","<MERGE>"),
    ("12e072d614dbe6e91b66d22a2ab55cf87860d693","<BASE>"),
    ("38180705952","<RUN>"),("48bfcbae","<HEAD>"),("12e072d6","<PREVHEAD>"),
    ("86a6e8c876f7904cadda2d006a56f8792dc5c1af","<PMERGE>"),("79731b961fac555f33be3cd90e379b246a5a80e3","<PBASE>"),("38150363063","<PRUN>"),
    ("theorem_count 73","theorem_count <COUNT>"),("theorem_count 60","theorem_count <PCOUNT>"),("All 73 ","All <COUNT> "),("all 73 ","all <COUNT> "),
    ("PR #434","<PR>"),("PR #423","<BASEPR>"),
    ("modules Side24Analysis, Side24Interfaces, RNCluster","modules <MODS>")]
W5=[("d8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff","<SHA>"),
    ("1320c5b217e491d301ea140e5cc25b07ff5a21b7","<MERGE>"),
    ("be01479435f98f34b802258ee9ec3c3578f37169","<BASE>"),
    ("38195810463","<RUN>"),("c6ce0256","<HEAD>"),("be014794","<PREVHEAD>"),
    ("79d3964cf70e2eadad5c9629eef8f4b47f218dd0","<PMERGE>"),("12e072d614dbe6e91b66d22a2ab55cf87860d693","<PBASE>"),("38187538744","<PRUN>"),
    ("theorem_count 75","theorem_count <COUNT>"),("theorem_count 73","theorem_count <PCOUNT>"),("All 75 ","All <COUNT> "),("all 75 ","all <COUNT> "),
    ("PR #439","<PR>"),("PR #434","<BASEPR>"),
    ("modules Side24Covariance, Side24ConeMoment","modules <MODS>")]
def sk(s, W):
    for a,b in W: s=s.replace(a,b)
    return s
def toks(s): return re.findall(r"<[A-Z]+>|[^\s;.,()<]+|[\s;.,()<]", s)
def hunks(a,b):
    ta,tb=toks(a),toks(b)
    sm=difflib.SequenceMatcher(None,ta,tb,autojunk=False)
    return [(op,''.join(ta[i1:i2]),''.join(tb[j1:j2])) for op,i1,i2,j1,j2 in sm.get_opcodes() if op!="equal"]
paths=[tuple(p) for p,_,_ in json.load(open("changed_r5.json"))]
r4paths=[tuple(p) for p,_,_ in json.load(open("changed_r4.json"))]
for p in paths:
    key="/".join(map(str,p))
    h5=hunks(sk(get(old,p),W5), sk(get(new,p),W5))
    if p in r4paths:
        h4=hunks(sk(get(bp,p),W4), sk(get(be,p),W4))
    else:
        h4=None
    print(key, "R4-shape-equal" if h4==h5 else "DIFFERS", len(h5))
    if h4!=h5 and key!="claims/0/next_step":
        print("  R4:",h4); print("  R5:",h5)

print("=== corroboration placement")
CN = "All 75 theorems of PR head c6ce0256, including every theorem of the tiers above, are receipted on that head by run 38195810463 (actual tested merge 1320c5b217e491d301ea140e5cc25b07ff5a21b7 with base be01479435f98f34b802258ee9ec3c3578f37169; receipt theorem_count 75; artifact lean-specification-proofs-38195810463-1, sha256 d8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff). "
CR = "; all 75 theorems of PR head c6ce0256 receipted on that head by run 38195810463 (actual tested merge 1320c5b217e491d301ea140e5cc25b07ff5a21b7 with base be01479435f98f34b802258ee9ec3c3578f37169; receipt theorem_count 75; artifact lean-specification-proofs-38195810463-1, sha256 d8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff)"
for p in paths:
    if p[2]!="kernel_evidence": continue
    v=get(new,p); o=get(old,p)
    if p[-1]=="note":
        ok = v.count(CN)==1 and (CN+"`proved` is the label") in v and v.replace(CN,"",1)==sk_ok if False else None
        i=v.find(CN)
        print("/".join(map(str,p)), "count",v.count(CN), "followed by proved:", v[i+len(CN):i+len(CN)+20]=="`proved` is the labe")
    else:
        print("/".join(map(str,p)), "count",v.count(CN.strip()) , "R endswith CR:", v.endswith(CR), "old+CR==new:", (o+CR)==v)
# row 2..54 note: removing CN gives old exactly?
print("=== removing corroboration restores c6ce0256 text (rows other than 0):")
for p in paths:
    if p[2]!="kernel_evidence" or p[1]==0: continue
    v=get(new,p); o=get(old,p)
    if p[-1]=="note": print("/".join(map(str,p)), v.replace(CN,"",1)==o)
# round 4 placement for comparison
C4N = CN.replace("75","73").replace("c6ce0256","48bfcbae").replace("38195810463","38180705952").replace("1320c5b217e491d301ea140e5cc25b07ff5a21b7","9833d6d9f9facc335e1a56e243c3144adefe6225").replace("be01479435f98f34b802258ee9ec3c3578f37169","12e072d614dbe6e91b66d22a2ab55cf87860d693").replace("d8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff","87309efad9b804c53c543b60fdb466de1f797971417363cc2e8fa1e10eac26bd")
print("=== R4 corroboration reproduced by substitution in be014794 rows:", [ (p[1], get(be,p).count(C4N)) for p in paths if p[2]=="kernel_evidence" and p[-1]=="note"])
