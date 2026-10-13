import re, sys
S="<SCRATCH>"
O=S+"/lvs/c254/overlay/ResearchFormalCoreR1/ClosedSuperlevel.lean"
F=S+"/repos/math_c254/formal/ResearchFormalCoreR1/ClosedSuperlevel.lean"
pat=re.compile(r'^\s*(theorem|lemma|def|abbrev|instance|structure|inductive|class|example|axiom|opaque)\s+([^\s({:\[]+)', re.M)
def decls(p):
    txt=open(p,encoding='utf-8').read()
    out=[]
    for m in pat.finditer(txt):
        line=txt.count('\n',0,m.start())+1
        out.append((m.group(1),m.group(2),line))
    return out
do, df = decls(O), decls(F)
print("ORIG decls:"); [print(f"  {k:8s} {n:45s} L{l}") for k,n,l in do]
print("FIXED decls:"); [print(f"  {k:8s} {n:45s} L{l}") for k,n,l in df]
to=[(k,n) for k,n,_ in do if k in("theorem","lemma")]
tf=[(k,n) for k,n,_ in df if k in("theorem","lemma")]
print("theorem/lemma count orig=%d fixed=%d"%(len(to),len(tf)))
print("NAMES_IDENTICAL_IN_ORDER" if to==tf else "NAMES_DIFFER")
ko=[(k,n) for k,n,_ in do]; kf=[(k,n) for k,n,_ in df]
print("ALL_DECL_KINDS_IDENTICAL_IN_ORDER" if ko==kf else "DECL_KINDS_DIFFER: "+str(set(ko)^set(kf)))
