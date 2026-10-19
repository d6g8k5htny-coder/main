import re,sys
def defs(path):
    src=open(path).read()
    out={}
    # capture "def|abbrev|noncomputable def NAME ... :=" bodies up to next docstring "/--" or blank line followed by non-indented
    for m in re.finditer(r'^(?:noncomputable )?(?:def|abbrev) (\w+)(.*?)(?=^\s*$|^/--|^structure|^end )', src, re.S|re.M):
        out[m.group(1)]=re.sub(r'\s+',' ',(m.group(1)+m.group(2)).strip())
    return out
F=defs(sys.argv[1]); L=defs(sys.argv[2])
names=["E","latticeVec","IsSmooth","IsLPeriodic","hessian","IsCriticalPoint","HessianHasInertia","IsNondegMax","IsMorseDistinct","torusDist"]
for n in names:
    f=F.get(n); l=L.get(n)
    print(f"{n:18s} field={'Y' if f else 'N'} lifetime={'Y' if l else 'N'} identical={f==l}")
    if f!=l: print("  F:",f); print("  L:",l)
