import json,re
base='<SCRATCH>/repos/math_r5/specifications/lean_statement_proofs_20261011/SpecificationProofs/'
n=json.load(open('<SCRATCH>/verif_r5d/new.json'))
st=n['claims'][0]['statements']
for idx,mod,thm in ((72,'Side24Covariance.lean','referenceCovarianceLowerBound'),(73,'Side24ConeMoment.lean','coneMomentD2Identity')):
    src=open(base+mod).read()
    m=re.search(r'/--((?:(?!/--).)*?)-/\s*theorem '+thm,src,re.S)
    doc=m.group(1)
    f=doc.split('Does not establish:',1)[1]
    f=' '.join(f.split())
    e=st[idx]['does_not_claim']
    print(idx, f==e, len(f), len(e))
    if f!=e:
        import difflib
        for i,(a,b) in enumerate(zip(f,e)):
            if a!=b: print(repr(f[i-50:i+50]));print(repr(e[i-50:i+50]));break
    # also check the docstring starts with "Proves:" and contains no other fields after
    print(' fields:',re.findall(r'\n(\w[\w ]*):',doc))
