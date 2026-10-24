"""Finite-law exact check of P9-P12 composition: random finite outcome spaces with
nested F counts n_minus<=n<=n_plus, P count m sandwiched on good outcomes, arbitrary on bad."""
from fractions import Fraction as F
import random
random.seed(7)
V=F(576); bad_ok=good_ok=0; hits=0
for trial in range(4000):
    Q=random.randint(1,30); K=random.randint(2,12)
    outs=[]
    for _ in range(K):
        nm=random.randint(0,40); n=nm+random.randint(0,10); npl=n+random.randint(0,10)
        good=random.random()<0.85
        m=random.randint(nm,npl) if good else random.randint(0,200)
        p=F(random.randint(1,50))
        outs.append((p,nm,n,npl,m,good))
    tot=sum(o[0] for o in outs); outs=[(o[0]/tot,)+o[1:] for o in outs]
    E=lambda f: sum(o[0]*f(o) for o in outs)
    beta=E(lambda o: 0 if o[5] else 1)
    H=E(lambda o: o[3]-o[1])/V           # nu(S)=nu(I+)-nu(I-)
    muP=E(lambda o: min(o[4],Q))/V
    nu=E(lambda o: o[2])/V
    TQ=E(lambda o: max(0,o[2]-Q))/V
    S2=E(lambda o: o[2]**2)
    B=H+Q*beta/V
    assert abs(E(lambda o:min(o[2],Q))/V-muP)<=B, "P10"
    assert nu==E(lambda o:min(o[2],Q))/V+TQ, "P11"
    assert max(0,muP-B)<=nu<=muP+B+TQ, "P12"
    assert TQ<=S2/(4*Q*V), "P13"
    hits+=1
print("P10-P13 exact finite-law checks passed:",hits)
