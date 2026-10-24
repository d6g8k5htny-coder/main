import mpmath as mp
from fractions import Fraction as Fr
import itertools
mp.mp.dps=30
def w(lo,hi): return mp.mpf(3)/2*(hi**(mp.mpf(2)/3)-lo**(mp.mpf(2)/3)) if hi>lo else mp.mpf(0)
def ratio(u): return (w(1-2*u,2+2*u)-w(1+2*u,2-2*u))/w(1,2)
print("slope", ratio(mp.mpf('1e-15'))/mp.mpf('1e-15'))
u1=mp.findroot(lambda u: ratio(u)-mp.mpf(1)/40, 0.003); u2=mp.findroot(lambda u: ratio(u)-mp.mpf(1)/20, 0.006)
print("P14 route a/delta >=",1/u1," X route a/delta >=",1/u2, " ell*/delta J=8:",256/u1+2,256/u2+2)
# exhaustive small check of reverse sandwich over all delta-matchings on tiny lattice
d=Fr(1,4); bad=0; tot=0
vals=[Fr(k,8) for k in range(1,25)]
for lf in vals:
  for sh in [Fr(k,8) for k in range(-4,5)]:
    lp=lf+sh
    for a in vals:
      for b in [a+Fr(k,8) for k in range(1,12)]:
        if not a>2*d: continue
        F=[lf]; P=[lp] if lp>0 else []
        if lp<=0: continue
        c=lambda xs,lo,hi: sum(1 for z in xs if lo<=z<hi) if lo<hi else 0
        tot+=1
        if not (c(P,a+2*d,b-2*d)<=c(F,a,b)<=c(P,a-2*d,b+2*d)): bad+=1
print("single matched bar exhaustive reverse-sandwich:",tot,"cases,",bad,"violations")
# counterexample to P20 coverage when P5 fails on shell: one F bar at lifetime a (in I), P partner at a-2delta (outside I and I^- ), deterministic
a=Fr(1);b=Fr(2);d=Fr(1,10)
F=[a]; P=[a-2*d]
c=lambda xs,lo,hi: sum(1 for z in xs if lo<=z<hi) if lo<hi else 0
print("nu(I)=",c(F,a,b)," mu_P(I)=",c(P,a,b)," N_P(I+)=",c(P,a-2*d,b+2*d), " N_P(I-)=",c(P,a+2*d,b-2*d))
