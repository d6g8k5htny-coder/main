from fractions import Fraction as F
# Check: P20 subset [yhat-D, yhat+D] always; under model |yhat-C0w|<=T+D -> P20 in C0w*(1+-eta) when P22 holds.
import itertools
ok=True
for Lb,Ub,r,V,Q,H,qb,t in itertools.product([F(0),F(3),F(10)],[F(10),F(20),F(50)],[F(1),F(5)],[F(576)],[F(50)],[F(0),F(1,1000)],[F(0),F(1,500)],[F(0),F(1,300)]):
    if Lb>Ub: continue
    B=H+qb
    Am=max(F(0),max(F(0),Lb-r)/V-B); Ap=min(Q,Ub+r)/V+B+t
    yhat=(Lb+Ub)/(2*V); g=(Ub-Lb)/(2*V); D=H+qb+t+g+r/V
    ok &= (Am>=yhat-D) and (Ap<=yhat+D)
print("P20_within_yhat_pm_D:",ok)
# Gap witness for COMPATIBLE_ONLY/INCOMPLETE overlap: all P20 intersect P21, one bin rows unresolved [0,Q]
V=F(576);Q=F(50);r=F(2)
Am0,Ap0=F(0),Q/V   # unresolved bin: vacuous
P21=(F(1,1000),F(2,1000))
print("compatible_and_incomplete:", Am0<=P21[1] and Ap0>=P21[0])
