# Exact-rational checks for F1 (SUPPORT circularity). Synthetic symbols only.
from fractions import Fraction as F
import random
random.seed(7)
def rnd(lo,hi,den=1009): return F(random.randint(int(lo*den),int(hi*den)),den)
# w(a,b) = 3/2 (b^{2/3}-a^{2/3}); use a=s^3 params so cube roots exact: a=x^3 -> a^{2/3}=x^2
def w(x,y): return F(3,2)*(y*y-x*x)   # x=a^{1/3}, y=b^{1/3}
ok={"tube_from_P5_on_Ipm_P14_P22":True,"null_SUPPORT_on_coverage":True,"ratio_dyadic":True,"P21_in_open_tube":True}
n1=n2=0
for _ in range(200000):
    # cube-root parametrisation of I=[a,b), I^-=[a+2d,b-2d), I^+=[a-2d,b+2d)
    x=rnd(1,2); y=x+rnd(0.05,1)
    a=x**3; b=y**3
    # choose cube-root-exact expanded/contracted endpoints approximately: pick xm<x<xp etc
    xm=x-rnd(0,0.02); xp=x+rnd(0,0.02); ym=y-rnd(0,0.02); yp=y+rnd(0,0.02)
    if not (xp<ym): continue
    wI=w(x,y); wP=w(xm,yp); wM=w(xp,ym)
    Cm=rnd(0.5,1.5); Cp=Cm+rnd(0,0.05); C=Cm+(Cp-Cm)*rnd(0,1); C0=(Cm+Cp)/2; uC=(Cp-Cm)/2
    RI=rnd(0,0.02); RP=rnd(0,0.02); RM=rnd(0,0.02)
    H=Cp*(wP-wM)+RP+RM                       # P14
    other=rnd(1e-4,0.05)                    # Q beta/V + t + g + r/V >0
    D=H+other; T=uC*wI+RI; eta=rnd(0,1)
    if not (T+2*D<=eta*C0*wI): continue
    n1+=1
    # any nu with P5 on I^+ and I^- (true C) and monotone sandwich nu(I^-)<=nu(I)<=nu(I^+)
    lo=C*wM-RM; hi=C*wP+RP
    if not ((1-eta)*C0*wI < lo and hi < (1+eta)*C0*wI): ok["tube_from_P5_on_Ipm_P14_P22"]=False
    lo21=max(F(0),Cm*wI-RI); hi21=Cp*wI+RI
    if not ((1-eta)*C0*wI<lo21 and hi21<(1+eta)*C0*wI): ok["P21_in_open_tube"]=False
    # under null + coverage: |yhat-C0 w|<=T+D, P20 within yhat+-D -> P20 inside closed tube
    nu=C*wI+RI*rnd(-1,1); yhat=nu+D*rnd(-1,1)
    if not (abs(yhat-C0*wI)<=T+D and yhat-D>=(1-eta)*C0*wI and yhat+D<=(1+eta)*C0*wI): ok["null_SUPPORT_on_coverage"]=False
# dyadic ratio: w([2t,4t))/w([t,2t)) = 2^{2/3}; check with t=u^3: (4^{2/3}-2^{2/3})/(2^{2/3}-1) = 2^{2/3}
import sympy as sp
r=sp.nsimplify((sp.Integer(4)**sp.Rational(2,3)-sp.Integer(2)**sp.Rational(2,3))/(sp.Integer(2)**sp.Rational(2,3)-1))
ok["ratio_dyadic"]= sp.simplify(r-sp.Integer(2)**sp.Rational(2,3))==0
print("admitted samples",n1)
for k,v in ok.items(): print(k,"PASS" if v else "FAIL")
