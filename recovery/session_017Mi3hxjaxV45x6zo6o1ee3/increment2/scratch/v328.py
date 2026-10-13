# Independent exact-rational verification of Math-#328 NOTE claims (reviewer 017Mi3hx). Standard library only.
from fractions import Fraction as F
import itertools, random
random.seed(20261006)
def moments(law):
    return [sum(w*x**j for x,w in law) for j in (2,4,6)]
def lean_scalars(s,m4,m6,t):
    # exact D2Schur.lean definitions (blob b95460c0)
    k4=m4-3*s*s; alpha=m4-2*k4*t; gamma=s*s+2*k4*t
    D=s*s*m4+k4*(m4+s*s)*t
    N=s*s*(m4*m4-s**4)
    S=alpha-(gamma**3+(2*gamma+alpha)*k4*k4*t*(1-4*t))/D
    Delta=m6-m4*m4/s
    T=(1-4*t)*Delta+4*t*(Delta+9*s*(m4-s*s))/4
    return D,N,S,T,Delta
def floors(p,q,a,b):
    u,v=a*a,b*b; A,B=p*u,q*v; s0=A+B
    g0=p*q*(u-v)**2/(p+q); d0=A*B*(u-v)**2/(A+B)
    return s0,g0,d0,min(s0*s0*(g0+s0*s0),g0*(g0+4*s0*s0)/4),min(g0,2*s0*s0),min(d0,(d0+9*s0*g0)/4)
ts=[F(i,80) for i in range(0,21)]  # 0 .. 1/4
vals=[F(n,d) for n in range(-7,8) for d in (1,2,3,4)]
checked=0; worst_ratio={"D":None,"S":None,"T":None}
for trial in range(4000):
    a=random.choice([x for x in vals if x!=0]); b=random.choice([x for x in vals if x!=0 and x*x!=a*a])
    p=F(random.randint(1,30),100); q=F(random.randint(1,30),100)
    # actual masses >= lower bounds, remainder spread on up to 3 arbitrary points (may coincide with atoms or 0)
    pa=p+F(random.randint(0,20),100); qb=q+F(random.randint(0,20),100)
    if pa+qb>1: continue
    rest=1-pa-qb; k=random.randint(0,3)
    law=[(a,pa),(b,qb)]
    if k==0 or rest==0:
        if rest>0: law.append((F(0),rest))
    else:
        ws=[F(random.randint(1,9)) for _ in range(k)]; tot=sum(ws)
        for w in ws: law.append((random.choice(vals),rest*w/tot))
    assert sum(w for _,w in law)==1
    s,m4,m6=moments(law); g=m4-s*s
    s0,g0,d0,Df,Sf,Tf=floors(p,q,a,b)
    assert s>=s0>0 and g>=g0>0, (law,s,s0,g,g0)
    # residual identity and Delta floor
    c=m4/s; resid=sum(w*(x**3-c*x)**2 for x,w in law)
    Delta=m6-m4*m4/s
    assert resid==Delta and Delta>=d0>0
    for t in ts:
        D,N,S,T,Dl=lean_scalars(s,m4,m6,t)
        assert D>0 and S==N/D and Dl==Delta
        assert D>=Df and S>=Sf and T>=Tf, (law,t)
        for key,val,fl in (("D",D,Df),("S",S,Sf),("T",T,Tf)):
            r=val/fl
            if worst_ratio[key] is None or r<worst_ratio[key]: worst_ratio[key]=r
    checked+=1
print("random laws checked:",checked,"| 21 directions each | min value/floor ratios:",{k:float(v) for k,v in worst_ratio.items()})
# sharpness witnesses
for a,b,p,q in itertools.product((F(1),F(-1,2),F(3)),(F(2),F(-3),F(1,3)),(F(1,10),F(1,3),F(1,2)),(F(1,4),F(1,2),F(1,5))):
    if p+q>1 or a*a==b*b: continue
    s0,g0,d0,*_=floors(p,q,a,b)
    s,m4,m6=moments([(a,p),(b,q),(F(0),1-p-q)])
    assert s==s0 and m6-m4*m4/s==d0, "Delta witness"
    z2=(p*a*a+q*b*b)/(p+q)   # gap witness: remainder at any point with square z2 (use the measure directly with X^2=z2)
    # even moments only depend on x^2, so place remainder on a symbolic point with x^2=z2:
    s2=p*a*a+q*b*b+(1-p-q)*z2; m4b=p*a**4+q*b**4+(1-p-q)*z2**2
    assert s2==z2 and m4b-s2*s2==g0, "gap witness"
print("sharpness witnesses: Delta and gap floors attained on all parameter combinations")
# section 6 example
s,m4,m6=moments([(F(1),F(1,4)),(F(2),F(1,4)),(F(0),F(1,2))])
print("§6:",s,m4,m6,"g=",m4-s*s,"Delta=",m6-m4*m4/s,"floors=",floors(F(1,4),F(1,4),F(1),F(2)))
# the two displayed inequalities of §5
for g,x in itertools.product([F(1,7),F(1),F(5)],[F(1,3),F(2),F(9)]):
    N=x*g*(g+2*x); D0=x*(g+x); D1=g*(g+4*x)/4
    assert N/D0==g*(g+2*x)/(g+x)>=g and N/D1==4*x*(g+2*x)/(g+4*x)>=2*x
print("§5 quotient inequalities hold on grid; D0=x(g+x) and D1=g(g+4x)/4 match d2_det_axis/d2_det_diagonal:",
      all(lean_scalars(s,m4,m6,F(0))[0]==s*s*m4 and lean_scalars(s,m4,m6,F(1,4))[0]==(m4-s*s)*(m4+3*s*s)/4 for s,m4,m6 in [(F(5,4),F(17,4),F(65,4)),(F(1),F(3),F(15))]))
# counterexamples: zero atom + one radius; same radius
s,m4,m6=moments([(F(0),F(1,2)),(F(1),F(1,4)),(F(-1),F(1,4))]); print("zero-atom control: g=",m4-s*s,"Delta=",m6-m4*m4/s)
# non-probability measure: unnormalized gap can be negative
s,m4,m6=moments([(F(1),F(1)),(F(2),F(1))]); print("mass-2 measure: m4-s^2 =",m4-s*s,"(negative as the NOTE warns)")
