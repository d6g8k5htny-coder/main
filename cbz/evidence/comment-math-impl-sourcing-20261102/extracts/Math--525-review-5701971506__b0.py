"""Independent exact-rational checks of the scalar identities used in SHARP_ACTION.md.
Polynomials are dicts {degree: Fraction}. No packet code is imported."""
from fractions import Fraction as F
import itertools, random

def pmul(a,b):
    out={}
    for i,x in a.items():
        for j,y in b.items():
            out[i+j]=out.get(i+j,0)+x*y
    return {k:v for k,v in out.items() if v!=0}
def padd(*ps):
    out={}
    for p in ps:
        for k,v in p.items(): out[k]=out.get(k,0)+v
    return {k:v for k,v in out.items() if v!=0}
def psc(p,c): return {k:v*c for k,v in p.items() if v*c!=0}
def pder(p,m=1):
    for _ in range(m):
        p={k-1:v*k for k,v in p.items() if k>=1}
    return p
def pev(p,x): return sum(v*x**k for k,v in p.items())

ok=True
def check(cond,label):
    global ok
    print(("PASS " if cond else "FAIL ")+label)
    ok = ok and cond

for (r,b,k) in [(F(1,3),F(2),F(5,7)),(F(1,10),F(-3,2),F(1)),(F(2,9),F(0),F(13,4))]:
    # cubic Hermite interpolant of phi(0)=rb, phi(r)=rb-k r^4, phi'(0)=phi'(r)=0
    p = padd({0:r*b}, psc({2:F(3)/r**2, 3:F(-2)/r**3}, -k*r**4))
    check(pev(p,0)==r*b and pev(p,r)==r*b-k*r**4 and pev(pder(p),0)==0 and pev(pder(p),r)==0,
          f"Hermite cubic pins r={r}")
    # (SA10) cubic part at 2r and remainder node polynomial at 2r
    node = pmul(pmul({1:1},{1:1}), pmul({0:-r,1:1},{0:-r,1:1}))  # t^2 (t-r)^2
    check(pev(p,2*r)-r*b==4*k*r**4, "(SA10) cubic Hermite value at 2r is rb+4kr^4")
    check(pev(node,2*r)/24==r**4/6, "(SA10) remainder coefficient r^4/6")
    # exactness of remainder formula for a quartic with constant 4th derivative c
    c=F(-31,3)
    phi = padd(p, psc(node, c/24))
    check(pev(pder(phi,4),0)==c and pev(phi,0)==r*b and pev(phi,r)==r*b-k*r**4
          and pev(pder(phi),0)==0 and pev(pder(phi),r)==0, "quartic model satisfies pins, D^4=c")
    # threshold: phi(2r)<=rb iff c<=-24k
    check((pev(phi,2*r)<=r*b)==(c<=-24*k), "phi(2r)<=rb iff c<=-24k (one c)")
    # optimality of witness point 2r: sup over t of required c: from bracket -3kr^2+2krt+(c/24)(t-r)^2<=0
    # need c/24 <= k(w^2-2w)... compute min over rational grid of (3kr^2-2krt)/(t-r)^2*24
    vals=[]
    for i in range(-400,401):
        t = r + r*F(i,100) if i!=0 else None
        if t is None or t==0: continue
        vals.append(((3*k*r**2-2*k*r*t)/(t-r)**2*24, t))
    m=min(vals)
    check(m[0]==-24*k and m[1]==2*r, f"tightest axial witness is t=2r with threshold -24k (grid min {m[0]} at t={m[1]})")
    # endpoint axial second derivatives (SA §5 lines 233): phi''(0)=r^2(c/12-6k), phi''(r)=r^2(c/12+6k)
    check(pev(pder(phi,2),0)==r**2*(c/12-6*k) and pev(pder(phi,2),r)==r**2*(c/12+6*k),
          "endpoint axial Hessians r^2(gamma-+6k), gamma=c/12")
    # targets v_r (f) and v_r^Y = r v_r (Y=r f)
    f = psc(phi, 1/r)  # unscaled f = Y/r
    U0=(pev(f,0)+pev(f,r))/2; U1=(pev(f,r)-pev(f,0))/r; U2=(pev(pder(f),r)-pev(pder(f),0))/r
    U3=F(6)/r**2*(pev(pder(f),0)+pev(pder(f),r)-2*U1)
    check((U0,U1,U2,U3)==(b-k*r**3/2,-k*r**2,0,12*k), "target v_r=(b-kr^3/2,-kr^2,0,12k)")
    U0y=(pev(phi,0)+pev(phi,r))/2; U1y=(pev(phi,r)-pev(phi,0))/r
    U2y=(pev(pder(phi),r)-pev(pder(phi),0))/r; U3y=F(6)/r**2*(pev(pder(phi),0)+pev(pder(phi),r)-2*U1y)
    check((U0y,U1y,U2y,U3y)==(r*b-k*r**4/2,-k*r**3,0,12*k*r), "(SA15) v_r^Y=(rb-kr^4/2,-kr^3,0,12kr)")

# U3 integral identity (6/r^3) int_0^r t(r-t) g'''(t) dt = (6/r^2)[g'(0)+g'(r)-2(g(r)-g(0))/r] for random quintics
random.seed(1)
for trial in range(5):
    g={i:F(random.randint(-9,9),random.randint(1,5)) for i in range(7)}
    r=F(random.randint(1,9),7)
    integrand = pmul({1:r,2:-1}, pder(g,3))
    anti = {k+1:v/(k+1) for k,v in integrand.items()}
    lhs = F(6)/r**3*(pev(anti,r)-pev(anti,0))
    rhs = F(6)/r**2*(pev(pder(g),0)+pev(pder(g),r)-2*(pev(g,r)-pev(g,0))/r)
    check(lhs==rhs, f"U3 kernel identity trial {trial}")
kern={1:F(1),2:F(-1)}; anti={k+1:v/(k+1) for k,v in kern.items()}
check(6*(pev(anti,1)-pev(anti,0))==1, "U3 kernel 6 t(1-t) has unit mass")

# Ridge correction (CW5): f(t,y)=phi(t)+(B/2)t(t-r)y+(A/2)y^2, A<0; max_y gives phi - B^2 t^2(t-r)^2/(8A)
for (A,B,c) in [(F(-2),F(3),F(-30)),(F(-1,2),F(1),F(-25))]:
    r=F(1,5);k=F(1);b=F(0)
    # effective 4th coefficient
    ceff = c - 3*B**2/A
    check(ceff == c + 3*B**2/(-A), "ridge-effective quartic c-3B^2/A")
    # max over y at t=2r
    t=2*r
    ystar = -(B/2)*t*(t-r)/A
    val = pev(padd(p:=padd({0:r*b}, psc({2:F(3)/r**2,3:F(-2)/r**3},-k*r**4)), psc(node:=pmul(pmul({1:1},{1:1}),pmul({0:-r,1:1},{0:-r,1:1})),c/24)),t) + (B/2)*t*(t-r)*ystar + A/2*ystar**2
    check(val - r*b == 4*k*r**4 + r**4/6*ceff, "ridge max at 2r equals 4kr^4+(r^4/6)(c-3B^2/A)")
    print("   example: c=",c," axial test c<=-24k:", c<=-24*k, " ridge test ceff<=-24k:", ceff<=-24*k)

# window: gamma_eps=-2k(1+eps) in (-6k,-2k) iff 0<eps<2 ; types gamma-6k<0, gamma+6k>0
for eps in [F(1,100),F(1),F(199,100),F(2),F(5,2)]:
    k=F(3,2); gam=-24*k*(1+eps)/12
    print(f"   eps={eps}: gamma in (-6k,-2k): {-6*k<gam<-2*k}; M soft {gam-6*k<0}; S soft {gam+6*k>0}")
print("ALL PASS" if ok else "SOME FAIL")
