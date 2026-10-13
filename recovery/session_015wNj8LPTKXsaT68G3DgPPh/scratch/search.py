from fractions import Fraction as F
import itertools, math
k=F(1)
def F0(s,a,b,c,X,Z): return 2*k*X**3-F(3,2)*k*X-k/2+(s/2)*Z**2+(a/2)*(X**2-F(1,4))*Z+(b/2)*X*Z**2+(c/6)*Z**3
def gx(s,a,b,c,X,Z): return 6*k*X**2-F(3,2)*k+a*X*Z+(b/2)*Z**2
def gz(s,a,b,c,X,Z): return s*Z+(a/2)*(X**2-F(1,4))+b*X*Z+(c/2)*Z**2
def Q2(s,a,b,c):
    c2=a**3*c/4-3*a**2*b**2/16-F(9,2)*a*b*c*k+3*b**3*k+9*c**2*k**2
    c1=-a**2*b/4-3*a*c*k+6*b**2*k
    c0=3*s**2*b*k-(a*b/8-3*c*k/2)**2
    return c2,s*c1,c0
def isqrt_frac(q):
    if q<0: return None
    n,d=q.numerator,q.denominator
    rn,rd=math.isqrt(n),math.isqrt(d)
    return F(rn,rd) if rn*rn==n and rd*rd==d else None
found=[]
vals=[F(i,2) for i in range(-6,7)]
for X1 in [F(i,4) for i in range(-8,9)]:
    for Z1 in [F(i,2) for i in range(1,7)]:
        for a in vals:
            for c in vals:
                if Z1==0: continue
                b=-2*(6*X1**2-F(3,2)+a*X1*Z1)/Z1**2
                s=-((a/2)*(X1**2-F(1,4))+b*X1*Z1+(c/2)*Z1**2)/Z1
                # index/weight positivity at pins
                w1=-6*k*(s-b/2)-a*a/4; w2=a*a/4-6*k*(s+b/2)
                if not (w1>0 and w2>0): continue
                c2,c1s,c0=Q2(s,a,b,c)
                if c2==0: continue
                # X1 must be a root
                assert c2*X1*X1+c1s*X1+c0==0, (X1,Z1,a,c)
                X2=-c1s/c2-X1
                # common Z at X2: solve gx(X2,Z)=0 and gz(X2,Z)=0: gcd; try rational roots of gx quadratic in Z
                A2,A1,A0=b/2,a*X2,6*k*X2**2-F(3,2)*k
                disc=A1*A1-4*A2*A0 if A2!=0 else None
                cands=[]
                if A2==0:
                    if A1!=0: cands=[-A0/A1]
                else:
                    r_=isqrt_frac(disc)
                    if r_ is None: continue
                    cands=[(-A1+r_)/(2*A2),(-A1-r_)/(2*A2)]
                Zs=[z for z in cands if gz(s,a,b,c,X2,z)==0]
                if len(Zs)!=1: continue
                Z2=Zs[0]
                h1,h2=F0(s,a,b,c,X1,Z1),F0(s,a,b,c,X2,Z2)
                in1=(-k<h1<0); in2=(-k<h2<0)
                if (X2,Z2)==(X1,Z1) or X2 in (F(1,2),F(-1,2)): continue
                found.append(((in1,in2),(s,a,b,c),(X1,Z1,h1),(X2,Z2,h2),(w1,w2)))
kinds={}
for f in found: kinds.setdefault(f[0],[]).append(f)
for kk,lst in kinds.items():
    print(kk,len(lst))
    for f in lst[:3]: print('   jets(s,a,b,c)=',f[1],' P1=',f[2],' P2=',f[3],' weights',f[4])
