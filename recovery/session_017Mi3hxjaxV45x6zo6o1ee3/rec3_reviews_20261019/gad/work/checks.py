from fractions import Fraction as Fr
import random
random.seed(20261019)
def det(M):
    M=[row[:] for row in M]; n=len(M); d=Fr(1)
    for c in range(n):
        p=next((r for r in range(c,n) if M[r][c]!=0),None)
        if p is None: return Fr(0)
        if p!=c: M[c],M[p]=M[p],M[c]; d=-d
        d*=M[c][c]
        for r in range(c+1,n):
            f=M[r][c]/M[c][c]
            for k in range(c,n): M[r][k]-=f*M[c][k]
    return d
# (A) |det(O->U_r)| = 12 r^-(d+3): axial 4x4 block in order (h_M,h_S,g_uM,g_uS), then (d-1) transverse 2x2 blocks
ok=True
for d in range(1,6):
    for r in [Fr(1,3),Fr(2,7),Fr(5,11),Fr(1,10)]:
        n=2*d+2; M=[[Fr(0)]*n for _ in range(n)]
        M[0][0]=Fr(1,2); M[0][1]=Fr(1,2)
        M[1][0]=-1/r; M[1][1]=1/r
        M[2][2]=-1/r; M[2][3]=1/r
        M[3][0]=12/r**3; M[3][1]=-12/r**3; M[3][2]=6/r**2; M[3][3]=6/r**2
        for i in range(d-1):
            a=4+2*i; M[a][a]=Fr(1,2); M[a][a+1]=Fr(1,2); M[a+1][a]=-1/r; M[a+1][a+1]=1/r
        D=abs(det(M)); ok&= (D==12/r**(d+3))
print("(A) |det| = 12 r^-(d+3) for d=1..5, 4 radii:", ok)
# (B) target v_r for h_M=b, h_S=b-k r^3, zero gradients
b,k,r=Fr(3,2),Fr(5,3),Fr(2,9)
hM,hS,guM,guS=b,b-k*r**3,Fr(0),Fr(0)
v=((hM+hS)/2,(hS-hM)/r,(guS-guM)/r,(6/r**2)*(guM+guS-2*(hS-hM)/r))
print("(B) v_r =",v==(b-k*r**3/2,-k*r**2,Fr(0),12*k))
# (C) fourth row = weight-one average of f''' with kernel (3/(4a^3))(a^2-s^2), r=2a; random integer polynomials deg<=7
def poly_eval(c,x): return sum(ci*x**i for i,ci in enumerate(c))
def deriv(c): return [i*c[i] for i in range(1,len(c))]
def integ_kernel(c3,a):  # ∫_{-a}^{a} (a^2-s^2) f'''(s) ds exactly
    tot=Fr(0)
    for i,ci in enumerate(c3):
        # ∫ (a^2 s^i - s^{i+2}) ds over [-a,a]
        def I(p): return Fr(0) if p%2==1 else 2*a**(p+1)/(p+1)
        tot+=ci*(a**2*I(i)-I(i+2))
    return tot
okC=True
for _ in range(200):
    c=[Fr(random.randint(-9,9)) for _ in range(8)]
    a=Fr(random.randint(1,9),random.randint(2,13)); r=2*a
    f=lambda x: poly_eval(c,x); f1=deriv(c); f3=deriv(deriv(f1))
    row4=(6/r**2)*(poly_eval(f1,a)+poly_eval(f1,-a)-2*(f(a)-f(-a))/r)
    okC&= row4==(Fr(3)/(4*a**3))*integ_kernel(f3,a)
print("(C) fourth row = (3/(4a^3)) ∫(a^2-s^2) f''' :", okC, "; kernel mass:", (Fr(3)/(4*Fr(1)**3))*(2*Fr(1)**3-2*Fr(1)**3/3))
# (D) cubic contact: f'(s)=6k(s^2-r^2/4) gives f(r/2)-f(-r/2)=-k r^3, f''(-r/2)/r=-6k, f''(r/2)/r=6k, f'''=12k
k,r=Fr(7,5),Fr(3,8)
F=lambda s: 6*k*(s**3/3-r**2*s/4)
print("(D) gap",F(r/2)-F(-r/2)==-k*r**3,"alphaM",(12*k*(-r/2))/r==-6*k,"alphaS",(12*k*(r/2))/r==6*k)
# (E) constants: 144*12^(-7/3) = 12^(-1/3); radial ledger exponent (d-1)+3-(d+3)+2 = 1; r^-2 dr -> (1/3) l^(-1/3) k^(-2/3) dk
print("(E) 144*12^(-7/3) vs 12^(-1/3):", abs(144*12**(-7/3)-12**(-1/3))<1e-15, "; ledger exponents:", {d:(d-1)+3-(d+3)+2 for d in range(1,7)})
import math
l,kk=0.37,2.9; rr=(l/kk)**(1/3); drdk=-(1/3)*l**(1/3)*kk**(-4/3)
print("    r^-2 |dr/dk| =", rr**-2*abs(drdk), " vs (1/3) l^(-1/3) k^(-2/3) =", (1/3)*l**(-1/3)*kk**(-2/3))
# (F) scaling: F->aF gives C->a^(-2/3); metric lambda^2 gives density product lambda^(3d-4-4(d-1)) = lambda^(-d)
print("(F) amplitude exponent:", {d:(-2*d)+(4/3)+2*(d-1) for d in range(1,5)}, "; metric exponent:", {d:3*d-4-4*(d-1) for d in range(1,5)})
