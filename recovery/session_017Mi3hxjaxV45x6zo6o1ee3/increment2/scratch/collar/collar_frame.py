"""Exploratory: degree-5 pin model, collar coordinates X = M + r(p,q).
Y = (f_x(X)/r^2, f_z(X)/r). Report coefficient rows for every random jet."""
from fractions import Fraction as F
import itertools, sys
sys.path.insert(0, '/home/user/Math-/reviews/pin_micro_covariance_nonauthor_20260928')
import cov_exact_check as ce

# extend variable table: reuse Poly with a fresh VARS tuple
VARS = ("r","p","q","k","S0","T","C","D","R4","E1","E2","E3","F4","R5","F41","F32","F23","F14","F05")
ce.VARS = VARS; ce.IDX = {v:i for i,v in enumerate(VARS)}
P = ce.Poly; V = ce.V
r,p,q,k = V("r"),V("p"),V("q"),V("k")
J = {n:V(n) for n in VARS[4:]}
S0,T,C,D,R4,E1,E2,E3,F4,R5,F41,F32,F23,F14,F05 = (J[n] for n in VARS[4:])
fxxx = 12*k - F(1,2)*r*R4 - F(3,20)*r**2*R5
fxx  = -6*k*r + F(1,12)*r**2*R4 + F(1,30)*r**3*R5
fxz  = -F(1,2)*r*T - F(1,6)*r**2*E1 - F(1,24)*r**3*F41
# 4th and 5th order symmetric tensors via monomial coefficients of f
# f jets: third order (fxxx,T,C,D); fourth (R4,E1,E2,E3,F4); fifth (R5,F41,F32,F23,F14,F05)
from math import comb
def taylor_grad(x, z):
    third = {(3,0):fxxx,(2,1):T,(1,2):C,(0,3):D}
    fourth= {(4,0):R4,(3,1):E1,(2,2):E2,(1,3):E3,(0,4):F4}
    fifth = {(5,0):R5,(4,1):F41,(3,2):F32,(2,3):F23,(1,4):F14,(0,5):F05}
    fx = fxx*x + fxz*z; fz = fxz*x + S0*z
    for order,tab in ((3,third),(4,fourth),(5,fifth)):
        fact = 1
        for i in range(2,order): fact *= i   # (order-1)!
        for (i,j),c in tab.items():
            # d/dx of c x^i z^j / (i! j!)  -> c x^(i-1) z^j/((i-1)! j!)
            from math import factorial as fa
            if i>=1: fx = fx + c*F(1,fa(i-1)*fa(j))*x**(i-1)*z**j
            if j>=1: fz = fz + c*F(1,fa(i)*fa(j-1))*x**i*z**(j-1)
    return fx, fz
def height(x,z):
    from math import factorial as fa
    tab={(2,0):fxx,(1,1):fxz,(0,2):S0,(3,0):fxxx,(2,1):T,(1,2):C,(0,3):D,(4,0):R4,(3,1):E1,(2,2):E2,(1,3):E3,(0,4):F4,
         (5,0):R5,(4,1):F41,(3,2):F32,(2,3):F23,(1,4):F14,(0,5):F05}
    out=P()
    for (i,j),c in tab.items(): out = out + c*F(1,fa(i)*fa(j))*x**i*z**j
    return out
z0=P()
fxS,fzS = taylor_grad(r, z0)
assert fxS.is_zero() and fzS.is_zero() and (height(r,z0)+k*r**3).is_zero(), "pins"
fx,fz = taylor_grad(r*p, r*q)
def div(poly, n):
    return P({tuple(m[i]-(n if VARS[i]=="r" else 0) for i in range(len(VARS))):c for m,c in poly.t.items()})
Y1, Y2 = div(fx,2), div(fz,1)
def show(poly):
    terms=[]
    for m,c in sorted(poly.t.items()):
        s='*'.join(f"{VARS[i]}^{e}" if e!=1 else VARS[i] for i,e in enumerate(m) if e)
        terms.append(f"{c}*{s}" if s else str(c))
    return ' + '.join(terms) if terms else '0'
print("mean Y1:", show(Y1.without(VARS[4:])))
print("mean Y2:", show(Y2.without(VARS[4:])))
for n in VARS[4:]:
    print(f"{n:4s} row1: {show(Y1.coeff(n))}\n     row2: {show(Y2.coeff(n))}")
