#!/usr/bin/env python3
"""PR82 finite-witness counterexample, exact Fraction arithmetic, stdlib only.
Independent reproduction of NOTE/QUARTIC formulas; no Gaussian rate claim.
Fix k=P=Q=1, c=d=0. q,A are solved from BOTH finite-r witness equations.
"""
from fractions import Fraction as F

def require(p, message):
    if not p:
        raise SystemExit(message)

def field(r, lam, q, A, x, z):
    u, v = x/r, z/r
    cubic = 2*u**3-F(3,2)*u-F(1,2) + q*F(1,2)*(u*u-F(1,4))*v + A*v*v/2
    f = r**3*cubic + lam*(x*x-r*r/4)**2
    fx = r*r*(6*u*u-F(3,2)+q*u*v) + 4*lam*x*(x*x-r*r/4)
    fz = r*r*(q*(u*u-F(1,4))/2+A*v)
    hxx = 12*r*u+lam*(12*x*x-r*r)
    hxz = r*q*u
    hzz = r*A
    return f, fx, fz, hxx*hzz-hxz*hxz

for r in (F(1,100), F(1,1000)):
    for lam in (F(0), F(1)):
        # Exact linear solve, not substitution of the r=0 solve.
        q = (-12+12*r+4*lam*r-12*lam*r*r+8*lam*r**3)/(1-2*r)
        A = q*(1-r)/2
        M = field(r,lam,q,A,-r/2,F(0))
        S = field(r,lam,q,A,r/2,F(0))
        X = field(r,lam,q,A,-r/2+r*r,r*r)
        require(M[:3] == (0,0,0), 'M pins')
        require(S[:3] == (-r**3,0,0), 'S pins')
        require(X[1:3] == (0,0), 'finite-r witness equations')
        require(M[3] != 0, 'nonzero constrained determinant')
        if lam == 0:
            require(M[3]/r**3 == -72*(1-r)**2/(1-2*r)**2, 'cubic identity')
        advertised = -12*lam*r**3
        require(M[3] != advertised, 'distinguish leading-only substitution')
        lead_X = field(r,lam,F(-12),F(-6),-r/2+r*r,r*r)
        require(lead_X[1:3] != (0,0), 'leading solve is not finite witness')
        print('r=%s lambda=%s: six pins and witness exact; det(H_M)/r^3=%s; leading-only fx/r^4=%s fz/r^4=%s'
              % (r,lam,M[3]/r**3,lead_X[1]/r**4,lead_X[2]/r**4))

# General first-order coefficient cancellation. q=q0+r*q1+O(r^2),
# a=A-c/2=a0+r*a1+O(r^2), at fixed P,Q with Q != 0.
for P,Q,k,c,d,lam in ((F(1),F(1),F(1),F(0),F(0),F(1)),
                      (F(2,3),F(-3,5),F(7,6),F(4,9),F(-2),F(5))):
    q0=-12*k*P/Q
    a0=-6*k*P*P/(Q*Q)
    q1=-12*k*P*P/Q+c*Q+4*lam*P/Q
    a1=-(c*P+d*Q)/2+2*lam*P*P/(Q*Q)
    constant=-6*k*a0-q0*q0/4
    coefficient=-6*k*a1+2*lam*a0-q0*q1/2
    target=9*k*c*P+3*k*d*Q-72*k*k*P**3/(Q*Q)
    require(constant==0 and coefficient==target, 'first order identity')
print('PASS: finite cubic det nonzero; quartic lambda cancels at order r^3 under full witness constraints')
