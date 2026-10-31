"""Exact check of the constrained det H_M in Math- reviews/d5_microdisk_20260926.

Source: Math- PR #82 (Grok lane), merged to Math- default 2026-09-27T17:35:56Z.
QUARTIC.md section 2 states det H_M = -12 lambda k P^2 r^3 / Q^2 after the
witness equations, and NOTE.md section 3 states det H_M = 0 identically.

This script takes the files' own expansions (NOTE.md section 2 plus the lambda
term of QUARTIC.md section 1, exact through r^4) and compares:
  (a) the files' route: substitute only the LEADING solve of grad f(X) = 0;
  (b) the exact witness event: solve grad f(X) = 0 through first order in r.

Recorded output (sympy, exact in all symbols), 2026-09-27:
    leading-solve det H_M = -12*P**2*k*lambda*r**3/Q**2
    exact-solve r^3 coeff of det H_M = 3*k*(-24*P**3*k + 3*P*Q**2*c + Q**3*d)/Q**2
    depends on lambda? False
    r^2 coeff = 0
So the lambda-dependence is an artefact of mixing orders: under the exact
solve lambda cancels at r^3, and the cubic alone already gives a generically
nonzero r^3 term, so "det H_M = 0 identically" holds only at leading order.
Exact arithmetic; certifying for these identities only. Requires sympy.

Run: python3 pr82_quartic_det_check.py
"""
import sympy as sp
r,k,c,d,P,Q,lam,A,q=sp.symbols('r k c d P Q lambda A q')
# Math- reviews/d5_microdisk_20260926/QUARTIC.md section 1 (exact through r^4), NOTE.md section 2
fx = r**3*(-6*k*P - q*Q/2) + r**4*(6*k*P**2 + P*q*Q + c*Q**2/2 + 2*lam*P)
fz = r**3*((A-c/2)*Q - q*P/2) + r**4*(q*P**2/2 + c*P*Q + d*Q**2/2)
HM = sp.Matrix([[-6*k*r + 2*lam*r**2, -q*r/2], [-q*r/2, r*(A-c/2)]])
det = sp.expand(HM.det())
# (a) QUARTIC's route: leading solve only
lead = {q: -12*k*P/Q, A: c/2 - 6*k*P**2/Q**2}
print('leading-solve det H_M =', sp.factor(sp.expand(det.subs(lead))))
# (b) exact witness solve of grad f(X)=0 to first order in r: q=q0+r q1, A=A0+r A1
q1,A1=sp.symbols('q1 A1')
sub={q: lead[q]+r*q1, A: lead[A]+r*A1}
eqs=[sp.expand(sp.series((fx/r**3).subs(sub),r,0,2).removeO().coeff(r,1)),
     sp.expand(sp.series((fz/r**3).subs(sub),r,0,2).removeO().coeff(r,1))]
sol=sp.solve(eqs,[q1,A1],dict=True)[0]
detx=sp.expand(det.subs(sub).subs(sol))
c3=sp.factor(sp.series(detx,r,0,4).removeO().coeff(r,3))
print('exact-solve r^3 coeff of det H_M =', c3)
print('depends on lambda?', sp.diff(c3,lam)!=0)
print('r^2 coeff =', sp.simplify(sp.series(detx,r,0,4).removeO().coeff(r,2)))
