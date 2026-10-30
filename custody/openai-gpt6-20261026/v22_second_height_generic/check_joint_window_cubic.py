#!/usr/bin/env python3
"""Exact polynomial checks for V22 second-height candidate, not scientific review.
Requires sympy; no Gaussian simulation, field samples, or imported project code.
"""
import sympy as S

q, eta, s, k, b = S.symbols("q eta s k b", nonzero=True)
c, t, a, d, u, v = S.symbols("c t a d u v", real=True)
C02, C12, C21, C03 = S.symbols("C02 C12 C21 C03")
D = c*d - a*t
r = q*s
P = (b-k*r**3/2) - S.Rational(3,2)*k*r**2*s*u + 2*k*s**3*u**3
P += -C21*q**2*v/4 + C02*v**2 + C21*u**2*v + C12*u*v**2 + C03*v**3
P_u, P_v = S.diff(P,u), S.diff(P,v)
z = {u:c, v:t}
zy = {u:c+a*eta, v:t+d*eta}
rows = [
    P_u.subs(z), P_v.subs(z),
    S.cancel((P_u.subs(zy)-P_u.subs(z))/eta),
    S.cancel((P_v.subs(zy)-P_v.subs(z))/eta),
]
sol = S.solve(rows, [C02, C12, C21, C03], dict=True)
assert len(sol)==1
P = P.subs(sol[0])
Pi = -k*s**3*(2*D+d*q)**2 * (2*q*t+eta*(2*D+d*q)) * (2*q*t-eta*(2*D-3*d*q)) / (32*d*D*(t+d*eta)**2)
Delta = k*s**3*eta**3*(2*t+d*eta)*(2*D+d*q)**2*(2*D-d*q)**2 / (32*t**2*D*(t+d*eta)**2)
assert S.factor(S.together(P.subs(z)-b-Pi)) == 0
assert S.factor(S.together(P.subs(zy)-P.subs(z)-Delta)) == 0
eta_X = -2*q*t/(2*D+d*q)
eta_Y = 2*q*t/(2*D-3*d*q)
R_Y = (2*D+d*q)**2*(D-d*q)/(D*(2*D-3*d*q)**2)
assert S.factor(S.together(Delta.subs(eta,eta_X)+k*r**3)) == 0
assert S.factor(S.together(Delta.subs(eta,eta_Y)-k*r**3*R_Y)) == 0
assert S.factor(S.together(S.limit(Delta/(k*s**3*eta**3),q,0).subs(eta,0)-(D/t)**3)) == 0
# Exact rational evaluations of direct cubic solution in three generic configurations
cases = [
    {c:1,t:1,a:0,d:1,q:S.Rational(1,10),eta:S.Rational(1,12),s:S.Rational(1,10),k:2,b:1},
    {c:1,t:-1,a:S.Rational(3,5),d:S.Rational(4,5),q:S.Rational(1,13),eta:S.Rational(1,8),s:S.Rational(1,10),k:2,b:1},
    {c:-1,t:S.Rational(3,2),a:S.Rational(5,13),d:S.Rational(12,13),q:S.Rational(1,12),eta:S.Rational(1,9),s:S.Rational(1,10),k:2,b:1},
]
for case in cases:
    assert S.factor(S.together((P.subs(zy)-P.subs(z)-Delta).subs(case))) == 0
print("PASS: exact cubic Pi, Delta, both root identities, contact limit; 3 rational configurations")
