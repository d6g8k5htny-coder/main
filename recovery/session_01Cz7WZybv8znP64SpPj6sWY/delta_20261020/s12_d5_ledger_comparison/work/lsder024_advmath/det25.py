# Independent sympy check of LS-DER-024 (22)-(25): station elimination, eta^2 factors,
# the (c s)^-2 chart denominators in each determinant, and the sign structure on eta->0.
import sympy as sp
X, Y, eta, kap, a, Q, w, z, al, c, s = sp.symbols('X Y eta kappa a Q w z alpha c s', real=True)
P = kap*(X**3/3 - eta**2*X/4 - eta**3/12) + (a/2)*(X**2 - eta**2/4)*Y + (Q/2)*Y**2 + (w/2)*X*Y**2 + (z/6)*Y**3
eqs = [P.subs({X: c, Y: s}) + al*kap*eta**3/6, sp.diff(P, X).subs({X: c, Y: s}), sp.diff(P, Y).subs({X: c, Y: s})]
sol = sp.solve(eqs, [a, Q, z], dict=True)
assert len(sol) == 1
sol = sol[0]
print('a =', sp.factor(sol[a]))
print('Q =', sp.factor(sol[Q]))
H = sp.hessian(P, (X, Y))
pts = {'M': (-eta/2, 0), 'S': (eta/2, 0), 'Y': (c, s)}
dets = {}
for k, (x0, y0) in pts.items():
    d = sp.factor(sp.together(H.subs({X: x0, Y: y0}).det().subs(sol)))
    dets[k] = d
    num, den = sp.fraction(d)
    print(k, 'denominator =', sp.factor(den))
# use c^2+s^2=1 for the limit
for k, d in dets.items():
    lim = sp.factor(sp.simplify(sp.limit(d/eta**2, eta, 0).subs(s**2, 1 - c**2)))
    print(k, 'det/eta^2 at eta->0 =', lim)
# Q order in eta (should be O(eta^2) on the unit circle?)
Qser = sp.series(sol[Q].subs(s, sp.sqrt(1 - c**2)), eta, 0, 3).removeO()
print('Q series in eta:', sp.factor(sp.simplify(Qser)))
# Typed support: M negative definite requires det_M>0; on c^2+s^2=1 the eta^2 coefficients of det_M and det_S coincide
# and are <=0, so det_M>0 and det_S<0 need the common coefficient to vanish to O(eta): K=O(eta).
K = 2*(w/kap)*s**2 - 4*c**2 - 4*c*eta - eta**2
pref = kap**2*eta**2/(64*c**2*s**2)
ref = {'M': pref*(64*al*c**2*eta**2 - K**2), 'S': pref*(64*(1-al)*c**2*eta**2 - (K + 8*c*eta)**2),
       'Y': -pref*((K + 8*al*c*eta)**2 + 64*al*(1-al)*c**2*eta**2)}
vals = {c: sp.Rational(3, 5), s: sp.Rational(4, 5), eta: sp.Rational(1, 3), al: sp.Rational(2, 7), w: sp.Rational(5, 11), kap: sp.Rational(9, 10)}
for k in dets:
    print(k, 'matches LS-DER-033-form reference at a rational point on c^2+s^2=1:', sp.simplify((dets[k] - ref[k]).subs(vals)) == 0)
# With K = eta*kk (typed band), each determinant is O(eta^4):
kk = sp.symbols('kk', real=True)
for k in ref:
    e = sp.expand(ref[k].subs(w, kap*(4*c**2 + 4*c*eta + eta**2 + eta*kk)/(2*s**2)))
    print(k, 'on K=eta*kk: lowest eta power =', min(sp.Poly(sp.numer(sp.together(e)), eta).monoms())[0])
