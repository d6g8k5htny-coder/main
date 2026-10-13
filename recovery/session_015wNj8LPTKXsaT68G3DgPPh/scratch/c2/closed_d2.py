import sympy as sp, pickle
r, b, k = sp.symbols('r b k', real=True)
D = pickle.load(open('a2_d2.pkl', 'rb'))
def kfp(expr, power):
    """(2pi/3) f.p. int_0^oo int_R expr k^power db dk, expr = e^{-12k^2} * polynomial in k (coeffs in b)"""
    e = sp.expand(sp.simplify(expr * sp.exp(12 * k**2)))
    poly = sp.Poly(e, k)
    tot = 0
    for (j,), cj in poly.terms():
        s = j + power + 1           # int k^(s-1) e^{-12k^2} = Gamma(s/2) 12^(-s/2) / 2 (finite part for -2 < s < 0)
        Fj = sp.gamma(sp.Rational(s, 1) / 2) * sp.Integer(12) ** (-sp.Rational(s, 1) / 2) / 2
        Ib = sp.integrate(sp.expand(cj), (b, -sp.oo, sp.oo))
        tot += sp.simplify(Ib) * Fj
    return sp.simplify(2 * sp.pi / 3 * tot)
c = kfp(D['A0'], sp.Rational(-2, 3))
print('c  =', c, '=', sp.N(c, 20))
c2 = kfp(D['A2'], sp.Rational(-4, 3))
print('c2 =', c2, '=', sp.N(c2, 25))
pickle.dump({'c': c, 'c2': c2}, open('closed_d2.pkl', 'wb'))
