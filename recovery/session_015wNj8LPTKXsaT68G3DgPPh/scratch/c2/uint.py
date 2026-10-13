import sympy as sp
ph = sp.Symbol('phi', real=True)
def I(j, m, a, b, c):
    """exact int_0^oo u^j (a u^2 + b u + c)^(-m) du, a, c > 0, 4ac - b^2 > 0"""
    a, b, c, m = sp.nsimplify(a), sp.nsimplify(b), sp.nsimplify(c), sp.nsimplify(m)
    D = 4 * a * c - b**2
    assert D > 0
    sD = sp.sqrt(D)
    u = sD / (2 * a) * sp.tan(ph) - b / (2 * a)
    # Q = D/(4a) sec^2, du = sD/(2a) sec^2 dphi ; integrand u^j cos^(2m-2) * const
    integrand = sp.expand(sp.expand_trig(u**j)) * sp.cos(ph)**(2 * m - 2) * (D / (4 * a))**(-m) * sD / (2 * a)
    integrand = sp.expand(sp.simplify(integrand))
    phi0 = sp.atan(b / sD)
    F = sp.integrate(integrand, ph)
    val = sp.limit(F, ph, sp.pi / 2, '-') - F.subs(ph, phi0)
    return sp.simplify(val)
if __name__ == '__main__':
    import mpmath as mp
    for (j, m) in ((3, sp.Rational(7, 2)), (1, sp.Rational(7, 2)), (2, sp.Rational(7, 2)), (2, 3), (0, 2)):
        v = I(j, m, 2, 3, 3)
        num = mp.quad(lambda x: x**j / (2 * x * x + 3 * x + 3) ** float(m), [0, mp.inf])
        print(j, m, v, sp.N(v, 20), num)
