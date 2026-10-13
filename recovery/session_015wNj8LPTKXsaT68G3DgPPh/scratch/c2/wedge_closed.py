import sympy as sp, pickle, time
t0 = time.time()
s, t, u = sp.symbols('s t u', positive=True)
def closed(W):
    W = sp.expand(sp.powsimp(sp.expand(W)))
    # group terms by exponential factor
    groups = {}
    for term in sp.Add.make_args(W):
        ex = [f for f in sp.Mul.make_args(term) if f.func == sp.exp]
        key = sp.Mul(*ex)
        groups[key] = groups.get(key, 0) + term / key
    tot = 0
    for ex, poly in groups.items():
        arg = sp.expand(sp.log(ex).expand(force=True)) if ex != 1 else 0
        Qf = -arg
        al, be, ga = Qf.coeff(s, 2), sp.expand(Qf).coeff(s, 1).coeff(t, 1), Qf.coeff(t, 2)
        assert sp.simplify(Qf - (al * s**2 + be * s * t + ga * t**2)) == 0, Qf
        P = sp.Poly(sp.expand(poly), s, t)
        for (i, j), c in P.terms():
            m = sp.Rational(i + j + 2, 2)
            I = sp.integrate(u**j / (al + be * u + ga * u**2)**m, (u, 0, sp.oo))
            tot += c * sp.gamma(m) / 2 * I
    return sp.simplify(tot)
for nm in ('W0', 'W2'):
    D = pickle.load(open('d3_%s.pkl' % nm, 'rb'))
    W = D[nm].subs({sp.Symbol('s', real=True): s, sp.Symbol('t', real=True): t})
    val = closed(W)
    full = sp.nsimplify(1) * D['const'] * val
    full = sp.simplify(full)
    print(nm, ':', full, '=', sp.N(full, 30), time.time() - t0)
    pickle.dump(full, open('d3_%s_closed.pkl' % nm, 'wb'))
