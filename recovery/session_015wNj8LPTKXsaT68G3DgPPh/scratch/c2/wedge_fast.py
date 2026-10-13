import sympy as sp, pickle, time
from uint import I
t0 = time.time()
s, t = sp.symbols('s t', positive=True)
cache = {}


def Ic(j, m, a, b, c):
    key = (j, m, a, b, c)
    if key not in cache:
        cache[key] = I(j, m, a, b, c)
    return cache[key]


def closed(W):
    W = sp.expand(sp.powsimp(sp.expand(W)))
    groups = {}
    for term in sp.Add.make_args(W):
        ex = [f for f in sp.Mul.make_args(term) if f.func == sp.exp]
        key = sp.Mul(*ex)
        groups[key] = groups.get(key, 0) + term / key
    tot = 0
    for ex, poly in groups.items():
        Qf = -sp.expand(sp.log(ex).expand(force=True))
        al, be, ga = Qf.coeff(s, 2), sp.expand(Qf).coeff(s, 1).coeff(t, 1), Qf.coeff(t, 2)
        assert sp.expand(Qf - (al * s**2 + be * s * t + ga * t**2)) == 0, Qf
        P = sp.Poly(sp.expand(poly), s, t)
        for (i, j), c in P.terms():
            m = sp.Rational(i + j + 2, 2)
            tot += c * sp.gamma(m) / 2 * Ic(j, m, ga, be, al)
    return tot


for nm in ('W0', 'W2'):
    D = pickle.load(open('d3_%s.pkl' % nm, 'rb'))
    W = D[nm].subs({sp.Symbol('s', real=True): s, sp.Symbol('t', real=True): t})
    val = closed(W)
    full = sp.nsimplify(D['const']) * val
    full = sp.radsimp(sp.expand(full))
    print(nm, ':', full)
    print('   =', sp.N(full, 40), time.time() - t0, flush=True)
    pickle.dump(full, open('d3_%s_closed.pkl' % nm, 'wb'))
