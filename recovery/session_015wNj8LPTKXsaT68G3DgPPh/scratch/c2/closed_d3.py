import sympy as sp, pickle, time
t0 = time.time()
r, b, k, l1, l2, s, t = sp.symbols('r b k l1 l2 s t', real=True)
D = pickle.load(open('a2_d3_raw.pkl', 'rb'))
F, q, Q = sp.sympify(D['F']), sp.sympify(D['q']), sp.sympify(D['Q'])
F = F.subs({sp.Symbol('r'): r, sp.Symbol('b'): b, sp.Symbol('k'): k, sp.Symbol('l1'): l1, sp.Symbol('l2'): l2})
q = q.subs({sp.Symbol('r'): r, sp.Symbol('b'): b, sp.Symbol('k'): k})
Q = Q.subs({sp.Symbol('r'): r, sp.Symbol('b'): b, sp.Symbol('k'): k, sp.Symbol('l1'): l1, sp.Symbol('l2'): l2})
detS = [sp.Rational(x) for x in D['detS']]; detA = [sp.Rational(x) for x in D['detA']]
def coeffs(e, n=6):
    e = sp.expand(e); return [e.coeff(r, i) for i in range(n + 1)]
qc, Qc = coeffs(q), coeffs(Q)
print('q0 =', sp.factor(qc[0]), '| Q0 =', sp.factor(Qc[0]))
print('q1, Q1 =', qc[1], Qc[1])
dS = sum(c * r**i for i, c in enumerate(detS[:7])); dA = sum(c * r**i for i, c in enumerate(detA[:7]))
pre = sp.series((dS / detS[0])**sp.Rational(-1, 2) * (dA / detA[0])**sp.Rational(-1, 2)
                * sp.exp(-(q - qc[0] + Q - Qc[0]) / 2), r, 0, 5).removeO()
G = sp.expand(sp.series(pre * F, r, 0, 5).removeO())
Gc = [G.coeff(r, i) for i in range(5)]
print('F leading orders:', [sp.expand(F).coeff(r, i) == 0 for i in range(3)], 'G r^3 zero:', sp.simplify(Gc[3]) == 0)
const = -12 * (2 * sp.pi)**-4 * detS[0]**sp.Rational(-1, 2) * sp.pi * (2 * sp.pi)**sp.Rational(-3, 2) * detA[0]**sp.Rational(-1, 2)
E0 = sp.exp(-(qc[0] + Qc[0]) / 2)
def coef(G, power):
    """(4pi/3) int db f.p. int dk k^power int_wedge const E0 G (l1 - l2) dl"""
    # k: E0 contains exp(-a k^2); write G as polynomial in k
    a = sp.expand(qc[0]).coeff(k, 2) / 2
    assert sp.expand(qc[0] - a * 2 * k**2).has(k) is False
    Pk = sp.Poly(sp.expand(G), k)
    rest = sp.exp(-(sp.expand(qc[0]).subs(k, 0) + Qc[0]) / 2)
    tot = 0
    for (j,), cj in Pk.terms():
        sx = j + power + 1
        Fj = sp.gamma(sp.Rational(sx) / 2) * a**(-sp.Rational(sx) / 2) / 2
        # b: Gaussian integral of cj * rest over R
        Ib = sp.integrate(sp.expand(cj * rest), (b, -sp.oo, sp.oo))
        Ib = sp.simplify(Ib)
        # wedge: l1 = -s, l2 = -s - t, s,t > 0, Jacobian 1, (l1 - l2) = t
        W = sp.expand(Ib.subs({l1: -s, l2: -s - t}) * t)
        tot += Fj * W
    return const * 4 * sp.pi / 3, sp.simplify(tot)
cst, wedge0 = coef(Gc[2], sp.Rational(-2, 3))
print('A0 wedge integrand ready', time.time() - t0)
pickle.dump({'const': cst, 'W0': wedge0}, open('d3_W0.pkl', 'wb'))
cst, wedge2 = coef(Gc[4], sp.Rational(-4, 3))
pickle.dump({'const': cst, 'W2': wedge2}, open('d3_W2.pkl', 'wb'))
import mpmath as mp
mp.mp.dps = 25
for nm, W in (('c', wedge0), ('c2', wedge2)):
    f = sp.lambdify((s, t), W, 'mpmath')
    val = mp.quad(lambda ss: mp.quad(lambda tt: f(ss, tt), [0, 2, 6, mp.inf]), [0, 2, 6, mp.inf])
    print(nm, '=', sp.N(cst) * val, time.time() - t0)
