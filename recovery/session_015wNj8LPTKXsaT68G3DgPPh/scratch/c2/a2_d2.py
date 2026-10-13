import sympy as sp, itertools, pickle, time
import pseries as P
from fractions import Fraction as Fr
t0 = time.time()
r, b, k, x = sp.symbols('r b k x', real=True)
NR = 7  # keep through r^NR
d = 2
X = lambda be, s: P.deriv_at(be, s, d)
forms = {'xxM': X((2,0),-1), 'xyM': X((1,1),-1), 'yyM': X((0,2),-1),
         'xxS': X((2,0),1), 'xyS': X((1,1),1), 'yyS': X((0,2),1), 'a': {(0,2): P.S.const(1)}}
pin = P.Pinned(d, forms)
def ser(s, upto=NR):
    return sum(sp.Rational(c.numerator, c.denominator) * r**i for i, c in enumerate(s.c[:upto + 1]))
def trunc(e, n=NR):
    e = sp.expand(e)
    return sum(e.coeff(r, i) * r**i for i in range(n + 1))
v = [b - k * r**3 / 2, -k * r**2, 0, 12 * k, 0, 0]
mean = {u: trunc(sum(ser(pin.G[u][j]) * v[j] for j in range(6))) for u in pin.names}
Cs = {key: ser(val) for key, val in pin.C.items()}
s2 = Cs[('a', 'a')]
beta = {u: trunc(sp.series(Cs[(u, 'a')] / s2, r, 0, NR + 1).removeO()) for u in pin.names if u != 'a'}
cp = {(u, w): trunc(Cs[(u, w)] - beta[u] * beta[w] * s2) for u in beta for w in beta}
terms = [(+1, ('xxM', 'yyM', 'xxS', 'yyS')), (-1, ('xxM', 'yyM', 'xyS', 'xyS')),
         (-1, ('xyM', 'xyM', 'xxS', 'yyS')), (+1, ('xyM', 'xyM', 'xyS', 'xyS'))]
def matchings(lst):
    if not lst:
        yield []; return
    f, rest = lst[0], lst[1:]
    for mm in matchings(rest):
        yield mm
    for i, y in enumerate(rest):
        for mm in matchings(rest[:i] + rest[i+1:]):
            yield [(f, y)] + mm
poly = 0
for sg, nm in terms:
    for mt in matchings(list(range(4))):
        used = set(i for p in mt for i in p)
        c = sp.Integer(1)
        for (i, j) in mt:
            c *= cp[(nm[i], nm[j])]
        for i in range(4):
            if i not in used:
                c *= (mean[nm[i]] + beta[nm[i]] * x)
        poly += sg * c
poly = trunc(poly)
print('poly built', time.time() - t0)
P_x = sp.Poly(poly, x)
# T_n = E[x^n 1{x < -m_a}], x ~ N(0, s2)
ma = mean['a']
z = sp.Symbol('z')
s = sp.sqrt(s2)
Phi = (1 + sp.erf(-ma / (s * sp.sqrt(2)))) / 2
phis = sp.exp(-ma**2 / (2 * s2)) / sp.sqrt(2 * sp.pi * s2)
T = [Phi, -s2 * phis]
for n in range(2, P_x.degree() + 1):
    T.append((n - 1) * s2 * T[n - 2] - s2 * (-ma)**(n - 1) * phis)
Z = sum(P_x.coeff_monomial(x**n) * T[n] for n in range(P_x.degree() + 1))
# pin density
q = sum(v[i] * ser(pin.Sinv[i][j]) * v[j] for i in range(6) for j in range(6))
pi_r = (2 * sp.pi)**-3 * ser(pin.detS)**sp.Rational(-1, 2) * sp.exp(-q / 2)
Ar = -12 * pi_r * Z / r**2
print('series...', time.time() - t0)
ser_A = sp.series(Ar, r, 0, 3).removeO()
A0 = sp.simplify(ser_A.coeff(r, 0))
A1 = sp.simplify(ser_A.coeff(r, 1))
A2 = sp.simplify(ser_A.coeff(r, 2))
print('A1 =', A1)
print('A0 =', A0)
print('A2 =', A2)
pickle.dump({'A0': A0, 'A2': A2}, open('a2_d2.pkl', 'wb'))
print('done', time.time() - t0)
