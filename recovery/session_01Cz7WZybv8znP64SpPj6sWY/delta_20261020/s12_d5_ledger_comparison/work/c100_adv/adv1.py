# Adversarial re-derivation for C100 vs P2 (read-only support).
import sympy as sp, random, math
import mpmath as mp
out = {}
# --- 1. jet law: condition at M itself with the finite-r six pins, then r->0 (not the coalesced shortcut)
hx, hy = sp.symbols('hx hy', real=True)
K = sp.exp(-(hx**2+hy**2)/2)
def kder(n1, n2):
    return sp.diff(K, hx, n1, hy, n2)
r = sp.symbols('r', positive=True)
b, k = sp.symbols('b k', real=True)
# functional = (site_x, (dx,dy)) ; cov(D^al f(x1), D^be f(x2)) = (-1)^{|be|} D^{al+be} K (x1-x2)
def cov(F, G):
    (x1, al), (x2, be) = F, G
    e = kder(al[0]+be[0], al[1]+be[1]).subs({hx: x1-x2, hy: 0})
    return (-1)**(be[0]+be[1]) * e
M0, S0 = 0, r
obs = [(M0,(0,0)), (M0,(1,0)), (M0,(0,1)), (S0,(0,0)), (S0,(1,0)), (S0,(0,1))]
tgt = sp.Matrix([b, 0, 0, b-k*r**3, 0, 0])
free = [(M0,(0,2)), (M0,(2,1)), (M0,(1,2)), (M0,(0,3)), (M0,(4,0))]  # S_0, T_3, C_3, f_yyy, A_4 at M
S11 = sp.Matrix(6,6, lambda i,j: cov(obs[i], obs[j]))
S21 = sp.Matrix(5,6, lambda i,j: cov(free[i], obs[j]))
S22 = sp.Matrix(5,5, lambda i,j: cov(free[i], free[j]))
# series in r: use numeric small r rationally then limit
def condlaw(rv):
    s11 = S11.subs(r, rv); s21 = S21.subs(r, rv); s22 = S22.subs(r, rv)
    inv = s11.inv()
    return (s22 - s21*inv*s21.T).applyfunc(sp.nsimplify), (s21*inv*tgt.subs(r, rv))
for rv in [sp.Rational(1,10), sp.Rational(1,100), sp.Rational(1,1000)]:
    C, m = condlaw(rv)
    out[f'condcov_r={rv}'] = [[float(C[i,j]) for j in range(5)] for i in range(5)]
    out[f'condmean_r={rv}'] = [float(sp.N(mi.subs({b: sp.Rational(6,5), k: sp.Rational(1,6)}))) for mi in m]
for kk,v in out.items(): print(kk, v)
