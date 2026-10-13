# Independent verification of P15-N1: is the error in PROOF.md (22) and line 340
# truly O(delta_eta) (i.e. O(rho^2)) rather than the stated O(sqrt(delta_eta)) = O(rho)?
import sympy as sp
from mpmath import mp, mpf, sqrt as msqrt, findroot, log as mlog

# ---------- Part A: exact symbolic series in rho ----------
rho, w = sp.symbols('rho w', real=True)
a_c = sp.Integer(3)**sp.Rational(-1, 4)
eq = sp.sqrt(3)*w**2 - rho*w**3 - 1          # PROOF.md line 324: the rescaled cubic

def series_root(w0, order=4):
    # Newton-style iteration for a root of eq near w0 as a power series in rho
    c = sp.symbols('c1:%d' % (order+1))
    wser = w0 + sum(c[i]*rho**(i+1) for i in range(order))
    expr = sp.expand(eq.subs(w, wser))
    sol = {}
    for k in range(1, order+1):
        coeff = sp.expand(expr.coeff(rho, k).subs(sol))
        ck = sp.solve(coeff, c[k-1])[0]
        sol[c[k-1]] = sp.nsimplify(sp.simplify(ck))
    return sp.expand(wser.subs(sol))

w0s = series_root(a_c)      # saddle-side root w_0(rho)
wms = series_root(-a_c)     # lower-maximum root w_-(rho)
print("w_0(rho) =", w0s)
print("w_-(rho) =", wms)
gap = sp.expand(w0s - wms)
print("w_0 - w_- =", gap)
gap_ratio = sp.series(gap/(2*a_c), rho, 0, 4).removeO()
print("(w_0-w_-)/(2 a_c) =", sp.simplify(gap_ratio))
print("  rho^1 coefficient:", sp.simplify(gap_ratio.coeff(rho, 1)), "   rho^2 coefficient:", sp.simplify(gap_ratio.coeff(rho, 2)), "=", sp.N(gap_ratio.coeff(rho, 2)))

# integral in (22): I(rho) = int_{w_-}^{w_0} [1 - sqrt3 w^2 + rho w^3] dw
F = sp.integrate(1 - sp.sqrt(3)*w**2 + rho*w**3, w)
I = sp.expand(F.subs(w, w0s) - F.subs(w, wms))
Iser = sp.series(I, rho, 0, 4).removeO()
K_c = sp.Rational(4, 3)*a_c
print("I(rho) series =", sp.simplify(Iser))
print("  I(0) == K_c ?", sp.simplify(Iser.coeff(rho, 0) - K_c) == 0, "  (K_c =", K_c, "=", sp.N(K_c), ")")
print("  rho^1 coefficient:", sp.simplify(Iser.coeff(rho, 1)))
print("  rho^2 coefficient:", sp.simplify(Iser.coeff(rho, 2)), "=", sp.N(Iser.coeff(rho, 2)))
print("  relative rho^2 coefficient I_2/K_c:", sp.simplify(Iser.coeff(rho, 2)/K_c), "=", sp.N(Iser.coeff(rho, 2)/K_c))

# ---------- Part B: high-precision numerics straight from the original cubic ----------
mp.dps = 60
eta_c = mpf(2)/(3*msqrt(3))
ac = mpf(3)**(-mpf(1)/4)
Kc = mpf(4)/mpf(3)**(mpf(5)/4)
def Phi(z, eta): return -z**4/4 + z**2/2 + eta*z
def two_roots(eta):
    # z_- in (-1, -1/sqrt3), z_0 in (-1/sqrt3, 0) of z^3 - z - eta = 0
    d = eta_c - eta
    zc = -1/msqrt(3)
    zm = findroot(lambda z: z**3 - z - eta, zc - ac*msqrt(d))
    z0 = findroot(lambda z: z**3 - z - eta, zc + ac*msqrt(d))
    assert -1 < zm < zc < z0 < 0
    return zm, z0
print("\nNumerics (mp.dps=60):")
prev = None
rows = []
for e in [2, 3, 4, 5, 6, 7, 8]:
    d = mpf(10)**(-e)
    zm, z0 = two_roots(eta_c - d)
    kap = Phi(zm, eta_c - d) - Phi(z0, eta_c - d)
    err_k = kap/d**(mpf(3)/2) - Kc
    err_g = (z0 - zm)/(2*ac*msqrt(d)) - 1
    rows.append((e, d, err_k, err_g))
    print("  delta=1e-%d  kappa/delta^1.5-K_c=%s  gap_ratio-1=%s  err_k/delta=%s  err_g/delta=%s"
          % (e, mp.nstr(err_k, 6), mp.nstr(err_g, 6), mp.nstr(err_k/d, 8), mp.nstr(err_g/d, 8)))
print("Log-log slopes between successive deltas (expect 1.0 for O(delta), 0.5 for O(sqrt delta)):")
for (e1, d1, k1, g1), (e2, d2, k2, g2) in zip(rows, rows[1:]):
    print("  1e-%d -> 1e-%d : slope_kappa=%s  slope_gap=%s" % (e1, e2, mp.nstr(mlog(k1/k2)/mlog(d1/d2), 8), mp.nstr(mlog(g1/g2)/mlog(d1/d2), 8)))
print("Hand-derived coefficients: I_2 = 3^(-3/4)/6 =", mp.nstr(mpf(3)**(-mpf(3)/4)/6, 8), ";  gap: 5 sqrt3/72 =", mp.nstr(5*msqrt(3)/72, 8))
