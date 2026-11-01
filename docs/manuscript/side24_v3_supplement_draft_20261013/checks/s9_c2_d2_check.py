"""Corollary S9.16, d = 2: the value c_{2,ref} = c_2[phi] and a certified enclosure of c_{2,24}.

1. Symbolically (SymPy), for the Gaussian covariance phi(z) = exp(-|z|^2/2) on R^2, the quantities of
   Definition S4.2 in the frame (t, n) = (e_1, e_2) (phi is isotropic, so the integrand does not depend on t):
     Cov g_t = I_2, p_{g_t}(0) = 1/(2 pi);
     h_t = (f_tt, f_tn): Var 3 and 1, uncorrelated, so p_{h_t}(0) = 1/(2 pi sqrt 3);
     tau_t^2 = Var(f_ttt | g_t = 0) = 15 - 3^2/1 = 6;
     A_t = f_nn (m = 1): Var(f_nn | h_t = 0) = 3 - 1^2/3 = 8/3, so D_t = E[A_t^2 1{A_t<0} | h_t = 0] = 4/3.
   Then (correction-coef) with |S^1| = 2 pi gives c_2[phi] = Gamma(7/6)(3/2)^{1/3}(4/3)/(2 sqrt 3 pi^{3/2}),
   which equals Gamma(1/6) 12^{-1/6} / (9 pi^{3/2}), the closed form of Theorem S11.5 for d = 2.
2. In interval arithmetic (mpmath.iv, 300 digits) the closed form is enclosed, and Theorem S4.4,
   |c_{2,24}/c_{2,ref} - 1 - P_2(24) q| < 1e-216 with P_2(24) = -26045568, q = exp(-288), transfers the
   enclosure to c_{2,24}. The script asserts that the result lies inside the interval printed in Corollary S9.16.

Run from the supplement directory: python3 -B checks/s9_c2_d2_check.py
"""
import sympy as sp
from mpmath import iv, mp

# --- 1. symbolic moments of the Gaussian field with covariance phi on R^2 ---
x, y = sp.symbols('x y', real=True)
phi = sp.exp(-(x**2 + y**2) / 2)


def cov(a, b):
    """Cov(d^a f(0), d^b f(0)) = (-1)^{|b|} d^{a+b} phi(0) for multi-indices a=(ax,ay), b=(bx,by)."""
    ax, ay = a
    bx, by = b
    expr = sp.diff(phi, x, ax + bx, y, ay + by) if (ax + bx + ay + by) else phi
    return sp.simplify((-1) ** (bx + by) * expr.subs({x: 0, y: 0}))


T, N = (1, 0), (0, 1)          # d/dt = d/dx, d/dn = d/dy
g = [T, N]
h = [(2, 0), (1, 1)]
Sg = sp.Matrix(2, 2, lambda i, j: cov(g[i], g[j]))
Sh = sp.Matrix(2, 2, lambda i, j: cov(h[i], h[j]))
assert Sg == sp.eye(2), Sg
assert Sh == sp.diag(3, 1), Sh
zeta = (3, 0)
c_zg = sp.Matrix([[cov(zeta, gi) for gi in g]])
tau2 = sp.simplify(cov(zeta, zeta) - (c_zg * Sg.inv() * c_zg.T)[0])
assert tau2 == 6, tau2
A = (0, 2)
c_Ah = sp.Matrix([[cov(A, hi) for hi in h]])
varA = sp.simplify(cov(A, A) - (c_Ah * Sh.inv() * c_Ah.T)[0])
assert varA == sp.Rational(8, 3), varA
D = varA / 2                      # E[A^2 1{A<0}] for a centred normal A
assert D == sp.Rational(4, 3)
p_g = 1 / (2 * sp.pi * sp.sqrt(Sg.det()))
p_h = 1 / (2 * sp.pi * sp.sqrt(Sh.det()))
c2 = sp.gamma(sp.Rational(7, 6)) / (24 ** sp.Rational(1, 3) * sp.sqrt(sp.pi)) * 2 * sp.pi * p_g * p_h * tau2 ** sp.Rational(2, 3) * D
closed = sp.gamma(sp.Rational(1, 6)) * 12 ** sp.Rational(-1, 6) / (9 * sp.pi ** sp.Rational(3, 2))
assert sp.simplify(sp.nsimplify(c2 / closed) - 1) == 0 or abs(sp.N(c2 / closed - 1, 60)) < sp.Float('1e-55')
print('symbolic: Cov g = I, Cov h = diag(3,1), tau^2 = 6, Var(A|h) = 8/3, D = 4/3')
print('c_2[phi] =', c2, '=', closed)

# --- 2. certified enclosure ---
iv.dps = 300
mp.dps = 300
c_ref = iv.gamma(iv.mpf(1) / 6) * iv.power(iv.mpf(12), iv.mpf(-1) / 6) / (9 * iv.power(iv.pi, iv.mpf(3) / 2))
q = iv.exp(iv.mpf(-288))
P2 = iv.mpf(-26045568)
eps = iv.mpf([-1, 1]) * iv.mpf('1e-216')
c24 = c_ref * (1 + P2 * q + eps)
lo, hi = mp.mpf('0.07340691930603427103'), mp.mpf('0.07340691930603427104')
print('c_{2,ref} in [%s, %s]' % (mp.nstr(mp.mpf(c_ref.a.a), 40), mp.nstr(mp.mpf(c_ref.b.b), 40)))
print('c_{2,24}  in [%s, %s]' % (mp.nstr(mp.mpf(c24.a.a), 40), mp.nstr(mp.mpf(c24.b.b), 40)))
assert lo <= c24.a and c24.b <= hi, 'enclosure of Corollary S9.16 not certified'
assert c24.b < c_ref.a, 'c_{2,24} < c_{2,ref} expected (Theorem S4.4)'
print('RESULT: c_{2,24} in [0.07340691930603427103, 0.07340691930603427104] CERTIFIED')
