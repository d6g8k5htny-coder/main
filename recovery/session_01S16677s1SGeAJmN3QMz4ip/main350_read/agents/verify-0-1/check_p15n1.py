# Independent check of P15-N1: order of the error term in PROOF.md (22) and line 340.
# Cubic (21): Phi_eta'(z_c+H) = sqrt3 H^2 - H^3 - delta, rho = sqrt(delta), H = rho w,
# roots of sqrt3 w^2 - rho w^3 - 1 = 0 near +-a_c, a_c = 3^(-1/4).
from mpmath import mp, mpf, sqrt, polyroots, quad, acos, cos, pi, findroot
import sympy as sp
mp.dps = 50
eta_c = mpf(2)/(3*sqrt(3)); a_c = mpf(3)**(-mpf(1)/4); K_c = mpf(4)/mpf(3)**(mpf(5)/4)
z_c = -1/sqrt(3)

def roots(eta):  # exact trig roots of z^3 - z - eta = 0 (Phi_eta' = -z^3+z+eta), Newton-polished
    th = acos(eta/eta_c)/3
    rs = sorted([(2/sqrt(3))*cos(th - 2*pi*k/3) for k in range(3)])
    return [findroot(lambda z: z**3 - z - eta, r) for r in rs]
Phi = lambda z, eta: -z**4/4 + z**2/2 + eta*z
def kappa(eta):
    zm, z0, zp = roots(eta); return Phi(zm, eta) - Phi(z0, eta)

# --- analytic prediction: I(rho) = int_{w_-}^{w_0} (1 - sqrt3 w^2 + rho w^3) dw, kappa = delta^{3/2} I(rho)
# I'(rho) = int w^3 dw (boundary terms vanish since integrand vanishes at the roots); I'(0) = int_{-a_c}^{a_c} w^3 = 0.
# I''(0) = w_0^3 w_0' - w_-^3 w_-' at rho=0, with w' = w^2/(2 sqrt3) = 1/6 for both roots  => I''(0) = a_c^3/3.
# => kappa/delta^{3/2} - K_c = (a_c^3/6) delta + O(delta^2)   (I is even in rho, so no odd powers)
rho = sp.symbols('rho', positive=True)
w = sp.symbols('w')
ac = sp.Integer(3)**sp.Rational(-1,4)
# series of the two roots of sqrt3 w^2 - rho w^3 - 1 = 0 near +-a_c, by iterated Newton in symbolic series
def root_series(w0, order=5):
    ws = sp.Integer(0) + w0
    for _ in range(order):
        f = sp.sqrt(3)*ws**2 - rho*ws**3 - 1
        fp = 2*sp.sqrt(3)*ws - 3*rho*ws**2
        ws = sp.series(ws - f/fp, rho, 0, order).removeO()
    return sp.expand(sp.series(ws, rho, 0, order).removeO())
wm = root_series(-ac); w0s = root_series(ac)
I = sp.integrate(1 - sp.sqrt(3)*w**2 + rho*w**3, (w, wm, w0s))
I_ser = sp.series(I, rho, 0, 4).removeO()
gap_ser = sp.series(w0s - wm, rho, 0, 4).removeO()
print("sympy: w_-(rho) =", sp.nsimplify(wm), "\n       w_0(rho) =", sp.nsimplify(w0s))
print("sympy: I(rho) series =", sp.simplify(I_ser), "  [K_c =", sp.nsimplify(sp.Rational(4,3)*ac), "]")
print("sympy: (w_0 - w_-)(rho) series =", sp.simplify(gap_ser))
c_kappa = sp.N(sp.simplify(I_ser).coeff(rho, 2), 20); c_gap = sp.N(sp.simplify(gap_ser/(2*ac)).coeff(rho, 2), 20)
print("predicted: kappa/delta^(3/2) - K_c ~ %s * delta ;  (z_0-z_-)/(2 a_c sqrt(delta)) - 1 ~ %s * delta" % (c_kappa, c_gap))
print("rho^1 coefficient of I:", sp.simplify(I_ser).coeff(rho, 1), " ; rho^1 coefficient of gap:", sp.simplify(gap_ser).coeff(rho, 1))

print("\nnumeric (exact roots, 50 digits):")
print("  delta      (kappa/d^1.5 - K_c)/d     (gap/(2a_c sqrt d) - 1)/d")
for e in [2, 4, 6, 8, 10]:
    d = mpf(10)**(-e); eta = eta_c - d
    zm, z0, zp = roots(eta)
    r1 = (kappa(eta)/d**(mpf(3)/2) - K_c)/d
    r2 = ((z0 - zm)/(2*a_c*sqrt(d)) - 1)/d
    print("  1e-%-3d  %s   %s" % (e, mp.nstr(r1, 12), mp.nstr(r2, 12)))
print("  => both ratios converge to constants, i.e. error is Theta(delta), not Theta(sqrt(delta)).")
print("  (If the error were genuinely of order sqrt(delta), the columns above would blow up like delta^(-1/2).)")

# Does anything downstream need the sharper rate?  (23): lim h^(1/3) q(h); check it only needs the limits.
Q_c = (mpf(2)/3)*K_c**(-mpf(2)/3)
for e in [6, 10, 14]:
    d = mpf(10)**(-e); eta = eta_c - d; zm, z0, zp = roots(eta); h = kappa(eta)
    print("  (23) delta=1e-%d: h^(1/3)/(z_0-z_-) = %s  (Q_c = %s)" % (e, mp.nstr(h**(mpf(1)/3)/(z0-zm), 15), mp.nstr(Q_c, 15)))
