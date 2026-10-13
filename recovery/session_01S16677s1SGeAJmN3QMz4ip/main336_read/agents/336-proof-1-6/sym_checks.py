# Symbolic re-derivation of displayed identities in PROOF.md sections 1-6
import sympy as sp
from sympy import sqrt, Rational as Q

print("=== Sec 1: eta bound (1) is the three-real-root condition of -z^3+z+eta ===")
z, eta, lam = sp.symbols('z eta lambda', real=True)
f = -z**3 + z + eta
disc = sp.discriminant(f, z)
print("discriminant(-z^3+z+eta) =", sp.factor(disc), " >0 iff eta^2 < 4/27, i.e. eta < 2/(3 sqrt 3) =", sp.nsimplify(2/(3*sqrt(3))), float(2/(3*sqrt(3))))
assert sp.simplify(disc - (4 - 27*eta**2)) == 0

print("=== Sec 1 (2): CDF of ell = C lambda^2 under density gamma lam^(gamma-1)/lam0^gamma ===")
gam, lam0, C, t = sp.symbols('gamma lambda_0 C t', positive=True)
cdf_lam = lambda v: (v/lam0)**gam
P_ell_le_t = cdf_lam(sqrt(t/C))
ell0 = C*lam0**2
print("P(ell<=t) =", sp.simplify(P_ell_le_t), " vs (t/ell0)^(gamma/2) =", sp.simplify((t/ell0)**(gam/2)))
assert sp.simplify(sp.powsimp(P_ell_le_t - (t/ell0)**(gam/2), force=True)) == 0
nu = sp.diff((t/ell0)**(gam/2), t)
print("nu_Z(t) =", sp.simplify(nu), " target (gamma/2) ell0^(-gamma/2) t^(gamma/2-1)")
assert sp.simplify(nu - (gam/2)*ell0**(-gam/2)*t**(gam/2-1)) == 0

print("=== Sec 2 (5),(6): seed and exact chart identity ===")
d = 3
L = sp.symbols('L', positive=True)
om = 2*sp.pi/L
ys = sp.symbols('y1:%d' % (d+1), real=True)
w = [Q(1, 3**i) for i in range(1, d+1)]
h = sum(wi*sp.cos(om*yi) for wi, yi in zip(w, ys))
b = sum(w)
zs = [sqrt(2*wi)*sp.sin(om*yi/2) for wi, yi in zip(w, ys)]
print("h - (b - |z|^2) =", sp.simplify(h - (b - sum(zi**2 for zi in zs))))
assert sp.simplify(h - (b - sum(zi**2 for zi in zs))) == 0
# Hessian at critical points y_i in {0, L/2}
H = sp.hessian(h, ys)
print("Hessian diag entries:", [sp.simplify(H[i, i]) for i in range(d)], "off-diag zero:", all(sp.simplify(H[i, j]) == 0 for i in range(d) for j in range(d) if i != j))
# distinctness and gap: second largest value is b - 2 w_d
import itertools
vals = sorted({sum(s*wi for s, wi in zip(signs, w)) for signs in itertools.product([1, -1], repeat=d)}, reverse=True)
print("2^d=%d distinct values? %s ; b=%s ; second=%s ; b-Delta=%s" % (2**d, len(vals) == 2**d, b, vals[1], b - 2*w[-1]))
assert len(vals) == 2**d and vals[1] == b - 2*w[-1]
# 2 sum_{j>i} w_j < w_i
for i in range(d):
    assert 2*sum(w[i+1:]) < w[i]
print("2*sum_{j>i} w_j < w_i for all i: OK")
# chart differential positive on |y_i|<L/4
print("dz_i/dy_i =", sp.simplify(sp.diff(zs[0], ys[0])), " >0 for |y|<L/4 (cos(pi/4)>0)")

print("=== Sec 3 (8),(9),(10): surgery gradient and local normal form ===")
s = sp.symbols('s', real=True)
tt = sp.symbols('t1:%d' % d, real=True)
v = s**2 + sum(ti**2 for ti in tt)
c = sp.Function('c')
F0 = b - s**2*c(v) - sum(ti**2 for ti in tt)
Fs = sp.diff(F0, s)
print("(F0)_s =", sp.factor(Fs), "  target -2s[c(v)+s^2 c'(v)]")
vv = sp.symbols('vv', positive=True)
cprime = sp.Function('cprime')
Fs_sub = Fs.replace(lambda e: isinstance(e, sp.Subs), lambda e: cprime(v))
assert sp.simplify(Fs_sub - (-2*s*(c(v) + s**2*cprime(v)))) == 0
Ft = sp.diff(F0, tt[0])
print("(F0)_t1 =", sp.factor(Ft), "  target -2 t1[1+s^2 c'(v)]")
Ft_sub = Ft.replace(lambda e: isinstance(e, sp.Subs), lambda e: cprime(v))
assert sp.simplify(Ft_sub - (-2*tt[0]*(1 + s**2*cprime(v)))) == 0
# near origin c(v)=v/4
F0near = b - s**2*(v/4) - sum(ti**2 for ti in tt)
target = b - s**4/4 - (1 + s**2/4)*sum(ti**2 for ti in tt)
assert sp.expand(F0near - target) == 0
print("near origin F0 = b - s^4/4 - (1+s^2/4)|t|^2 : OK")
us = [ti*sqrt(1 + s**2/4) for ti in tt]
assert sp.simplify(target - (b - s**4/4 - sum(ui**2 for ui in us))) == 0
print("(10) with u = t sqrt(1+s^2/4): OK")
Hn = sp.hessian(target, (s,) + tuple(tt)).subs({s: 0, **{ti: 0 for ti in tt}})
print("Hessian at origin:", Hn.tolist(), " corank =", d - Hn.rank())
print("3rd, 4th axial derivatives at 0:", sp.diff(target, s, 3).subs({s: 0, **{ti: 0 for ti in tt}}), sp.diff(target, s, 4).subs({s: 0, **{ti: 0 for ti in tt}}))

print("=== Sec 4 (11)-(13): restricted unfolding ===")
Psi = -s**4/4 + lam*s**2/2 + eta*lam**Q(3, 2)*s
Phi = -z**4/4 + z**2/2 + eta*z
assert sp.simplify(Psi.subs(s, sqrt(lam)*z) - lam**2*Phi) == 0
print("Psi_lambda(sqrt(lam) z) = lam^2 Phi_eta(z): OK")
assert sp.expand(sp.diff(Phi, z) - f) == 0
print("Phi' = -z^3+z+eta: OK")
Flam_core = b + Psi - (1 + s**2/4)*sum(ti**2 for ti in tt)
Hc = sp.hessian(Flam_core, (s,) + tuple(tt)).subs({ti: 0 for ti in tt}).subs(s, sqrt(lam)*z)
Hc = sp.simplify(Hc)
print("Hess at (sqrt(lam) z, 0):", Hc.tolist())
assert sp.simplify(Hc[0, 0] - lam*(1 - 3*z**2)) == 0 and sp.simplify(Hc[1, 1] + 2*(1 + lam*z**2/4)) == 0 and all(Hc[i, j] == 0 for i in range(d) for j in range(d) if i != j)
print("(13) Hessian: OK")
# signs of f at -1, -1/sqrt3, 0, 1
for pt in [-1, -1/sqrt(3), 0, 1]:
    print("  f(%s) = %s" % (pt, sp.simplify(f.subs(z, pt))))
# (14) derivative of Phi(z+)-Phi(z-) in eta equals z+ - z-
print("d/deta Phi_eta(z) at fixed z =", sp.diff(Phi, eta))
# Phi(1) = 1/4+eta
print("Phi(1) =", sp.expand(Phi.subs(z, 1)))
# endpoint: eta=2/(3 sqrt3), z_-=z_0=-1/sqrt3
eta_end = 2/(3*sqrt(3))
print("Phi at double root (-1/sqrt3) for eta=2/(3sqrt3):", sp.nsimplify(sp.simplify(Phi.subs({z: -1/sqrt(3), eta: eta_end}))))
print("Phi_0(+-1) =", Phi.subs({eta: 0, z: 1}), Phi.subs({eta: 0, z: -1}))

print("=== Sec 6: delta and barrier arithmetic ===")
R = sp.symbols('R', positive=True)
r1 = R/4
delta = r1**4/64
print("c(r1^2/4) = (r1^2/4)/4 =", sp.simplify((r1**2/4)/4), " = r1^2/16:", sp.simplify((r1**2/4)/4 - r1**2/16) == 0)
print("(r1^2/16)*(r1^2/4) =", sp.simplify((r1**2/16)*(r1**2/4)), " vs delta =", delta, ":", sp.simplify((r1**2/16)*(r1**2/4) - delta) == 0)
print("R^2 - delta =", sp.factor(R**2 - delta), " positive for 0<R<1")
