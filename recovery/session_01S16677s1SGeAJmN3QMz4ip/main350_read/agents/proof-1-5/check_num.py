# Numeric checks of PROOF.md Sections 4-5 (kappa, q, nu, tip, mass)
from mpmath import mp, mpf, sqrt, polyroots, quad, findroot, diff, inf
mp.dps = 40
eta_c = mpf(2)/(3*sqrt(3))
a_c = mpf(3)**(-mpf(1)/4)
K_c = mpf(4)/mpf(3)**(mpf(5)/4)
Q_c = (mpf(2)/3)*K_c**(-mpf(2)/3)

from mpmath import acos, cos, pi
def roots(eta):
    # z^3 - z - eta = 0, three real roots for 0<=eta<eta_c: z = (2/sqrt3) cos(theta - 2 pi k/3), cos(3 theta) = eta/eta_c
    th = acos(eta/eta_c)/3
    rs = sorted([(2/sqrt(3))*cos(th - 2*pi*k/3) for k in range(3)])
    # polish with Newton on exact cubic
    rs = [findroot(lambda z: z**3 - z - eta, r) for r in rs]
    return rs
def Phi(z, eta): return -z**4/4 + z**2/2 + eta*z
def kappa(eta):
    zm, z0, zp = roots(eta)
    return Phi(zm, eta) - Phi(z0, eta)
def qfun_eta(eta):
    zm, z0, zp = roots(eta)
    return 1/(z0 - zm)

print("Section 3 root intervals check over eta grid:")
ok = True
for k in range(1, 200):
    eta = eta_c*k/200
    zm, z0, zp = roots(eta)
    cond = (-1 < zm < -1/sqrt(3)) and (-1/sqrt(3) < z0 < 0) and (zp > 1)
    ok &= cond
    # Phi(z_-) > Phi(z_0), Phi(z_+) > Phi(z_-)
    ok &= Phi(zm,eta) > Phi(z0,eta) and Phi(zp,eta) > Phi(zm,eta)
print("  all intervals and orderings hold:", ok)

print("Section 4: kappa(0+) =", kappa(mpf(10)**-20), " kappa'(eta) vs z_- - z_0:")
for eta in [mpf('0.05'), mpf('0.2'), mpf('0.35')]:
    zm, z0, zp = roots(eta)
    print("  eta=%s  num kappa'=%s   z_- - z_0=%s" % (eta, diff(kappa, eta), zm - z0))
print("  kappa strictly decreasing on grid:", all(kappa(eta_c*(k+1)/300) < kappa(eta_c*k/300) for k in range(0,299)))

print("(22) kappa/delta^(3/2) -> K_c = %s ; (z_0-z_-)/(2 a_c sqrt(delta)) -> 1" % K_c)
for e in [2, 4, 6, 8, 10]:
    d = mpf(10)**(-e)
    eta = eta_c - d
    zm, z0, zp = roots(eta)
    print("  delta=1e-%d  ratio_kappa=%s  ratio_gap=%s" % (e, kappa(eta)/d**(mpf(3)/2), (z0 - zm)/(2*a_c*sqrt(d))))

print("(23) h^(1/3) q(h) -> Q_c = %s" % Q_c)
for e in [2, 4, 6, 8, 10, 12]:
    d = mpf(10)**(-e)
    eta = eta_c - d
    h = kappa(eta)
    print("  h=%.3e  h^(1/3) q = %s" % (h, h**(mpf(1)/3)*qfun_eta(eta)))
print("  q at h->1/4 (eta=1e-12):", qfun_eta(mpf(10)**-12))

# q(h) as function of h via inversion
def eta_of_h(h):
    return findroot(lambda e: kappa(e) - h, (mpf(0), eta_c - mpf(10)**-30), solver='bisect', tol=mpf(10)**-35)
def q(h):
    if h >= mpf(1)/4: return mpf(1)
    return qfun_eta(eta_of_h(h))

# A_q bound: sup h^(1/3) q(h) on (0,1/4]
sup = max(h**(mpf(1)/3)*q(h) for h in [mpf(1)/4*k/60 for k in range(1,61)] + [mpf(10)**-k for k in range(2,9)])
print("(24) sup over grid of h^(1/3) q(h) =", sup, " (Q_c=%s, 1/4^(1/3)=%s)" % (Q_c, (mpf(1)/4)**(mpf(1)/3)))

# Section 5: choose a sample u_0, v_0 = eta_c u_0^(3/2), c_0
u0 = mpf(1)/2
v0 = eta_c*u0**(mpf(3)/2)
c0 = 1/(4*u0*v0)
C_edge = (8*c0/7)*K_c**(-mpf(2)/3)*u0**(mpf(7)/6)
def nu(t):
    if t >= u0**2/4: return mpf(0)
    return 2*c0*quad(lambda uu: uu**(-mpf(1)/2)*q(t/uu**2), [2*sqrt(t), u0])
print("(8)/(27) t^(1/3) nu(t) -> C_edge = %s" % C_edge)
mp.dps = 25
for e in [2, 3, 4, 5, 6]:
    t = mpf(10)**(-e)
    print("  t=1e-%d  t^(1/3) nu(t) = %s" % (e, t**(mpf(1)/3)*nu(t)))

# (6) vs direct pushforward: integral_0^T nu = 2 c0 int_0^{u0} int_0^{eta_c} 1{u^2 kappa(eta) < T} u^{3/2} deta du
def direct_cum(T):
    def inner(uu):
        # eta range with u^2 kappa(eta) < T  <=> kappa(eta) < T/u^2 <=> eta > kappa^{-1}(T/u^2)
        h = T/uu**2
        if h >= mpf(1)/4: return eta_c
        return eta_c - eta_of_h(h)
    return 2*c0*quad(lambda uu: uu**(mpf(3)/2)*inner(uu), [0, 2*sqrt(T), u0])
for T in [mpf('0.001'), mpf('0.01')]:
    lhs = quad(nu, [0, T])
    rhs = direct_cum(T)
    print("(6) cumulative via nu = %s ; direct 2D pushforward = %s ; (3/2)C_edge T^(2/3) = %s" % (lhs, rhs, mpf(3)/2*C_edge*T**(mpf(2)/3)))

# (26) total mass
print("(26) total mass via direct = %s ; eta_c u0^(3/2)/(5 v0) = %s" % (direct_cum(u0**2), eta_c*u0**(mpf(3)/2)/(5*v0)))

# (9)/(29) tip density: restrict eta <= eta_c - eps
eps = mpf('0.1')
C_tip = c0*quad(lambda e: kappa(e)**(-mpf(5)/4), [0, eta_c - eps])
kmin = kappa(eta_c - eps)
def tip_cum(T):
    def inner(uu):
        h = T/uu**2
        if h >= mpf(1)/4: return eta_c - eps
        e0 = eta_of_h(h)
        return max(mpf(0), (eta_c - eps) - e0)
    return 2*c0*quad(lambda uu: uu**(mpf(3)/2)*inner(uu), [0, 2*sqrt(T), u0])
for T in [u0**2*kmin/2, u0**2*kmin/10]:
    print("(9) tip cumulative direct = %s ; (4/5) C_tip T^(5/4) = %s" % (tip_cum(T), mpf(4)/5*C_tip*T**(mpf(5)/4)))
