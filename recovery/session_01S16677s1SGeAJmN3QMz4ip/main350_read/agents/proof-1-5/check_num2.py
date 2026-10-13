from mpmath import mp, mpf, sqrt, quad, findroot, acos, cos, pi
mp.dps = 20
eta_c = mpf(2)/(3*sqrt(3)); a_c = mpf(3)**(-mpf(1)/4)
K_c = mpf(4)/mpf(3)**(mpf(5)/4); Q_c = (mpf(2)/3)*K_c**(-mpf(2)/3)
def roots(eta):
    th = acos(eta/eta_c)/3
    rs = sorted([(2/sqrt(3))*cos(th - 2*pi*k/3) for k in range(3)])
    return [findroot(lambda z: z**3 - z - eta, r) for r in rs]
def Phi(z, eta): return -z**4/4 + z**2/2 + eta*z
def kappa(eta):
    zm, z0, zp = roots(eta); return Phi(zm, eta) - Phi(z0, eta)
def eta_of_h(h):
    return findroot(lambda e: kappa(e) - h, (mpf(0), eta_c - mpf(10)**-18), solver='bisect', tol=mpf(10)**-18)
def q(h):
    if h >= mpf(1)/4: return mpf(1)
    zm, z0, zp = roots(eta_of_h(h)); return 1/(z0 - zm)
u0 = mpf(1)/2; v0 = eta_c*u0**(mpf(3)/2); c0 = 1/(4*u0*v0)
C_edge = (8*c0/7)*K_c**(-mpf(2)/3)*u0**(mpf(7)/6)
def nu6(t):   # PROOF (6) literally
    if t >= u0**2/4: return mpf(0)
    return 2*c0*quad(lambda uu: uu**(-mpf(1)/2)*q(t/uu**2), [2*sqrt(t), u0], maxdegree=6)
def nu_eta(t): # independent eta-form: c0 t^(1/4) int_0^{kappa^{-1}(t/u0^2)} kappa^(-5/4) d eta
    if t >= u0**2/4: return mpf(0)
    et = eta_of_h(t/u0**2)
    return c0*t**(mpf(1)/4)*quad(lambda e: kappa(e)**(-mpf(5)/4), [0, et])
for t in [mpf('0.01'), mpf('0.001'), mpf('0.05')]:
    print("t=%s  nu(6)=%s  nu(eta-form)=%s" % (t, nu6(t), nu_eta(t)))
# direct 2D pushforward cumulative (1D in eta, closed form in u) and mass
def direct_cum(T):
    return 2*c0*quad(lambda e: (mpf(2)/5)*min(u0, sqrt(T/kappa(e)))**(mpf(5)/2), [0, eta_c])
for T in [mpf('0.001'), mpf('0.01')]:
    print("T=%s  int_0^T nu(6) = %s  direct 2D = %s  (3/2)C_edge T^(2/3) = %s" % (T, quad(nu6, [0, T], maxdegree=5), direct_cum(T), mpf(3)/2*C_edge*T**(mpf(2)/3)))
print("(26) mass direct = %s ; eta_c u0^(3/2)/(5 v0) = %s" % (direct_cum(u0**2), eta_c*u0**(mpf(3)/2)/(5*v0)))
# (9) tip law
eps = mpf('0.1'); kmin = kappa(eta_c - eps)
C_tip = c0*quad(lambda e: kappa(e)**(-mpf(5)/4), [0, eta_c - eps])
def tip_cum(T):
    return 2*c0*quad(lambda e: (mpf(2)/5)*min(u0, sqrt(T/kappa(e)))**(mpf(5)/2), [0, eta_c - eps])
for T in [u0**2*kmin/2, u0**2*kmin/10]:
    print("(9) T=%s tip cumulative direct = %s ; (4/5) C_tip T^(5/4) = %s" % (T, tip_cum(T), mpf(4)/5*C_tip*T**(mpf(5)/4)))
# (28) small-u contribution to t^(1/3) nu(t): compare to (12 c0 Q_c/7) r^(7/6) limit at small t
r = mpf('0.05'); t = mpf(10)**-6
small = 2*c0*quad(lambda uu: t**(mpf(1)/3)*uu**(-mpf(1)/2)*q(t/uu**2), [2*sqrt(t), r], maxdegree=6)
print("(28) contribution u<=r=%s at t=1e-6: %s ; limit (12 c0 Q_c/7) r^(7/6) = %s" % (r, small, 12*c0*Q_c/7*r**(mpf(7)/6)))
