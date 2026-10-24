"""Lens: confidence/dependence/power. Paper algebra only; no field/sampler/driver."""
import sympy as sp
import mpmath as mp
from fractions import Fraction as Fr
import random, itertools

mp.mp.dps = 40
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

# ---- P3 integers
say("P3:", 21**6, 2*19**6, 21**6 < 2*19**6, "q_old^3<2:", Fr(11,9)**3 < 2, "q_old^6>2:", Fr(11,9)**6 > 2)

# ---- P16 optimizer and P18 algebra (symbolic)
t, n, r, Q, V, s, J, al = sp.symbols('t n r Q V s J alpha', positive=True)
expo = -t*n*r + n*t**2*Q**2/8
tstar = sp.solve(sp.diff(expo, t), t)[0]
say("P16 tilt:", tstar, " exponent:", sp.simplify(expo.subs(t, tstar)))
nreq = sp.solve(sp.Eq(2*J*sp.exp(-2*n*(r/Q)**2), al), n)[0]
say("P18 n from P16=alpha:", sp.simplify(nreq), " with r=sV:", sp.simplify(nreq.subs(r, s*V)))

# ---- P20 width / D_j check (symbolic, untruncated branch)
Lb, Ub, B, tt = sp.symbols('Lbar Ubar B t_j', positive=True)
Am = (Lb - r)/V - B; Ap = (Ub + r)/V + B + tt
yhat = (Lb+Ub)/(2*V); D = (Ub-Lb)/(2*V) + r/V + B + tt
say("P19 width:", sp.simplify(Ap-Am), " | A+ - yhat - D =", sp.simplify(Ap-yhat-D), " | yhat - A- - D =", sp.simplify(yhat-Am-D))

# ---- Floor (a): P13 + Hoeffding (P16) + P22, single bin.
# constraints (per bin, null scenario nu*V >= (1-eta) m):  (1-eta)^2 m^2/(2Q) + 2 r <= eta m ;  n >= Q^2 ln(2J/alpha)/(2 r^2)
eta, x = sp.symbols('eta x', positive=True)
c = (1-eta)**2/2
rho = (eta - c/x)/2          # r/m at equality
f = x**2/(2*rho**2)          # n / ln(2J/alpha)
xs = sp.solve(sp.diff(sp.log(f), x), x)
xs = [z for z in xs if z.is_positive is not False]
say("floor(a) argmin Q/m:", [sp.simplify(z) for z in xs])
fmin = sp.simplify(f.subs(x, xs[0]))
say("floor(a) n/ln(2J/alpha) >=", sp.factor(fmin))
e20 = sp.Rational(1,20)
fa = fmin.subs(eta, e20)
say("  at eta=1/20:", fa, "; Q/m* =", sp.nsimplify(xs[0].subs(eta, e20)), "=", float(xs[0].subs(eta, e20)))
for (JJ, aa) in [(2, 0.999999), (8, 0.05), (8, 0.01), (8, 0.10)]:
    say(f"  J={JJ}, alpha={aa}: n >= {float(fa)*float(mp.log(2*JJ/aa)):.7g}")

# ---- Floor (b): any t-route (Jensen T_Q >= (nuV-Q)_+/V), Hoeffding, P22, single bin.
fb = 2*(1-eta)**2/eta**2
say("floor(b) n/ln(2J/alpha) >=", fb, " at eta=1/20:", fb.subs(eta, e20))
for (JJ, aa) in [(2, 0.999999), (8, 0.05), (8, 0.01)]:
    say(f"  J={JJ}, alpha={aa}: n >= {float(fb.subs(eta,e20))*float(mp.log(2*JJ/aa)):.7g}")
say("ratio floor(a)/floor(b) =", sp.simplify(fmin/fb), "=", sp.simplify(fmin/fb).subs(eta, e20))

# ---- KL (mean-adaptive) version of floor (a): numeric optimisation over Q/m and allocation.
def kl(p, q):
    return p*mp.log(p/q) + (1-p)*mp.log((1-p)/(1-q))
def n_kl(etav, xq, a):
    # Q = xq*m, mean of Z/Q ~ q0=1/xq (upper bound side uses q0+eps, lower q0-eps); P13 term uses (1-eta)^2 m/(2Q)
    tshare = (1-etav)**2/(2*xq)
    if tshare >= etav: return mp.inf
    rr = (etav - tshare)/2         # r/m
    q0 = (1-etav)/xq               # E Z/Q lower-ish (null); use q0 ~ nu V/Q
    eps = rr/xq
    if q0+eps >= 1 or q0-eps <= 0: return mp.inf
    k = min(kl(q0+eps, q0), kl(q0-eps, q0))
    return 1/k
best = min((n_kl(mp.mpf(1)/20, mp.mpf(xq)/4, 0), mp.mpf(xq)/4) for xq in range(5, 400))
say("floor(a) with KL instead of Hoeffding: n/ln(2J/alpha) >= approx", mp.nstr(best[0], 8), "at Q/m =", mp.nstr(best[1], 5))

# ---- Common Q and common r across J dyadic bins (masses m_j = m_0 2^(2j/3)).
def n_common(etav, Jb, p13=True):
    R = [mp.mpf(2)**(mp.mpf(2*j)/3) for j in range(Jb)]   # m_j/m_0
    best = (mp.inf, None)
    for k in range(1, 4000):
        xq = mp.mpf(k)/8  # Q/m_0
        # t_j*V/m_0 lower bounds
        if p13:
            tj = [(1-etav)**2 * Rj**2/(4*xq) for Rj in R]
        else:  # Jensen floor only
            tj = [max(0, (1-etav)*Rj - xq) for Rj in R]
        # r/m_0 <= min_j (eta*R_j - 2 t_j)/2
        rr = min((etav*Rj - 2*tjj)/2 for Rj, tjj in zip(R, tj))
        if rr <= 0: continue
        val = xq**2/(2*rr**2)
        if val < best[0]: best = (val, xq)
    return best
for Jb in [2, 8]:
    a1 = n_common(mp.mpf(1)/20, Jb, True); b1 = n_common(mp.mpf(1)/20, Jb, False)
    say(f"common Q,r, J={Jb}: P13 route n/ln >= {mp.nstr(a1[0],7)} (Q/m0={mp.nstr(a1[1],5)}); Jensen floor n/ln >= {mp.nstr(b1[0],7)} (Q/m0={mp.nstr(b1[1],5)})")
say("(m_{J-1}/m_0)^2 for J=8:", mp.nstr(mp.mpf(2)**(mp.mpf(28)/3), 8))

# ---- Shell ratio for dyadic bin [a,2a): (w(I+)-w(I-))/w(I) as function of u=delta/a
def w(lo, hi):
    return mp.mpf(3)/2*(hi**(mp.mpf(2)/3) - lo**(mp.mpf(2)/3)) if hi > lo else mp.mpf(0)
def shell_ratio(u):
    a = mp.mpf(1); b = 2*a; d = u
    return (w(a-2*d, b+2*d) - w(a+2*d, b-2*d))/w(a, b)
say("shell ratio small-u slope (x delta/a):", mp.nstr(shell_ratio(mp.mpf('1e-12'))/mp.mpf('1e-12'), 10),
    " closed form (8/3)(1+2^(-1/3))/(2^(2/3)-1) =", mp.nstr(mp.mpf(8)/3*(1+mp.mpf(2)**(-mp.mpf(1)/3))/(mp.mpf(2)**(mp.mpf(2)/3)-1), 10))
e = mp.mpf(1)/20
u14 = mp.findroot(lambda u: 2*shell_ratio(u) - e, mp.mpf(1)/300)
uX = mp.findroot(lambda u: shell_ratio(u) - e, mp.mpf(1)/150)
say("P14 route (2H<=eta C0 w, all budget): need a/delta >=", mp.nstr(1/u14, 8), "; X+- route (H once):", mp.nstr(1/uX, 8))
say("  => J=8 window ratio ell_*/delta >= 2^8 a/delta + 2 :", mp.nstr(256/u14 + 2, 8), "(P14),", mp.nstr(256/uX+2, 8), "(X+-)")

# ---- Randomised exact check of both sandwiches on synthetic delta-matched diagrams.
def cnt(xs, lo, hi):
    return sum(1 for z in xs if lo <= z < hi) if lo < hi else 0
random.seed(1)
bad_P8 = bad_X = 0
for trial in range(20000):
    dlt = Fr(random.randint(1, 20), 100)
    F, P = [], []
    for _ in range(random.randint(0, 12)):
        lf = Fr(random.randint(1, 400), 100)
        # matched partner, lifetime within 2 delta, or diagonal if lf <= 2 delta
        if lf <= 2*dlt and random.random() < 0.5:
            F.append(lf); continue
        lp = lf + Fr(random.randint(-int(200*dlt), int(200*dlt)), 100)
        if lp <= 0:
            if lf <= 2*dlt: F.append(lf); continue
            lp = lf
        F.append(lf); P.append(lp)
    for _ in range(random.randint(0, 3)):   # unmatched P bars near diagonal
        P.append(Fr(random.randint(1, int(200*dlt)), 100))
    a = Fr(random.randint(1, 300), 100); b = a + Fr(random.randint(1, 200), 100)
    if not a > 2*dlt: continue
    # P8
    if not (cnt(F, a+2*dlt, b-2*dlt) <= cnt(P, a, b) <= cnt(F, a-2*dlt, b+2*dlt)): bad_P8 += 1
    # reverse (X+-) sandwich
    if not (cnt(P, a+2*dlt, b-2*dlt) <= cnt(F, a, b) <= cnt(P, a-2*dlt, b+2*dlt)): bad_X += 1
say("randomised sandwiches: P8 violations", bad_P8, "; reverse N_P(I-)<=N_F(I)<=N_P(I+) violations", bad_X)

open('<SCRATCH>/proto/work_power/power_checks.out', 'w').write("\n".join(out) + "\n")
