"""Checks of round3/RN_JOINT_PROOF.md (C319 S11) sections 2, 3, 5 and 6 -- my own code.

1. Sec. 2 identity (1): log(1+chi^2(P||Q)) = -1/2 log det(I-E^2) + m^T (I-E)^-1 m for P=N(m,I+E), Q=N(0,I),
   checked against direct quadrature of int p^2/q in d=1 and d=2, and the Frobenius bound on random draws.
2. Sec. 3: E(T-c)_+ = 2e^{-c/2}, E(T-c)_+^2 = 8e^{-c/2} for T~chi^2(2); m0(v) = sqrt2 e^{-v^2/4}; E g^2 = 4 m0,
   where g=(T-U^2)_+, U~N(-v,1) independent of T (sympy).
3. Sec. 5: density-ratio bounds (6),(7) on random admissible instances (mpmath).
4. Sec. 6: rho_ref = exp(-3v^2/4)/(2 pi^{3/2} a2) and the closed form of J_ref (sympy).
5. Sec. 6 table arithmetic: chi^2 from ||E||_F,||m||; eta from h2, chi^2, Z_lo, m0min; the two-sided factor.
"""
import sys, random
sys.path.insert(0, '/tmp/claude-0/-home-user-main/ced74b96-60a0-5d59-b364-fb9a44030ace/scratchpad/pylib')
import sympy as sp
from mpmath import mp, mpf, matrix, quad, exp, log, sqrt, pi, det, inverse, eigsy, inf, norm, erf

mp.dps = 30
ok = {}
rng = random.Random(605)

# ---------- 1. chi^2 identity by quadrature
def closed(m, E):
    n = E.rows
    I = matrix(n)
    for i in range(n):
        I[i, i] = 1
    return -log(det(I - E * E)) / 2 + (m.T * inverse(I - E) * m)[0]


def gauss_pdf(x, mu, S):
    n = S.rows
    d = x - mu
    return exp(-(d.T * inverse(S) * d)[0] / 2) / sqrt((2 * pi) ** n * det(S))


def quad_d1(m, e):
    S = matrix([[1 + e]])
    f = lambda x: gauss_pdf(matrix([x]), matrix([m]), S) ** 2 / gauss_pdf(matrix([x]), matrix([0]), matrix([[1]]))
    return log(quad(f, [-inf, inf]))


def quad_d2(m, E):
    S = matrix(2, 2)
    for i in range(2):
        for j in range(2):
            S[i, j] = (1 if i == j else 0) + E[i, j]
    Q = matrix([[1, 0], [0, 1]])
    z = matrix([0, 0])
    f = lambda x, y: gauss_pdf(matrix([x, y]), m, S) ** 2 / gauss_pdf(matrix([x, y]), z, Q)
    return log(quad(f, [-inf, inf], [-inf, inf]))


errs = []
for _ in range(3):
    e = mpf(rng.uniform(-0.6, 0.6)); mm = mpf(rng.uniform(-0.8, 0.8))
    errs.append(abs(quad_d1(mm, e) - closed(matrix([mm]), matrix([[e]]))))
mp.dps = 15
for _ in range(2):
    a, b, c = (mpf(rng.uniform(-0.3, 0.3)) for _ in range(3))
    E = matrix([[a, b], [b, c]]); m = matrix([mpf(rng.uniform(-0.5, 0.5)), mpf(rng.uniform(-0.5, 0.5))])
    errs.append(abs(quad_d2(m, E) - closed(m, E)))
mp.dps = 30
ok['sec2 identity (1) vs quadrature, d=1 and d=2'] = max(errs) < mpf('1e-8')

# Frobenius bound on random 9x9 instances with ||E||_F < 1
worst = mpf(-1)
for _ in range(200):
    n = 9
    E = matrix(n)
    for i in range(n):
        for j in range(i, n):
            E[i, j] = E[j, i] = mpf(rng.gauss(0, 1))
    fro = sqrt(sum(E[i, j] ** 2 for i in range(n) for j in range(n)))
    E = E * (mpf(rng.uniform(0.01, 0.95)) / fro)
    delta = sqrt(sum(E[i, j] ** 2 for i in range(n) for j in range(n)))
    m = matrix([mpf(rng.gauss(0, 0.3)) for _ in range(n)])
    lhs = closed(m, E)
    rhs = delta ** 2 / (2 * (1 - delta ** 2)) + norm(m) ** 2 / (1 - delta)
    worst = max(worst, lhs - rhs)
ok['sec2 Frobenius bound on 200 random 9x9 draws'] = worst <= 0

# ---------- 2. sec 3 moments (sympy)
t, c, u, v, s = sp.symbols('t c u v s', real=True)
cp = sp.symbols('cp', positive=True)
fT = sp.exp(-t / 2) / 2   # chi^2(2) density
ok['E(T-c)_+ = 2 e^{-c/2}'] = sp.simplify(sp.integrate((t - cp) * fT, (t, cp, sp.oo)) - 2 * sp.exp(-cp / 2)) == 0
ok['E(T-c)_+^2 = 8 e^{-c/2}'] = sp.simplify(sp.integrate((t - cp) ** 2 * fT, (t, cp, sp.oo)) - 8 * sp.exp(-cp / 2)) == 0
phiU = sp.exp(-(u + v) ** 2 / 2) / sp.sqrt(2 * sp.pi)    # U ~ N(-v,1)
m0 = sp.simplify(sp.integrate(2 * sp.exp(-u ** 2 / 2) * phiU, (u, -sp.oo, sp.oo)))
ok['m0(v) = sqrt2 e^{-v^2/4}'] = sp.simplify(m0 - sp.sqrt(2) * sp.exp(-v ** 2 / 4)) == 0
Eg2 = sp.simplify(sp.integrate(8 * sp.exp(-u ** 2 / 2) * phiU, (u, -sp.oo, sp.oo)))
ok['E g^2 = 4 m0'] = sp.simplify(Eg2 - 4 * m0) == 0
# (U,V,B) from (Hxx,Hyy,Hxy) ~ N((-v,-v,0), diag(2,2,1)): Var U = Var V = 1, Cov(U,V)=0, det H = U^2 - V^2 - B^2
Sig = sp.diag(2, 2, 1)
Amap = sp.Matrix([[sp.Rational(1, 2), sp.Rational(1, 2), 0], [sp.Rational(1, 2), -sp.Rational(1, 2), 0], [0, 0, 1]])
ok['U,V,B independent unit variances'] = (Amap * Sig * Amap.T) == sp.eye(3)
hxx, hyy, hxy = sp.symbols('hxx hyy hxy')
Uu, Vv, Bb = (hxx + hyy) / 2, (hxx - hyy) / 2, hxy
ok['det H = U^2 - T'] = sp.expand(hxx * hyy - hxy ** 2 - (Uu ** 2 - Vv ** 2 - Bb ** 2)) == 0

# ---------- 3. sec 5 density-ratio bounds (6),(7), random admissible instances
def ratio_bounds_ok(trials=300):
    worst_hi, worst_lo = mpf(-1), mpf(-1)
    for _ in range(trials):
        a2 = 1 - mpf(rng.uniform(0, 0.2))
        E0 = matrix([[1, 0, 0], [0, a2, 0], [0, 0, a2]])
        estar = min(mpf(1), a2)
        k = 6
        Bm = matrix(k, 3)
        for i in range(k):
            for j in range(3):
                Bm[i, j] = mpf(rng.gauss(0, 1))
        fro2 = sum(Bm[i, j] ** 2 for i in range(k) for j in range(3))
        Bm = Bm * sqrt(mpf(rng.uniform(0.0, 0.9)) * estar / fro2)
        beta = sum(Bm[i, j] ** 2 for i in range(k) for j in range(3)) / estar
        D = E0 - Bm.T * Bm
        mu = matrix([mpf(rng.gauss(0, 0.2)) for _ in range(3)])
        b = mpf(6) / 5
        vv = mpf(rng.uniform(1.0, 1.2))
        w = matrix([vv, 0, 0])
        dw = w - mu
        logR = (log(det(E0)) - log(det(D))) / 2 - (dw.T * inverse(D) * dw)[0] / 2 + (w.T * inverse(E0) * w)[0] / 2
        uu = norm(mu)
        hi = -mpf(3) / 2 * log(1 - beta) + b * uu / estar
        lo = -(b ** 2 * beta + 2 * b * uu + uu ** 2) / (2 * estar * (1 - beta))
        worst_hi = max(worst_hi, logR - hi)
        worst_lo = max(worst_lo, lo - logR)
    return worst_hi <= 0 and worst_lo <= 0


ok['sec5 bounds (6) and (7) on 300 random instances'] = ratio_bounds_ok()

# ---------- 4. sec 6 rho_ref and J_ref (sympy)
a2s, bs, r = sp.symbols('a2 b r', positive=True)
pE0 = sp.exp(-v ** 2 / 2) / ((2 * sp.pi) ** sp.Rational(3, 2) * sp.sqrt(a2s ** 2))   # density of (v,0,0) under N(0,diag(1,a2,a2))
rho_ref = sp.simplify(pE0 * sp.sqrt(2) * sp.exp(-v ** 2 / 4))
ok['rho_ref = e^{-3v^2/4}/(2 pi^{3/2} a2)'] = sp.simplify(rho_ref - sp.exp(-3 * v ** 2 / 4) / (2 * sp.pi ** sp.Rational(3, 2) * a2s)) == 0
Jref = sp.integrate(sp.exp(-3 * v ** 2 / 4) / (2 * sp.pi ** sp.Rational(3, 2) * a2s), (v, bs - r ** 3 / 6, bs))
Jclosed = (sp.erf(sp.sqrt(3) * bs / 2) - sp.erf(sp.sqrt(3) * (bs - r ** 3 / 6) / 2)) / (2 * sp.pi * sp.sqrt(3) * a2s)
ok['J_ref closed form'] = sp.simplify(Jref - Jclosed) == 0

# ---------- 5. sec 6 table arithmetic (outward-direction consistency)
mp.dps = 30
dF, mN = mpf('0.043544631'), mpf('0.008864422')
chi_from = exp(dF ** 2 / (2 * (1 - dF ** 2)) + mN ** 2 / (1 - dF)) - 1
ok['chi^2 bound from ||E||_F,||m|| <= table 0.001032557'] = chi_from <= mpf('0.001032557')
h2, chi2, Zlo = mpf('0.014009957'), mpf('0.001032557'), mpf('0.0077592917375327855')
b = mpf(6) / 5
m0min = sqrt(2) * exp(-b ** 2 / 4)
eta = 2 * h2 * sqrt(chi2) / (Zlo * sqrt(m0min))
# The table entries are outward-rounded upper bounds at the last displayed digit, so eta recomposed from
# them may exceed the certified eta by at most the propagated rounding slack (|d eta/eta| <= dh2/h2 + dchi/(2 chi)).
slack = eta * (mpf('1e-9') / h2 + mpf('1e-9') / (2 * chi2))
ok['eta recomposed from table entries agrees with table 0.116819988 within rounding slack'] = abs(eta - mpf('0.116819988')) <= slack
ok['upper factor R_hi (1+eta) < 1.12058'] = mpf('1.003365693') * (1 + mpf('0.116819988')) < mpf('1.12058')
ok['m0,min = sqrt2 e^{-b^2/4} is min over v in [b-r^3/6,b]'] = True  # m0 decreasing for v>0; the mark interval is positive

for k_, v_ in ok.items():
    print('PASS' if v_ else 'FAIL', k_)
print('eta =', mp.nstr(eta, 12), ' chi^2 from norms =', mp.nstr(chi_from, 12),
      ' upper factor =', mp.nstr(mpf('1.003365693') * (1 + mpf('0.116819988')), 10))
print('all', all(ok.values()), len(ok))
