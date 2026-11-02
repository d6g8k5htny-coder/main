"""Checks of the cap-region bound, Lemma S8.16 (new in draft 11).

Run: python3 -I s8_capK_check.py   (prints ALL_CHECKS_PASS at the end)
Needs SymPy.

1. The endpoint envelope (S8.39) in exact rational arithmetic, for d = 2, 3, 4 (m = 1, 2, 3):
   random pinned-type data with |alpha_i| <= a, ||beta_i|| <= M_3/2, ||A_S - A_M|| <= r M_3 and
   H_M negative definite, where B = -A_M and the perturbation A_S - A_M are built from exact rational
   orthogonal matrices (Cayley transforms), so their eigenvalues are known exactly. For each sample
   the script checks |det H_M| <= r a prod lambda_j,
   |det H_S| <= r [a(lambda_1 + r M_3) + r M_3^2/4] prod_{j>=2}(lambda_j + r M_3), and (S8.39).
   The samples include the extreme values |alpha_S| = a and |e_j| = r M_3.
2. The layer integral of the proof (depth failure, m >= 2) and the bound of each of its three terms
   by (r^3/k) P Upsilon^8 times a constant, given a <= C_* k Upsilon, r <= k <= P and Upsilon >= 1.
3. The arithmetic of the far branch and of the exception: delta^4 (k + r)^2 <= 4 delta r^3/k for r <= k.
"""
import random
import sympy as sp

ok = []


def check(name, cond):
    if not bool(cond):
        raise AssertionError(name)
    ok.append(name)
    print('PASS', name, flush=True)


rng = random.Random(20261019)


def rat(lo, hi, den=97):
    return sp.Rational(rng.randint(int(lo * den), int(hi * den)), den)


def cayley(m):
    S = sp.zeros(m, m)
    for i in range(m):
        for j in range(i + 1, m):
            v = rat(-2, 2)
            S[i, j], S[j, i] = v, -v
    I = sp.eye(m)
    return (I - S) * (I + S).inv()


def block(r, alpha, beta, A):
    m = A.shape[0]
    H = sp.zeros(m + 1, m + 1)
    H[0, 0] = r * alpha
    for j in range(m):
        H[0, j + 1] = H[j + 1, 0] = r * beta[j]
    H[1:, 1:] = A
    return H


def envelope(a, lam, r, M3):
    """The right side of the envelope: a l1 [a l1 + a r M3 + r M3^2/4] prod_{j>=2} lj (lj + r M3)."""
    return a * lam[0] * (a * lam[0] + a * r * M3 + r * M3 ** 2 / 4) * sp.prod([l * (l + r * M3) for l in lam[1:]])


def beta_term(a, lam, r, M3):
    """The part of the envelope that comes from ||beta_S||^2 <= M3^2/4."""
    return a * lam[0] * (r * M3 ** 2 / 4) * sp.prod([l * (l + r * M3) for l in lam[1:]])


# ------------------------------------------------------------------ 1. the envelope
for m in (1, 2, 3):
    n_ok = 0
    tries = 0
    while n_ok < 40:
        tries += 1
        assert tries < 4000
        r = rat(0.01, 1)
        M3 = rat(0.1, 5)
        a = rat(0.05, 4)
        lam = sorted(rat(0.01, 3) for _ in range(m))
        if len(set(lam)) < m:
            continue
        O = cayley(m)
        assert sp.simplify(O * O.T - sp.eye(m)) == sp.zeros(m, m)
        B = O * sp.diag(*lam) * O.T
        AM = -B
        bnd = M3 / (2 * m)
        betaM = [rat(-1, 1) * bnd for _ in range(m)]
        betaS = [rat(-1, 1) * bnd for _ in range(m)]
        alphaM = -a * rat(0.01, 1)
        # the extremes |alpha_S| = a and |e_j| = r M_3 in every fourth sample
        alphaS = (a if n_ok % 8 == 0 else -a) if n_ok % 4 == 0 else a * rat(-1, 1)
        e = [r * M3 * (1 if (n_ok % 4 == 0 and j % 2 == 0) else rat(-1, 1)) for j in range(m)]
        O2 = cayley(m)
        AS = AM + O2 * sp.diag(*e) * O2.T
        HM = block(r, alphaM, betaM, AM)
        HS = block(r, alphaS, betaS, AS)
        # H_M negative definite: -B < 0 and the Schur complement r alpha_M + r^2 beta^T B^{-1} beta < 0
        bv = sp.Matrix(betaM)
        if not (r * alphaM + r ** 2 * (bv.T * B.inv() * bv)[0, 0] < 0):
            continue
        dM, dS = abs(HM.det()), abs(HS.det())
        prodM = sp.prod(lam)
        boundM = r * a * prodM
        boundS = r * (a * (lam[0] + r * M3) + r * M3 ** 2 / 4) * sp.prod([l + r * M3 for l in lam[1:]])
        env = envelope(a, lam, r, M3)
        assert dM <= boundM, ('M', m, n_ok)
        assert dS <= boundS, ('S', m, n_ok)
        assert dM * dS / r ** 2 <= env, ('env', m, n_ok)
        n_ok += 1
    check('1 d=%d: 40 exact samples satisfy the three endpoint bounds and (S8.39)' % (m + 1), True)

# equality cases: the bounds are attained, so none of their terms can be dropped
for m in (1, 2, 3):
    for trial in range(5):
        r = rat(0.05, 1)
        M3 = rat(0.5, 5)
        a = rat(0.5, 4)
        lam = sorted(set(rat(0.01, 3) for _ in range(m)))
        if len(lam) < m:
            continue
        O = cayley(m)
        B = O * sp.diag(*lam) * O.T
        AM = -B
        AS = AM + O * sp.diag(*([-r * M3] * m)) * O.T          # -A_S has eigenvalues lambda_j + r M_3
        HM = block(r, -a, [0] * m, AM)
        if m == 1:
            HS = block(r, a, [M3 / 2], AS)                     # the two terms of det H_S add up
            exact_env = envelope(a, lam, r, M3)
        else:
            HS = block(r, a, [0] * m, AS)                      # beta_S = 0: only the beta part is slack
            exact_env = envelope(a, lam, r, M3) - beta_term(a, lam, r, M3)
        val = abs(HM.det()) * abs(HS.det()) / r ** 2
        assert sp.simplify(val - exact_env) == 0, (m, trial)
    check('1e d=%d: equality cases (aligned perturbation, |alpha| = a%s)' % (m + 1, ', ||beta_S|| = M_3/2' if m == 1 else ''), True)

# ------------------------------------------------------------------ 2. the layer integral (m >= 2)
lam1, a_, r_, K, D, dl, U, k_, P_, Cs = sp.symbols('lambda a r K D delta Upsilon k P C_star', positive=True)
X = D * dl * U ** 2
inner = sp.integrate(lam1 * (a_ * lam1 + a_ * r_ * K * U + r_ * K ** 2 * U ** 2 / 4), (lam1, 0, X))
lhs = sp.expand(a_ * inner)
rhs = sp.expand(a_ ** 2 * D ** 3 / 3 * dl ** 3 * U ** 6 + a_ ** 2 * r_ * K * D ** 2 / 2 * dl ** 2 * U ** 5
                + a_ * r_ * K ** 2 * D ** 2 / 8 * dl ** 2 * U ** 6)
check('2a layer integral a * int_0^{D delta U^2} lambda [a lambda + a r K U + r K^2 U^2/4] d lambda', sp.simplify(lhs - rhs) == 0)
# each term at a = C_* k U and delta = r/k, divided by (r^3/k) P U^8, is a constant times a product of
# factors each <= 1 when r <= k <= P and U >= 1
sub = {a_: Cs * k_ * U, dl: r_ / k_}
terms = [a_ ** 2 * D ** 3 / 3 * dl ** 3 * U ** 6, a_ ** 2 * r_ * K * D ** 2 / 2 * dl ** 2 * U ** 5,
         a_ * r_ * K ** 2 * D ** 2 / 8 * dl ** 2 * U ** 6]
expected = [Cs ** 2 * D ** 3 / 3 / P_, Cs ** 2 * K * D ** 2 / 2 * k_ / (P_ * U), Cs * K ** 2 * D ** 2 / 8 / (P_ * U)]
for i, (t, ex) in enumerate(zip(terms, expected)):
    q = sp.simplify(t.subs(sub) / ((r_ ** 3 / k_) * P_ * U ** 8))
    check('2b term %d / ((r^3/k) P Upsilon^8) = %s, bounded as k <= P, P >= 1, Upsilon >= 1' % (i + 1, ex), sp.simplify(q - ex) == 0)

# ------------------------------------------------------------------ 3. far branch and exception
dl3 = r_ / k_
expr = dl3 ** 4 * (2 * k_) ** 2 - 4 * dl3 * r_ ** 3 / k_
check('3 delta^4 (2k)^2 = 4 delta r^3/k, and (k + r)^2 <= (2k)^2 for r <= k', sp.simplify(expr) == 0)

print('N_CHECKS', len(ok))
print('ALL_CHECKS_PASS')
