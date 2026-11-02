"""Exact checks of the fold identities behind Lemma S10.12, Theorem S11.7 and Corollary S11.11.

Run: python3 -I s10_s11_fold_check.py   (prints ALL_CHECKS_PASS at the end; under a minute)
Needs SymPy. New in draft 10.

Take a polynomial field f of total degree D in the coordinates (x, y), x along u and y in u^perp
(m = d - 1 transverse coordinates), with symbolic coefficients. Solve the pins (S10.1),
    f(M) = b, f(S) = b - k r^3, grad f(M) = grad f(S) = 0,  M = -r/2 u, S = r/2 u,
for the pinned jets at 0 (f, d_x f, d_x^2 f, d_x^3 f, grad_y f, grad_y d_x f). The other
coefficients (the free jets) stay arbitrary. For such fields the remainder rho of (S10.40)
vanishes, so the r-expansion of -det K_M det K_S is the polynomial Omega of Lemma S10.12, Step 1.
The script checks, as identities of polynomials in the free jets, k and b:
  a. the pinned jets are polynomials in r (no negative power of r);
  b. alpha_i = d_x^2 f(i)/r and beta_i = grad_y d_x f(i)/r are polynomials in r (S10 inputs);
  c. (S10.16): det K_M = -6 k Delta + r Y_r + O(r^2) and det K_S = 6 k Delta + r Y_r + O(r^2);
  d. (S10.41): the coefficients of r^0 and r^1 of Omega are 36 k^2 Delta^2 and 0, and the
     coefficient of r^2 at k = 0 is -Y^2;
  e. (S11.22)-(S11.23): the coefficient of r^2 of Omega is X_2 = 12 k Delta Upsilon_2 - Upsilon_1^2
     of Theorem S11.7, Step 2, so the fold coefficient polynomial of S10 and the surrogate
     polynomial of S11 agree;
  f. the coefficients of r^0, r^1 and r^2 do not change from degree D to D + 1, so jets of
     order at most eight determine them.
It runs d = 2 with D = 6, 7, 8 and d = 3 with D = 6, 7. For d = 2 it also compares the bordered
formula det K_i = alpha_i det A_i - r beta_i^T adj(A_i) beta_i with det(H_i)/r.
"""
import itertools
import sympy as sp

ok = []


def check(name, cond):
    if not bool(cond):
        raise AssertionError(name)
    ok.append(name)
    print('PASS', name, flush=True)


def trunc(e, r, n):
    """Terms of degree 0..n in r of a polynomial in r (asserting there is no negative power)."""
    e = sp.expand(e)
    p = sp.Poly(e, r)
    return sp.Add(*[p.coeff_monomial(r ** j) * r ** j for j in range(n + 1)])


def run(m, D, R=4):
    x = sp.Symbol('x')
    ys = sp.symbols('y1:%d' % (m + 1))
    r, b, k = sp.symbols('r b k', positive=True)
    idx = [(a,) + beta for tot in range(D + 1) for a in range(tot + 1)
           for beta in itertools.product(range(tot - a + 1), repeat=m) if sum(beta) == tot - a]
    c = {al: sp.Symbol('c_' + '_'.join(map(str, al))) for al in idx}

    def mono(al):
        e, den = x ** al[0], sp.factorial(al[0])
        for j in range(m):
            e *= ys[j] ** al[1 + j]
            den *= sp.factorial(al[1 + j])
        return e / den

    f = sum(c[al] * mono(al) for al in idx)  # c[al] is the jet D^al f(0)
    zero = (0,) * m

    def unit(j):
        u = [0] * m
        u[j] = 1
        return tuple(u)

    pinned = [(0,) + zero, (1,) + zero, (2,) + zero, (3,) + zero] + \
             [(0,) + unit(j) for j in range(m)] + [(1,) + unit(j) for j in range(m)]
    sub0 = {yy: 0 for yy in ys}

    def at(expr, xv):
        s = dict(sub0)
        s[x] = xv
        return sp.expand(expr.subs(s))

    eqs = [sp.Eq(at(f, -r / 2), b), sp.Eq(at(f, r / 2), b - k * r ** 3)]
    for v in (x,) + ys:
        g = sp.diff(f, v)
        eqs += [sp.Eq(at(g, -r / 2), 0), sp.Eq(at(g, r / 2), 0)]
    sol = sp.solve(eqs, [c[p] for p in pinned], dict=True)
    assert len(sol) == 1
    sol = {kk: sp.expand(vv) for kk, vv in sol[0].items()}
    check('a d=%d D=%d pinned jets are polynomials in r' % (m + 1, D),
          all(sp.fraction(sp.together(vv))[1].free_symbols.isdisjoint({r}) for vv in sol.values()))
    fp = f.subs(sol)
    vars_ = (x,) + ys
    dets = []
    for xv in (-r / 2, r / 2):
        a_raw = at(sp.diff(fp, x, 2), xv)
        b_raw = [at(sp.diff(fp, x, ys[i]), xv) for i in range(m)]
        check('b d=%d D=%d alpha, beta at x=%s vanish at r=0' % (m + 1, D, xv),
              a_raw.subs(r, 0) == 0 and all(bb.subs(r, 0) == 0 for bb in b_raw))
        alpha = trunc(a_raw / r, r, R)
        beta = sp.Matrix(m, 1, lambda i, _: trunc(b_raw[i] / r, r, R))
        Ai = sp.Matrix(m, m, lambda i, j: trunc(at(sp.diff(fp, ys[i], ys[j]), xv), r, R))
        dK = trunc(alpha * Ai.det() - r * (beta.T * Ai.adjugate() * beta)[0, 0], r, R)
        if m == 1:
            Hfull = sp.Matrix(2, 2, lambda i, j: at(sp.diff(fp, vars_[i], vars_[j]), xv))
            direct = trunc(sp.cancel(Hfull.det() / r), r, R)
            check('b d=2 D=%d bordered formula = det(H_i)/r at x=%s' % (D, xv), sp.expand(direct - dK) == 0)
        dets.append(dK)
    Omega = trunc(-dets[0] * dets[1], r, 3)
    O = [sp.expand(Omega.coeff(r, j)) for j in range(4)]

    def jet(a, beta):
        return c[(a,) + tuple(beta)]

    def two(i, j):
        return [(1 if t == i else 0) + (1 if t == j else 0) for t in range(m)]

    A = sp.Matrix(m, m, lambda i, j: jet(0, two(i, j)))
    Bm = sp.Matrix(m, m, lambda i, j: jet(1, two(i, j)))
    Cm = sp.Matrix(m, m, lambda i, j: jet(2, two(i, j)))
    gam = sp.Matrix(m, 1, lambda i, _: jet(2, unit(i)))
    eta = sp.Matrix(m, 1, lambda i, _: jet(3, unit(i)))
    f4, f5 = jet(4, zero), jet(5, zero)
    tt = sp.Symbol('tt')
    Delta, J = A.det(), A.adjugate()
    dB = sp.expand(sp.diff((A + tt * Bm).det(), tt).subs(tt, 0))
    dC = sp.expand(sp.diff((A + tt * Cm).det(), tt).subs(tt, 0))
    dBB = sp.expand(sp.diff((A + tt * Bm).det(), tt, 2).subs(tt, 0))
    JB = sp.diff((A + tt * Bm).adjugate(), tt).subs(tt, 0)
    Y = f4 / 12 * Delta - sp.Rational(1, 4) * (gam.T * J * gam)[0, 0]
    Yr = Y + 3 * k * dB  # Y_r of (S10.16) under the pins: tr(adj(A) B) = dB
    Ups1 = Y + 3 * k * dB
    Ups2 = sp.Rational(3, 4) * k * (dC + dBB) + f4 / 24 * dB + f5 / 120 * Delta \
        - sp.Rational(1, 12) * (gam.T * J * eta)[0, 0] - sp.Rational(1, 8) * (gam.T * JB * gam)[0, 0]
    X2 = sp.expand(12 * k * Delta * Ups2 - Ups1 ** 2)
    tag = 'd=%d D=%d' % (m + 1, D)
    check('c %s (S10.16) det K_M = -6k Delta + r Y_r + O(r^2)' % tag, sp.expand(trunc(dets[0], r, 1) - (-6 * k * Delta + r * Yr)) == 0)
    check('c %s (S10.16) det K_S = 6k Delta + r Y_r + O(r^2)' % tag, sp.expand(trunc(dets[1], r, 1) - (6 * k * Delta + r * Yr)) == 0)
    check('d %s (S10.41) Omega_0 = 36 k^2 Delta^2' % tag, sp.expand(O[0] - 36 * k ** 2 * Delta ** 2) == 0)
    check('d %s (S10.41) Omega_1 = 0' % tag, O[1] == 0)
    check('d %s (S10.41) Omega_2 at k=0 = -Y^2' % tag, sp.expand(O[2].subs(k, 0) + Y ** 2) == 0)
    check('e %s (S11.23) Omega_2 = X_2' % tag, sp.expand(O[2] - X2) == 0)
    return O


for m, Ds in ((1, (6, 7, 8)), (2, (6, 7))):
    prev = None
    for D in Ds:
        O = run(m, D)
        if prev is not None:
            check('f d=%d coefficients of r^0, r^1, r^2 unchanged from D=%d to D=%d' % (m + 1, D - 1, D),
                  all(sp.expand(a - b_) == 0 for a, b_ in zip(prev[:3], O[:3])))
        prev = O

print('N_CHECKS', len(ok))
print('ALL_CHECKS_PASS')
