"""Exact checks for the finite-scale estimates of supplement S5 (checkpoint draft 9).

A polynomial field of degree N = 6 at the pair midpoint x = 0, with free Taylor
coefficients a_ijk = d_t^i d_v^j d_w^k f(0) in an orthonormal frame (t, v, w), is
subjected to the exact finite-r pair pins: grad f = 0 at M = (-r/2,0,0) and
S = (r/2,0,0), f(M) = b and f(S) = b - kappa r^3/6.  The eight contact
coordinates P = (a000, a100, a200, a300, a010, a110, a001, a101) are eliminated
through the pins.  Every scaled variable is then a combination of the other
coefficients (the "random part") plus a combination of the pin targets (the
"pin part", obtained with all other coefficients set to zero).

It checks the following.

1. Singular-near chart (Lemma "finite-scale singular-near frame").  With
   y = x + s(c t + rho v), eps^2 = rho^2 + s^2, (rho, s) = eps (cos th, sin th),
   r = eta s, lambda = 1 - eta^2/4 and both signs of c:
   (o)   with r = eta s, the coefficients of V0..V3 have no negative power of s;
   (i)   the random parts of V0..V3 vanish at eps = 0;
   (ii)  their eps^1 terms satisfy a0 + beta a1 = 0 with
         beta = sign(c) (4 + 3 eta^2)/48;
   (iii) the limits of tilde V0..tilde V3 = eps^-2 (V0 + beta V1), eps^-1 V1,
         eps^-1 V2, eps^-1 V3 are exactly eq:selection-Wlim, and the
         coefficients of the sixth-order derivatives vanish in the limit;
   (iv)  the pin part of V1 is kappa lambda + O(eps^2), and that of V0 is
         -sign(c) kappa eta^2/6 + O(eps^2).
2. Collar chart away from the endpoints (Lemma "finite-scale collar frames",
   part (b)).  With y = x + r(X t + rho v), eps^2 = rho^2 + r^2 and
   (rho, r) = eps (cos th, sin th): the coefficients of G have no negative
   power of r; the random parts of G vanish at eps = 0;
   the eps^1 terms of G2, G3 are eq:selection-Glim; the pin part of G1 is
   kappa (X^2 - 1/4) + O(eps^2) and the pin parts of G2, G3 vanish.
3. Collision charts (part (c)).  With y = p + r tau (cos a t + sin a v),
   p in {M, S}, eps'^2 = sin^2 a + r^2, and tilde G = N below: the
   coefficients of N have no negative power of r or tau; at r = 0 the random
   part of N is sin a ((tau cos a + vs/2) f_ttv + (tau/2) sin a f_tvv, f_vv,
   f_vw); the random parts of N vanish at eps' = 0; the eps'^1 terms of N2, N3
   are eq:selection-Nlim; the pin part of N1 is kappa (tau + vs sigma) +
   O(eps'^2), with vs = chi = -1 at M, +1 at S and sigma = sign(cos a); the
   pin parts of N2, N3 vanish.
4. The corrected value: under the pins, f(x) - (f(M) + f(S))/2 =
   (r^4/384) f_tttt(x) + O(r^6).
5. Lemma "exact triple-contact elimination": the kappa^6 coefficient of
   det H_M det H_S det H_y is (4c^2 - eta^2)^6 / (64 c^2 rho^2)^3, and at
   c = 1/2, rho = sqrt(3)/2, eta = 1, q_M = q_S = 0 the cubic
   kappa (X - 1)(2X + 1)^2 / 12 satisfies the ten conditions and all three
   Hessian determinants vanish.
6. A numerical cross-check with the exact planar kernel exp(-|z|^2/2): the
   finite-scale det Cov(V | pins) / eps^10 at eps = 0.004, computed from exact
   kernel derivatives in 60-digit arithmetic, agrees to relative error 1e-3
   with det Cov(W0lim, ..., W3lim | P), at sample angles.

Requires SymPy and mpmath.  Run: python3 s5_finite_scale_check.py   (about a quarter of a minute)
"""
import sympy as sp
import mpmath as mp

N = 6
z1, z2, z3 = sp.symbols('z1 z2 z3')
r, b, kap = sp.symbols('r b kappa')
eps, th, eta, X, tau = sp.symbols('epsilon theta eta X tau', real=True)
IDX = [(i, j, n - i - j) for n in range(N + 1) for i in range(n + 1) for j in range(n + 1 - i)]
A = {t: sp.Symbol('a_%d%d%d' % t) for t in IDX}
F = sum(A[t] * z1**t[0] * z2**t[1] * z3**t[2]
        / (sp.factorial(t[0]) * sp.factorial(t[1]) * sp.factorial(t[2])) for t in IDX)
GRAD = [sp.diff(F, v) for v in (z1, z2, z3)]
PIDX = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1)]
FREE = [A[t] for t in IDX if t not in PIDX]


def at(expr, p):
    return sp.expand(expr.subs({z1: p[0], z2: p[1], z3: p[2]}, simultaneous=True))


M = (-r / 2, 0, 0)
S = (r / 2, 0, 0)
PIN_L = [at(F, M), at(F, S)] + [at(g, M) for g in GRAD] + [at(g, S) for g in GRAD]
PIN_R = [b, b - kap * r**3 / 6, 0, 0, 0, 0, 0, 0]
SOL = sp.solve([sp.Eq(l, q) for l, q in zip(PIN_L, PIN_R)], [A[t] for t in PIDX], dict=True)[0]

RESULTS = []


def report(name, ok):
    RESULTS.append(ok)
    print('%s -> %s' % (name, 'PASS' if ok else 'FAIL'))


def split(expr):
    expr = sp.expand(expr)
    co = {}
    for fv in FREE:
        c = expr.coeff(fv)
        if c != 0:
            co[fv] = c
    pin = sp.expand(expr.subs({fv: 0 for fv in FREE}))
    return co, pin


def eps_poly(e, sq_pair=None, order=2):
    """Truncated eps-expansion of e, which must be a polynomial in eps after the substitution."""
    if sq_pair is not None:
        e = e.subs(sq_pair[0], sq_pair[1])
    e = sp.expand(e)
    out = {}
    if not e.has(eps):
        return {0: sp.simplify(e)}
    p = sp.Poly(e, eps)
    for (k,), c in p.terms():
        if k <= order:
            out[k] = sp.simplify(out.get(k, 0) + c)
    return out


def coeff_table(exprs, sub, sq_pair, order=2):
    """For each expression: {k: {jet: coeff of eps^k}} for the random part, and the pin-part expansion."""
    tables, pins = [], []
    for e in exprs:
        co, pin = split(e.subs(SOL))
        tab = {}
        for fv, cf in co.items():
            for k, v in eps_poly(cf.subs(sub), sq_pair, order).items():
                if v != 0:
                    tab.setdefault(k, {})[fv] = v
        tables.append(tab)
        pins.append(eps_poly(pin.subs(sub), sq_pair, order))
    return tables, pins


def same(d1, d2):
    keys = set(d1) | set(d2)
    return all(sp.simplify(d1.get(k, 0) - d2.get(k, 0)) == 0 for k in keys)


a = lambda i, j, k: A[(i, j, k)]
lam = 1 - eta**2 / 4
ct, st = sp.cos(th), sp.sin(th)

# ---------------------------------------------------------------- 1. singular-near
s_, c_, rho_ = sp.symbols('s c rho')
y = (s_ * c_, s_ * rho_, 0)
fy, f1y, f2y, f3y = at(F, y), at(GRAD[0], y), at(GRAD[1], y), at(GRAD[2], y)
U0 = b - kap * r**3 / 12
V = [(fy - U0 - s_ * rho_ / 2 * f2y - s_ * c_ / 3 * f1y) / s_**3, f1y / s_**2, f2y / s_, f3y / s_]


def no_negative_power(exprs, subs, syms):
    for e in exprs:
        co, pin = split(e.subs(SOL))
        for c in list(co.values()) + [pin]:
            den = sp.denom(sp.together(sp.expand(c.subs(subs))))
            if any(den.has(x) for x in syms):
                return False
    return True


report('sing: with r = eta s the coefficients of V0..V3 have no negative power of s',
       no_negative_power(V, {r: eta * s_}, [s_]))
for sg in (1, -1):
    sq = sp.sqrt(1 - eps**2 * ct**2)
    sq_pair = (sq, 1 - eps**2 * ct**2 / 2 - eps**4 * ct**4 / 8)
    sub = {c_: sg * sq, s_: eps * st, r: eta * eps * st, rho_: eps * ct}
    tab, pins = coeff_table(V, sub, sq_pair)
    report('sing, c sign %+d: random parts of V0..V3 vanish at eps=0' % sg,
           all(0 not in t for t in tab))
    beta = sg * (4 + 3 * eta**2) / 48
    a0, a1 = tab[0].get(1, {}), tab[1].get(1, {})
    report('sing, c sign %+d: eps^1 terms satisfy a0 + beta a1 = 0, beta = sign(c)(4+3eta^2)/48' % sg,
           all(sp.simplify(a0.get(k, 0) + beta * a1.get(k, 0)) == 0 for k in set(a0) | set(a1)))
    W0 = {k: sp.simplify(tab[0].get(2, {}).get(k, 0) + beta * tab[1].get(2, {}).get(k, 0))
          for k in set(tab[0].get(2, {})) | set(tab[1].get(2, {}))}
    W0 = {k: v for k, v in W0.items() if v != 0}
    W0x = {a(1, 2, 0): -sg * lam / 8 * ct**2, a(3, 1, 0): -sg * lam / 8 * lam / 3 * st * ct,
           a(5, 0, 0): -sg * lam / 8 * lam**2 / 60 * st**2}
    W1x = {a(2, 1, 0): sg * ct, a(4, 0, 0): sg * lam / 6 * st}
    W2x = {a(0, 2, 0): ct, a(2, 1, 0): lam / 2 * st}
    W3x = {a(0, 1, 1): ct, a(2, 0, 1): lam / 2 * st}
    report('sing, c sign %+d: W0 limit is eq:selection-Wlim' % sg, same(W0, W0x))
    report('sing, c sign %+d: W1 limit is eq:selection-Wlim' % sg, same(tab[1].get(1, {}), W1x))
    report('sing, c sign %+d: W2 limit is eq:selection-Wlim' % sg, same(tab[2].get(1, {}), W2x))
    report('sing, c sign %+d: W3 limit is eq:selection-Wlim' % sg, same(tab[3].get(1, {}), W3x))
    sixth = [A[t] for t in IDX if sum(t) == 6]
    report('sing, c sign %+d: sixth-order derivatives absent from the limits' % sg,
           all(v not in W0 for v in sixth) and all(v not in tab[i].get(1, {}) for i in (1, 2, 3) for v in sixth))
    report('sing, c sign %+d: pin part of V1 is kappa*lambda + O(eps^2)' % sg,
           sp.simplify(pins[1].get(0, 0) - kap * lam) == 0 and sp.simplify(pins[1].get(1, 0)) == 0)
    report('sing, c sign %+d: pin part of V0 is -sign(c) kappa eta^2/6 + O(eps^2)' % sg,
           sp.simplify(pins[0].get(0, 0) + sg * kap * eta**2 / 6) == 0 and sp.simplify(pins[0].get(1, 0)) == 0)
    report('sing, c sign %+d: pin parts of V2, V3 vanish to order eps' % sg,
           all(sp.simplify(pins[i].get(k, 0)) == 0 for i in (2, 3) for k in (0, 1)))

# ---------------------------------------------------------------- 2. collar, away from the endpoints
rhoc = sp.Symbol('rhoc')
y = (r * X, r * rhoc, 0)
G = [at(GRAD[0], y) / r**2, at(GRAD[1], y) / r, at(GRAD[2], y) / r]
report('collar: the coefficients of G have no negative power of r', no_negative_power(G, {}, [r]))
sub = {rhoc: eps * ct, r: eps * st}
tab, pins = coeff_table(G, sub, None)
report('collar: random parts of G vanish at eps=0', all(0 not in t for t in tab))
q = X**2 - sp.Rational(1, 4)
report('collar: G2 limit is eq:selection-Glim', same(tab[1].get(1, {}), {a(0, 2, 0): ct, a(2, 1, 0): q / 2 * st}))
report('collar: G3 limit is eq:selection-Glim', same(tab[2].get(1, {}), {a(0, 1, 1): ct, a(2, 0, 1): q / 2 * st}))
report('collar: pin part of G1 is kappa (X^2-1/4) + O(eps^2)',
       sp.simplify(pins[0].get(0, 0) - kap * q) == 0 and sp.simplify(pins[0].get(1, 0)) == 0)
report('collar: pin parts of G2, G3 vanish identically',
       all(sp.simplify(split(g.subs(SOL))[1]) == 0 for g in G[1:]))

# ---------------------------------------------------------------- 3. collision charts
CA, SA = sp.symbols('CA SA')
for pname, p, vs in (('M', M, -1), ('S', S, 1)):
    y = (p[0] + r * tau * CA, r * tau * SA, 0)
    Nv = [at(GRAD[0], y) / (r**2 * tau), at(GRAD[1], y) / (r * tau), at(GRAD[2], y) / (r * tau)]
    report('collision at %s: the coefficients of N have no negative power of r or tau' % pname,
           no_negative_power(Nv, {}, [r, tau]))
    lim0 = []
    for nv in Nv:
        co, pin = split(nv.subs(SOL))
        lim0.append({fv: sp.expand(c).subs(r, 0) for fv, c in co.items()})
    exp0 = [{a(2, 1, 0): SA * (tau * CA + sp.Rational(vs, 2)), a(1, 2, 0): SA**2 * tau / 2},
            {a(0, 2, 0): SA}, {a(0, 1, 1): SA}]
    report('collision at %s: at r = 0 the random part of N is sin a ((tau cos a + chi/2) f_ttv + (tau/2) sin a f_tvv, f_vv, f_vw)' % pname,
           all(same({k: v for k, v in l.items() if v != 0}, e) for l, e in zip(lim0, exp0)))
    for sg in (1, -1):
        sq = sp.sqrt(1 - eps**2 * ct**2)
        sq_pair = (sq, 1 - eps**2 * ct**2 / 2 - eps**4 * ct**4 / 8)
        sub = {CA: sg * sq, SA: eps * ct, r: eps * st}
        tab, pins = coeff_table(Nv, sub, sq_pair)
        k_ = (tau + vs * sg) / 2
        report('collision at %s, cos sign %+d: random parts of N vanish at eps=0' % (pname, sg),
               all(0 not in t for t in tab))
        report('collision at %s, cos sign %+d: N2 limit is eq:selection-Nlim' % (pname, sg),
               same(tab[1].get(1, {}), {a(0, 2, 0): ct, a(2, 1, 0): k_ * st}))
        report('collision at %s, cos sign %+d: N3 limit is eq:selection-Nlim' % (pname, sg),
               same(tab[2].get(1, {}), {a(0, 1, 1): ct, a(2, 0, 1): k_ * st}))
        report('collision at %s, cos sign %+d: pin part of N1 is kappa(tau + vs sigma) + O(eps^2)' % (pname, sg),
               sp.simplify(pins[0].get(0, 0) - kap * (tau + vs * sg)) == 0 and sp.simplify(pins[0].get(1, 0)) == 0)
        report('collision at %s, cos sign %+d: pin parts of N2, N3 vanish' % (pname, sg),
               all(sp.simplify(pins[i].get(k, 0)) == 0 for i in (1, 2) for k in (0, 1)))

# ---------------------------------------------------------------- 4. corrected value
diff = sp.expand(SOL[A[(0, 0, 0)]] - (b - kap * r**3 / 12))
co, pin = split(diff)
ok = sp.simplify(pin) == 0
for fv, c in co.items():
    pc = sp.Poly(sp.expand(c), r)
    lowest = min(k for (k,), cc in pc.terms())
    if fv == A[(4, 0, 0)]:
        ok = ok and lowest == 4 and sp.simplify(pc.coeff_monomial(r**4) - sp.Rational(1, 384)) == 0
    else:
        ok = ok and lowest >= 6
report('pins give f(x) - (f(M)+f(S))/2 = (r^4/384) f_tttt + O(r^6)', ok)

# ---------------------------------------------------------------- 5. the elimination: degree in kappa
cc, rr, ee, qM, qS, kk = sp.symbols('c rho eta q_M q_S kappa', real=True)
dHM = -(kk**2 * (4 * cc**2 - ee**2)**2 + 4 * kk * rr**2 * ((12 * cc**2 + ee**2) * qM + (4 * cc**2 - ee**2) * qS)
        + 4 * rr**4 * (qM - qS)**2) / (64 * cc**2 * rr**2)
dHS = -(kk**2 * (4 * cc**2 - ee**2)**2 - 4 * kk * rr**2 * ((4 * cc**2 - ee**2) * qM + (12 * cc**2 + ee**2) * qS)
        + 4 * rr**4 * (qM - qS)**2) / (64 * cc**2 * rr**2)
Ny = (ee * kk**2 * (4 * cc**2 - ee**2)**2 + 4 * kk * rr**2 * (4 * cc**2 + ee**2) * (ee * (qM - qS) - 2 * cc * (qM + qS))
      - 16 * cc * rr**4 * (qM**2 - qS**2) + 4 * ee * rr**4 * (qM - qS)**2)
dHy = Ny / (64 * cc**2 * ee * rr**2)
prod = sp.expand(dHM * dHS * dHy)
report('elimination: kappa^6 coefficient of the product is (4c^2-eta^2)^6/(64c^2rho^2)^3',
       sp.simplify(prod.coeff(kk, 6) - (4 * cc**2 - ee**2)**6 / (64 * cc**2 * rr**2)**3) == 0
       and all(sp.simplify(prod.coeff(kk, j)) == 0 for j in (7, 8, 9)))
Xs, Ys = sp.symbols('X Y', real=True)
pcub = kk * (Xs - 1) * (2 * Xs + 1)**2 / 12
yv = (sp.Rational(1, 2), sp.sqrt(3) / 2)
conds = [sp.diff(pcub, v).subs({Xs: P[0], Ys: P[1]}) for P in ((-sp.Rational(1, 2), 0), (sp.Rational(1, 2), 0), yv)
         for v in (Xs, Ys)]
conds += [pcub.subs({Xs: -sp.Rational(1, 2)}), pcub.subs({Xs: sp.Rational(1, 2)}) + kk / 6,
          sp.diff(pcub, Ys, 2).subs({Xs: -sp.Rational(1, 2)}), sp.diff(pcub, Ys, 2).subs({Xs: sp.Rational(1, 2)})]
Hm = sp.hessian(pcub, (Xs, Ys))
dets = [sp.simplify(Hm.subs({Xs: P[0], Ys: P[1]}).det()) for P in ((-sp.Rational(1, 2), 0), (sp.Rational(1, 2), 0), yv)]
report('elimination: at c=1/2, rho=sqrt3/2, eta=1, q=0 the cubic satisfies the ten conditions',
       all(sp.simplify(e) == 0 for e in conds))
report('elimination: and its three Hessian determinants vanish', all(d == 0 for d in dets))
report('elimination: the formulas give zero there as well',
       all(sp.simplify(e.subs({cc: sp.Rational(1, 2), rr: sp.sqrt(3) / 2, ee: 1, qM: 0, qS: 0})) == 0
           for e in (dHM, dHS, dHy)))

# ---------------------------------------------------------------- 6. planar numerical cross-check
mp.mp.dps = 60


def dK(alpha, zz):
    v = mp.mpf(1)
    for k, zi in zip(alpha, zz):
        v *= (-1)**k * mp.hermite(k, zi / mp.sqrt(2)) / mp.power(mp.sqrt(2), k) * mp.e**(-zi**2 / 2)
    return v


def cv(F1, F2):
    tot = mp.mpf(0)
    for (p1, al, c1) in F1:
        for (p2, be, c2) in F2:
            tot += c1 * c2 * (-1)**sum(be) * dK(tuple(x + y for x, y in zip(al, be)),
                                                [x - y for x, y in zip(p1, p2)])
    return tot


def condcov(Vs, Ps):
    n, m = len(Vs), len(Ps)
    AA, BB, PP = mp.matrix(n, n), mp.matrix(n, m), mp.matrix(m, m)
    for i in range(n):
        for j in range(n):
            AA[i, j] = cv(Vs[i], Vs[j])
        for j in range(m):
            BB[i, j] = cv(Vs[i], Ps[j])
    for i in range(m):
        for j in range(m):
            PP[i, j] = cv(Ps[i], Ps[j])
    return AA - BB * mp.inverse(PP) * BB.T


def ev(p, al=(0, 0, 0), c=1):
    return [(p, al, mp.mpf(c))]


def comb(*terms):
    out = []
    for coef, fn in terms:
        out += [(p, al, c * coef) for (p, al, c) in fn]
    return out


def numeric_det(epsv, thv, etav):
    rhov, sv = epsv * mp.cos(thv), epsv * mp.sin(thv)
    rv = etav * sv
    cv_ = mp.sqrt(1 - rhov**2)
    Mp, Sp, yp = (-rv / 2, 0, 0), (rv / 2, 0, 0), (sv * cv_, sv * rhov, 0)
    E3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    V0 = comb((sv**-3, ev(yp)), (-sv**-3 / 2, ev(Mp)), (-sv**-3 / 2, ev(Sp)),
              (-sv**-2 * rhov / 2, ev(yp, (0, 1, 0))), (-sv**-2 * cv_ / 3, ev(yp, (1, 0, 0))))
    Vs = [V0, comb((sv**-2, ev(yp, (1, 0, 0)))), comb((sv**-1, ev(yp, (0, 1, 0)))), comb((sv**-1, ev(yp, (0, 0, 1))))]
    Ps = [ev(Mp), ev(Sp)] + [ev(Mp, e) for e in E3] + [ev(Sp, e) for e in E3]
    return mp.det(condcov(Vs, Ps)) / epsv**10


def limit_det(thv, etav):
    lv = 1 - etav**2 / 4
    c1, s1 = mp.cos(thv), mp.sin(thv)
    W = [{(1, 2, 0): -lv / 8 * c1**2, (3, 1, 0): -lv / 8 * lv / 3 * s1 * c1, (5, 0, 0): -lv / 8 * lv**2 / 60 * s1**2},
         {(2, 1, 0): c1, (4, 0, 0): lv / 6 * s1},
         {(0, 2, 0): c1, (2, 1, 0): lv / 2 * s1},
         {(0, 1, 1): c1, (2, 0, 1): lv / 2 * s1}]
    Fs = [[((0, 0, 0), al, mp.mpf(1) * co) for al, co in w.items()] for w in W]
    Ps = [ev((0, 0, 0), al) for al in PIDX]
    return mp.det(condcov(Fs, Ps))


for thd, etav in ((10, mp.mpf('0.3')), (45, mp.mpf('0.6')), (80, mp.mpf('0.9')), (90, mp.mpf('0.5'))):
    num = numeric_det(mp.mpf('0.004'), mp.radians(thd), etav)
    lim = limit_det(mp.radians(thd), etav)
    rel = abs(num / lim - 1)
    report('planar numerics: theta=%d deg, eta=%s: det Cov(V|pins)/eps^10 = %s vs limit %s (rel. err. %s)'
           % (thd, mp.nstr(etav, 2), mp.nstr(num, 8), mp.nstr(lim, 8), mp.nstr(rel, 2)), lim > 0 and rel < mp.mpf('1e-3'))

print('ALL PASS' if all(RESULTS) else 'SOME CHECKS FAILED')
