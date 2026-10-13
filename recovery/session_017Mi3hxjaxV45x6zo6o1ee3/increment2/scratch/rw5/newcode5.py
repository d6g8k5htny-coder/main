

# ---------------- exact polynomial algebra (Math-#199 cone_moment_d4.py, unchanged)
POLY_ALGEBRA_PLACEHOLDER


def plin(coeffs):
    """sum_i coeffs[i] * variable_i."""
    out = {}
    for i, c in enumerate(coeffs):
        if c:
            out = padd(out, pscale(pvar(i, len(coeffs)), Fr(c)))
    return out


def lpadd(u, v):
    n = max(len(u), len(v))
    return [(u[i] if i < len(u) else Fr(0)) + (v[i] if i < len(v) else Fr(0)) for i in range(n)]


def lpmul(u, v):
    if not u or not v:
        return []
    w = [Fr(0)] * (len(u) + len(v) - 1)
    for i, a in enumerate(u):
        for j, b in enumerate(v):
            w[i + j] += a * b
    return w


# ---------------- the reference birth integral as a one-dimensional integral in the total trace (NOTE section 2)
def Jk_closed(k):
    """J_k(s) = int_0^s q^k e^{-q^2/8} dq for odd k: J_k = B_k + C_k(s) e^{-s^2/8} with B_k rational and C_k a rational polynomial
    (list by power): J_1 = 4 - 4 e^{-s^2/8}, J_{k+2} = -4 s^{k+1} e^{-s^2/8} + 4 (k+1) J_k."""
    if k % 2 != 1:
        raise ValueError("even power of q")
    B, C, j = Fr(4), [Fr(-4)], 1
    while j < k:
        C2 = [4 * (j + 1) * c for c in C] + [Fr(0)] * (j + 2 - len(C))
        C2[j + 1] += -4
        B, C, j = 4 * (j + 1) * B, C2, j + 2
    return B, C


def moments(gamma, nmax):
    """M_j(T) = int_0^T s^j e^{-gamma s^2} ds = alpha_j + rho_j(T) e^{-gamma T^2} + beta_j E_gamma(T), E_gamma = (1/2) sqrt(pi/gamma)
    erf(sqrt(gamma) T): M_0 = E_gamma, M_1 = (1 - e^{-gamma T^2})/(2 gamma), M_{j+2} = ((j+1) M_j - T^{j+1} e^{-gamma T^2})/(2 gamma)."""
    g = Fr(gamma)
    M = [(Fr(0), [], Fr(1)), (1 / (2 * g), [-1 / (2 * g)], Fr(0))]
    for j in range(nmax - 1):
        a, r, b = M[j]
        f = Fr(j if MUT == "moment-recurrence" else j + 1)
        r2 = [x * f / (2 * g) for x in r] + [Fr(0)] * (j + 2 - len(r))
        r2[j + 1] += -1 / (2 * g)
        M.append((f * a / (2 * g), r2, f * b / (2 * g)))
    return M


F_MEMO = {}


def F_pieces(m):
    if m not in F_MEMO:
        F_MEMO[m] = F_pieces_build(m)
    return F_MEMO[m]


def F_pieces_build(m):
    """The inner integrals in closed form.  m = 3: on the ordered sector mu_3 = -a, mu_2 = -a-p, mu_1 = -a-p-q put T = 3a + 2p + q
    (minus the trace), s = 2p + q, q (Jacobian 1/6, region T > s > q > 0); the integrand is P(T, s, q) e^{-T^2/24 - s^2/24 - q^2/8}
    with P = a^2 (a+p)^2 (a+p+q)^2 pq(p+q), a = (T-s)/3, p = (s-q)/2.  m = 2: mu_2 = -a, mu_1 = -a-p, T = 2a + p, p (Jacobian 1/2),
    P = (T^2 - p^2)^2 p/16, exponent 3T^2/40 + p^2/8.  Returns ({kind: polynomial in T}, outer gamma) with F_m(T) = sum of
    poly * [1 | e^{-g T^2} | E_g(T)] by kind ('1',), ('e', g), ('E', g)."""
    out = {}

    def acc(key, poly):
        out[key] = lpadd(out.get(key, []), poly)
    if m == 3:
        T, s, q = pvar(0, 3), pvar(1, 3), pvar(2, 3)
        a, p = plin([Fr(1, 3), Fr(-1, 3), 0]), plin([0, Fr(1, 2), Fr(-1, 2)])
        ap = padd(a, p)
        apq = padd(ap, q)
        P = pmul(pmul(pmul(a, a), pmul(ap, ap)), pmul(apq, apq))
        P = pmul(P, pmul(pmul(p, q), padd(p, q)))
        U, V = {}, {}
        for (i, j, k), c in P.items():
            Bk, Ck = Jk_closed(k)                                  # raises on an even power of q
            U = padd(U, {(i, j): c * Bk})
            V = padd(V, pmul({(i, j): c}, {(0, n): x for n, x in enumerate(Ck) if x}))
        for W_, gam in ((U, Fr(1, 24)), (V, Fr(1, 24) + Fr(1, 8))):
            Ms = moments(gam, max(j for (_, j) in W_))
            for (i, j), c in W_.items():
                a_, r_, b_ = Ms[j]
                Ti = [Fr(0)] * i + [c]
                acc(("1",), lpmul(Ti, [a_]))
                acc(("e", gam), lpmul(Ti, r_))
                acc(("E", gam), lpmul(Ti, [b_]))
        return out, Fr(1, 24)
    if m == 2:
        T, p = pvar(0, 2), pvar(1, 2)
        d = padd(pmul(T, T), pscale(pmul(p, p), Fr(-1)))
        P = pscale(pmul(pmul(d, d), p), Fr(1, 16))
        for (i, k), c in P.items():
            Bk, Ck = Jk_closed(k)
            acc(("1",), lpmul([Fr(0)] * i + [c], [Bk]))
            acc(("e", Fr(1, 8)), lpmul([Fr(0)] * i + [c], Ck))
        return out, Fr(3, 40)
    if m == 4:
        T, w, s, q = (pvar(i, 4) for i in range(4))
        a = plin([Fr(1, 4), Fr(-1, 4), 0, 0])
        p1 = plin([0, Fr(1, 3), Fr(-1, 3), 0])
        p2 = plin([0, 0, Fr(1, 2), Fr(-1, 2)])
        mus, run = [pscale(a, Fr(-1))], a
        for p in (p1, p2, q):
            run = padd(run, p)
            mus.append(pscale(run, Fr(-1)))
        P = {(0, 0, 0, 0): Fr(1)}
        for mu in mus:
            P = pmul(P, pmul(mu, mu))
        for i in range(4):
            for j in range(i + 1, 4):
                P = pmul(P, padd(mus[i], pscale(mus[j], Fr(-1))))       # mus[i] > mus[j]
        U, V = {}, {}
        for (i, j, k, l), c in P.items():
            Bk, Ck = Jk_closed(l)                                      # raises on an even power of q
            U = padd(U, {(i, j, k): c * Bk})
            V = padd(V, pmul({(i, j, k): c}, {(0, 0, n): x for n, x in enumerate(Ck) if x}))
        mid = {}                                                       # the s-layer over (0, w): pieces in (T, w)
        for W_, gam in ((U, Fr(1, 24)), (V, Fr(1, 24) + Fr(1, 8))):
            Ms = moments(gam, max(k for (_, _, k) in W_))
            for (i, j, k), c in W_.items():
                a_, r_, b_ = Ms[k]
                if a_:
                    mid[("1",)] = padd(mid.get(("1",), {}), {(i, j): c * a_})
                for n, x in enumerate(r_):
                    if x:
                        mid[("e", gam)] = padd(mid.get(("e", gam), {}), {(i, j + n): c * x})
                if b_:
                    mid[("E", gam)] = padd(mid.get(("E", gam), {}), {(i, j): c * b_})
        aw = Fr(1, 48)                                                 # the w-layer over (0, T) against e^{-w^2/48}
        for key, poly in mid.items():
            for (i, j), c in poly.items():
                if key[0] == "1":
                    layer = moment_dict(aw, j)
                elif key[0] == "e":
                    layer = moment_dict(aw + key[1], j)
                else:
                    layer = L_dict(aw, key[1], j)
                for kind, pl in layer.items():
                    acc(kind, lpmul([Fr(0)] * i + [c], pl))
        if MUT == "layer-dropped":                                     # a cancelled layer silently not derived
            out.pop(("E", Fr(3, 16)), None)
        return out, Fr(3, 112)
    raise ValueError("m must be 2, 3 or 4")


def moment_dict(gamma, j):
    """M_j(gamma, T) = int_0^T w^j e^{-gamma w^2} dw by kind: ('1',) constant, ('e', gamma) e^{-gamma T^2}, ('E', gamma) E_gamma(T)."""
    a_, r_, b_ = moments(gamma, max(j, 1))[j]
    out = {}
    if a_:
        out[("1",)] = [a_]
    if any(r_):
        out[("e", gamma)] = list(r_)
    if b_:
        out[("E", gamma)] = [b_]
    return out


L_MEMO = {}


def L_dict(al, g, j):
    """L_j(T) = int_0^T w^j e^{-al w^2} E_g(w) dw, E_g(w) = (1/2) sqrt(pi/g) erf(sqrt(g) w): L_0 = Kfun_{al,g}(T) and, by parts,
    L_j = -(1/(2 al)) T^{j-1} e^{-al T^2} E_g(T) + ((j-1)/(2 al)) L_{j-2} + (1/(2 al)) M_{j-1}(al + g, T)
    (the last coefficient is (1/2) sqrt(pi/g) . sqrt(g)/(al sqrt(pi)) = 1/(2 al), rational)."""
    key = (al, g, j)
    if key in L_MEMO:
        return L_MEMO[key]
    if j == 0:
        res = {("K", al, g): [Fr(1)]}
    else:
        res = {}

        def add(kind, poly):
            res[kind] = lpadd(res.get(kind, []), poly)
        add(("eE", al, g), [Fr(0)] * (j - 1) + [(1 if MUT == "parts-sign" else -1) / (2 * al)])
        if j >= 2:
            for kind, pl in L_dict(al, g, j - 2).items():
                add(kind, [x * Fr(j - 1) / (2 * al) for x in pl])
        for kind, pl in moment_dict(al + g, j - 1).items():
            add(kind, [x / (2 * al) for x in pl])
    L_MEMO[key] = res
    return res


def trim(poly):
    poly = list(poly)
    while poly and poly[-1] == 0:
        poly.pop()
    return poly


def g_terms(m, gamma_out=None):
    """g_m(T) = e^{-gamma_out T^2} F_m(T) as terms (const, polynomial, gamma, kappa): const * poly(T) e^{-gamma T^2} [erf(kappa T)]."""
    pieces, g_ref = F_pieces(m)
    gout = g_ref if gamma_out is None else gamma_out
    terms = []
    for key, poly in sorted(pieces.items(), key=lambda kv: str(kv[0])):
        poly = trim(poly)
        if not poly:
            continue
        if key[0] == "1":
            terms.append((Iv(ONE), poly, Iv.frac(gout), None))
        elif key[0] == "e":
            terms.append((Iv(ONE), poly, Iv.frac(gout + key[1]), None))
        elif key[0] == "E":
            terms.append(((PI / Iv.frac(key[1])).sqrt() / 2, poly, Iv.frac(gout), Iv.frac(key[1]).sqrt()))
        elif key[0] == "eE":
            terms.append(((PI / Iv.frac(key[2])).sqrt() / 2, poly, Iv.frac(gout + key[1]), Iv.frac(key[2]).sqrt()))
        else:
            raise ValueError("an Owen-type term int_0^T e^{-al w^2} E_g(w) dw survived the w-layer")
    return terms


def F_float(pieces, T):
    tot = 0.0
    for key, poly in pieces.items():
        pv = sum(float(c) * T ** i for i, c in enumerate(poly))
        if key[0] == "1":
            tot += pv
        elif key[0] == "e":
            tot += pv * math.exp(-float(key[1]) * T * T)
        elif key[0] == "E":
            g = float(key[1])
            tot += pv * 0.5 * math.sqrt(math.pi / g) * math.erf(math.sqrt(g) * T)
        elif key[0] == "eE":
            al, g = float(key[1]), float(key[2])
            tot += pv * math.exp(-al * T * T) * 0.5 * math.sqrt(math.pi / g) * math.erf(math.sqrt(g) * T)
        elif any(poly):
            raise ValueError("an Owen-type term int_0^T e^{-al w^2} E_g(w) dw survived the w-layer")
    return tot


# ---------------- Taylor quadrature with Cauchy remainders (Math-#199 section 3; factors as functions, plus Phi(alpha - beta T))
def ipow(x, n):
    r = Iv(ONE)
    for _ in range(n):
        r = r * x
    return r


ERF_FAST = Decimal(12)


def erf_point_fast(x):
    """erf at a Decimal point: the toolkit series for |x| < 12; for |x| >= 12 the enclosure [1 - e^{-x^2}/(x sqrt(pi)), 1] (with sign),
    since 0 < erfc(x) < e^{-x^2}/(x sqrt(pi)) for x > 0 (here below 1.5e-64)."""
    if abs(x) < ERF_FAST:
        return erf_point(x)
    X = Iv(abs(x))
    t = (-(X.sq())).exp() / (X * SQRTPI)
    v = Iv((1 - t).lo, ONE)
    return v if x > 0 else -v


def erf_fast(x):
    return Iv(erf_point_fast(x.lo).lo, erf_point_fast(x.hi).hi)


def Phi_fast(x):
    return (1 + erf_fast(x / SQRT2)) / 2


CACHE = {}


def gauss_coeffs(g, s0, K):
    """Taylor coefficients of exp(-g s^2) at s0: (n+1) y_{n+1} = -2 g (s0 y_n + y_{n-1})."""
    key = ("g", g.lo, g.hi, s0, K)
    if key in CACHE:
        return CACHE[key]
    S0 = Iv.frac(s0)
    y = [(-(g * S0.sq())).exp()]
    prev = Iv(ZERO)
    for n in range(K):
        nxt = -(2 * g) * (S0 * y[n] + prev) / (n + 1)
        prev = y[n]
        y.append(nxt)
    CACHE[key] = y
    return y


def gauss_bound(g, s0, rho):
    inner = max(Fr(0), abs(Fr(s0)) - Fr(rho))
    return (g * Iv.frac(rho * rho) - g * Iv.frac(inner * inner)).exp()   # |e^{-g z^2}| = e^{-g(x^2 - y^2)}, |y| <= rho


def erf_coeffs(kappa, s0, K):
    """erf(kappa s): value at s0, then (2 kappa/sqrt pi) times the Gaussian's coefficients divided by n + 1."""
    key = ("e", kappa.lo, kappa.hi, s0, K)
    if key in CACHE:
        return CACHE[key]
    y = gauss_coeffs(kappa.sq(), s0, K)
    pref = 2 * kappa / SQRTPI
    out = [erf_fast(kappa * Iv.frac(s0))] + [pref * y[n] / (n + 1) for n in range(K)]
    CACHE[key] = out
    return out


def erf_bound(kappa, s0, rho):
    return 2 * kappa / SQRTPI * Iv.frac(abs(Fr(s0)) + Fr(rho)) * (kappa.sq() * Iv.frac(Fr(rho) ** 2)).exp()


def poly_coeffs(c, s0, K):
    out = []
    for n in range(K + 1):
        v = Fr(0)
        for j, cj in enumerate(c):
            if j >= n:
                v += cj * math.comb(j, n) * Fr(s0) ** (j - n)
        out.append(Iv.frac(v))
    return out


def poly_bound(c, s0, rho):
    r = abs(Fr(s0)) + Fr(rho)
    return Iv.frac(sum(abs(cj) * r ** j for j, cj in enumerate(c)))


def t_mul(u, v):
    K = len(u) - 1
    out = []
    for n in range(K + 1):
        acc = Iv(ZERO)
        for i in range(n + 1):
            acc = acc + u[i] * v[n - i]
        out.append(acc)
    return out


def phi_aff_coeffs(alpha, beta, s0, K):
    """Phi(alpha - beta T) at T = s0 + x (alpha, beta real intervals, beta > 0): c_0 = Phi(w0), w0 = alpha - beta s0, and
    c_{n+1} = -(beta/sqrt(2 pi)) y_n/(n+1) with y the coefficients of exp(-(w0 - beta x)^2/2), (n+1) y_{n+1} = beta w0 y_n - beta^2 y_{n-1}."""
    w0 = alpha - beta * Iv.frac(s0)
    y = [(-(w0.sq()) / 2).exp()]
    prev = Iv(ZERO)
    b2 = beta.sq()
    for n in range(K):
        nxt = (beta * w0 * y[n] - b2 * prev) / (n + 1)
        prev = y[n]
        y.append(nxt)
    pref = beta / SQRT2PI
    return [Phi_fast(w0)] + [-(pref * y[n] / (n + 1)) for n in range(K)]


def phi_aff_bound(beta, rho):
    """|Phi(alpha - beta z)| <= Phi(w0) + |int_{w0}^{w} phi| <= 1 + beta rho e^{beta^2 rho^2/2}/sqrt(2 pi) on |z - s0| <= rho
    (the segment from the real point w0 has |Im t| <= beta rho, and |phi(t)| <= e^{(Im t)^2/2}/sqrt(2 pi))."""
    br = beta * Iv.frac(rho)
    return 1 + br * (br.sq() / 2).exp() / SQRT2PI


class Model:
    """int_0^inf g(T) Phi(alpha - beta T) dT for g = sum const * poly(T) e^{-gamma T^2} [erf(kappa T)] (entire).  On each cell of
    width CELL the point Taylor coefficients of g at the centre are computed once and contracted with the cell moments, so that an
    evaluation for given (alpha, beta) costs one product of coefficient vectors per cell.  The remainder of the order-K Taylor
    polynomial of the product is bounded by Cauchy's estimate at the centre, |c_n| <= M_g M_Phi / rho^n, summed geometrically;
    beyond T_CUT the integrand is bounded by |g| (0 <= Phi <= 1, |erf| <= 1) and incomplete gamma bounds."""

    def __init__(self, terms, K=K_ORDER, cell=CELL, T=T_CUT, rho=RHO):
        self.K, self.rho = K, rho
        half = cell / 2
        ncell = int(T / cell)
        if ncell * cell != T or Fr(half) >= Fr(rho):
            raise ValueError("bad quadrature parameters")
        mom = [Iv.frac(2 * half ** (n + 1) / (n + 1)) if n % 2 == 0 else Iv(ZERO) for n in range(K + 1)]
        rem_moment = Iv.frac(2 * half ** (K + 2) / (K + 2) / (1 - Fr(half) / Fr(rho)))
        rho_pow = Iv.frac(Fr(rho) ** (K + 1))
        self.cells, Rsum = [], Iv(ZERO)
        for c in range(ncell):
            s0 = (2 * c + 1) * half
            G, M = [Iv(ZERO)] * (K + 1), Iv(ZERO)
            for const, poly, gam, kap in terms:
                pc = t_mul(poly_coeffs(poly, s0, K), gauss_coeffs(gam, s0, K))
                Mb = poly_bound(poly, s0, rho) * gauss_bound(gam, s0, rho)
                if kap is not None:
                    pc = t_mul(pc, erf_coeffs(kap, s0, K))
                    Mb = Mb * erf_bound(kap, s0, rho)
                G = [x + const * y for x, y in zip(G, pc)]
                M = M + Iv(const.mag()) * Mb
            Gm = []
            for j in range(K + 1):
                acc = Iv(ZERO)
                for i in range(j % 2, K + 1 - j, 2):
                    acc = acc + G[i] * mom[i + j]
                Gm.append(acc)
            self.cells.append((s0, Gm))
            Rsum = Rsum + M / rho_pow * rem_moment
        self.Rsum = Rsum.hi
        tail = Iv(ZERO)
        for const, poly, gam, kap in terms:                  # int_T^inf t^j e^{-g t^2} dt = (1/2) g^{-(j+1)/2} Gamma((j+1)/2, g T^2)
            x = gam * Iv.frac(T * T)
            for j, cj in enumerate(poly):
                if cj == 0:
                    continue
                a = Fr(j + 1, 2)
                if a.denominator == 2:
                    xa1, gpow = ipow(x, int(a - Fr(1, 2))) * x.sqrt() / x, ipow(gam, int(a)) * gam.sqrt()
                else:
                    xa1, gpow = ipow(x, int(a) - 1), ipow(gam, int(a))
                up = xa1 * (-x).exp()                          # Gamma(a, x) <= x^(a-1) e^-x (a <= 1), / (1 - (a-1)/x) (a > 1)
                if a > 1:
                    up = up / (1 - Iv.frac(a - 1) / x)
                tail = tail + Iv(const.mag()) * Iv.frac(abs(cj)) * up / gpow / 2
        self.tail = tail.hi

    def integral(self, alpha=None, beta=None):
        tot = Iv(ZERO)
        if alpha is None:
            for _, Gm in self.cells:
                tot = tot + Gm[0]
            R = self.Rsum
        else:
            for s0, Gm in self.cells:
                Wc = phi_aff_coeffs(alpha, beta, s0, self.K)
                for j in range(self.K + 1):
                    tot = tot + Wc[j] * Gm[j]
            R = (Iv(self.Rsum) * phi_aff_bound(beta, self.rho)).hi
        if MUT == "remainder-dropped":
            R = ZERO
        R = CC.add(R, self.tail)
        return tot + Iv(R.copy_negate(), R)


MODELS = {}


def model(m, gamma_out=None):
    key = (m, gamma_out)
    if key not in MODELS:
        MODELS[key] = Model(g_terms(m, gamma_out))
    return MODELS[key]


Z_MEHTA = {2: 8 * SQRT2PI, 3: 48 * SQRT2 * PI, 4: 1536 * PI}   # Mehta's integral (Math-#199 section 1; Z_4 = 1536 pi, Math-#202)


class Birth:
    """I_B = kappa0/Z_m int_0^inf e^{-gamma_out T^2} F_m(T) [Phi(sl (b_+ - c T)) - Phi(sl (b_- - c T))] dT for the Gaussian pair
    f ~ N(0, v), g = mu f + sigma Z (A = Q - g I): conditioning on f = b and integrating the GOE shift exactly (NOTE section 2).
    Reference: v = 2/3, mu = 1, sigma = 0, i.e. kappa0 = sqrt(3/(m+3)), gamma_out = 3/(4m(m+3)), sl = sqrt((m+3)/2), c = 1/(m+3)."""

    def __init__(self, m, v=Fr(2, 3), mu=Fr(1), sig2=Fr(0)):
        D = 1 + m * sig2 / 2
        A = 1 / (2 * v) + m * mu * mu / (4 * D)
        Bc = mu / (2 * D)
        g0 = Bc * Bc / (4 * A) + (Fr(0) if MUT == "tilt-gamma" else sig2 / (8 * D))
        self.m = m
        self.gamma_out = Fr(1, 4 * m) - g0
        c = Bc / (2 * A)
        if MUT == "trace-slope" and sig2 == 0:
            c = Fr(1, m + 2)
        self.sl = Iv.frac(2 * A).sqrt()
        self.beta = self.sl * Iv.frac(c)
        self.pref = 1 / Iv.frac(2 * v * A * D).sqrt() / Z_MEHTA[m]
        self.M = model(m, None if self.gamma_out == F_pieces(m)[1] else self.gamma_out)
        self.cache = {}

    def lam(self, b):
        if isinstance(b, float):
            return self.M.integral() if b > 0 else Iv(ZERO)
        key = (b.lo, b.hi)
        if key not in self.cache:
            self.cache[key] = self.M.integral(self.sl * b, self.beta)
        return self.cache[key]

    def I(self, B):
        lo, hi = (x if isinstance(x, (float, Iv)) else Iv.frac(x) for x in B)
        return self.pref * (self.lam(hi) - self.lam(lo))


def gl_nodes(n):
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < 1e-15:
                break
        xs.append(x)
        ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws


def M4_float(b, nodes, R=14.0):
    """E[det(bI - Q)^2 1{Q < bI}] for the 4 x 4 GOE, by Gauss-Legendre over the ordered eigenvalues lambda_4 = b - a,
    lambda_3 = lambda_4 - p1, lambda_2 = lambda_3 - p2, lambda_1 = lambda_2 - p3 with Mehta's density 24/Z_4 |Delta| e^{-|lambda|^2/4}
    (float control, independent of the total-trace coordinates and of every exact layer)."""
    xs, ws = nodes
    pts = [(R * (x + 1) / 2, w * R / 2) for x, w in zip(xs, ws)]
    tot = 0.0
    for a, wa in pts:
        l4 = b - a
        for p1, w1 in pts:
            l3 = l4 - p1
            for p2, w2 in pts:
                l2 = l3 - p2
                e3 = l4 * l4 + l3 * l3 + l2 * l2
                f3 = wa * w1 * w2 * (a * (a + p1) * (a + p1 + p2)) ** 2 * p1 * p2 * (p1 + p2)
                for p3, w3 in pts:
                    l1 = l2 - p3
                    tot += f3 * w3 * (a + p1 + p2 + p3) ** 2 * p3 * (p2 + p3) * (p1 + p2 + p3) * math.exp(-(e3 + l1 * l1) / 4)
    return 24 / (1536 * math.pi) * tot


def I4_float(B, nb=12, n=28):
    xs, ws = gl_nodes(nb)
    nodes = gl_nodes(n)
    lo, hi = float(B[0]), float(B[1])
    tot = 0.0
    for x, w in zip(xs, ws):
        b = lo + (hi - lo) * (x + 1) / 2
        tot += w * math.sqrt(3 / (4 * math.pi)) * math.exp(-3 * b * b / 4) * M4_float(b, nodes)
    return tot * (hi - lo) / 2


def F4_direct_float(T, n=24):
    """int_{T > w > s > q > 0} P(T, w, s, q) e^{-w^2/48 - s^2/24 - q^2/8} by nested Gauss-Legendre (float control of the exact layers)."""
    xs, ws = gl_nodes(n)
    tot = 0.0
    for xw, ww in zip(xs, ws):
        w = T * (xw + 1) / 2
        for xs_, ws_ in zip(xs, ws):
            s_ = w * (xs_ + 1) / 2
            for xq, wq in zip(xs, ws):
                q = s_ * (xq + 1) / 2
                a, p1, p2, p3 = (T - w) / 4, (w - s_) / 3, (s_ - q) / 2, q
                m4, m3, m2, m1 = a, a + p1, a + p1 + p2, a + p1 + p2 + p3
                vand = p1 * (p1 + p2) * (p1 + p2 + p3) * p2 * (p2 + p3) * p3
                tot += ww * ws_ * wq * (m4 * m3 * m2 * m1) ** 2 * vand * math.exp(-w * w / 48 - s_ * s_ / 24 - q * q / 8) * (T / 2) * (w / 2) * (s_ / 2)
    return tot


# ---------------- the d = 5 jet of the reference kernel exp(-|z|^2/2), frame (u, w1, w2, w3, w4)
def _e(*idx):
    v = [0] * 5
    for i in idx:
        v[i] += 1
    return tuple(v)


A_PAIRS = [(i, j) for i in range(1, 5) for j in range(i, 5)]                          # A11 A12 A13 A14 A22 A23 A24 A33 A34 A44
EVEN = [_e(), _e(0, 0)] + [_e(0, i) for i in range(1, 5)] + [_e(i, j) for i, j in A_PAIRS]   # f, f_uu, f_uw1..4, A (16 entries)
ODD = [_e(0)] + [_e(i) for i in range(1, 5)] + [_e(0, 0, 0)]                                  # f_u, f_w1..4, t = f_uuu
IDX_X, IDX_V = [0] + list(range(6, 16)), [1, 2, 3, 4, 5]
DIAG_A = [1 + k for k, (i, j) in enumerate(A_PAIRS) if i == j]                         # positions of A11, A22, A33, A44 in IDX_X
OFF_A = [1 + k for k, (i, j) in enumerate(A_PAIRS) if i != j]
N_VEC, DEGREE = 11, 8                                                                  # Lemma S: dimension of (f, A), degree of det(A)^2


def dfact2(n):
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def ref_cov(a, b):
    """Cov(d^a f, d^b f) = (-1)^|a| d^(a+b) exp(-|z|^2/2) at 0 = (-1)^|a| prod_i (-1)^(al_i/2) (al_i - 1)!! (all al_i even)."""
    al = [x + y for x, y in zip(a, b)]
    if any(x % 2 for x in al):
        return Fr(0)
    v = Fr((-1) ** sum(a))
    for x in al:
        v *= (-1) ** (x // 2) * dfact2(x - 1)
    if MUT == "tau-cross" and sorted([a, b]) == sorted([_e(0, 0, 0), _e(0)]):
        return Fr(0)
    return v


def ref_matrix(idx):
    return [[ref_cov(a, b) for b in idx] for a in idx]


def det_n(M):
    """Determinant by Laplace expansion along the first row (n <= 5); Fractions or intervals."""
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    tot = None
    for j in range(n):
        minor = [[M[i][k] for k in range(n) if k != j] for i in range(1, n)]
        t = M[0][j] * det_n(minor)
        tot = t if tot is None else (tot + t if j % 2 == 0 else tot - t)
    return tot


def inv_n(M):
    """Inverse by the adjugate (n <= 5); works for Fractions and for intervals."""
    n = len(M)
    d = det_n(M)
    cof = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            minor = [[M[r][c] for c in range(n) if c != j] for r in range(n) if r != i]
            v = det_n(minor)
            cof[i][j] = v if (i + j) % 2 == 0 else -v
    return [[cof[j][i] / d for j in range(n)] for i in range(n)], d


def schur(C, X, V):
    """Cov(X | V) = C_XX - C_XV C_VV^-1 C_VX for index lists into C (Fractions or intervals)."""
    Vinv, detV = inv_n([[C[i][j] for j in V] for i in V])
    out = []
    for i in X:
        row = []
        for j in X:
            s = None
            for p in range(len(V)):
                for q in range(len(V)):
                    t = C[i][V[p]] * Vinv[p][q] * C[V[q]][j]
                    s = t if s is None else s + t
            row.append(C[i][j] + s if MUT == "schur-sign" else C[i][j] - s)
        out.append(row)
    return out, detV


def sylvester_pd(M):
    """Exact positive definiteness of a symmetric rational matrix (Gaussian elimination; all pivots positive)."""
    n = len(M)
    A = [list(map(Fr, r)) for r in M]
    for k in range(n):
        if A[k][k] <= 0:
            return False
        for i in range(k + 1, n):
            f = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= f * A[k][j]
    return True


C_EVEN_REF = ref_matrix(EVEN)
C_ODD_REF = ref_matrix(ODD)
CPRIME_REF, DETV_REF = schur(C_EVEN_REF, IDX_X, IDX_V)
LAMBDA0 = Fr(1, 4)


def shifted(M, lam):
    return [[M[i][j] - (lam if i == j else 0) for j in range(len(M))] for i in range(len(M))]


# ---------------- image bound for the torus kernel in d = 5
def image_bound5(L):
    """E^(5)_L = 60500 (76 L^6 + 15) exp(-L^2/2): the shell |n|_inf = j carries (2j+1)^5 - (2j-1)^5 = 160 j^4 + 80 j^2 + 2 <= 242 j^4
    points with |n|^2 <= 5 j^2, so sum_{n != 0} |n|^6 exp(-L^2 |n|^2/2) <= 30250 sum_j j^10 exp(-L^2 j^2/2) <= 60500 exp(-L^2/2)
    (successive terms ratio at most 2^10 exp(-3L^2/2) < 1/2 for L >= 10); the contraction bound 76|x|^6 phi(x) (q <= 6, |x| >= 1),
    |D^q phi(0)| <= 15 and the normalization S >= 1 are those of Math-#197 section 3 (dimension-free); the constant is Math-#200's E_5."""
    shells = 160 if MUT == "image-shells" else 242
    L = Fr(L)
    return Iv.frac(2 * shells * 125 * (76 * L ** 6 + 15)) * (-Iv.frac(L * L / 2)).exp()


def half_power(x, twice):
    r = ipow(x, twice // 2)
    return r * x.sqrt() if twice % 2 else r


SANDWICH_VIOLATIONS = []


def scaled(b, s):
    if isinstance(b, float):
        return b
    b = b if isinstance(b, Iv) else Iv.frac(b)
    return b * s if MUT == "window-scale" else b / s


def sandwich(birth, B, eps):
    """Lemma S with n = 11, degree 8: (1-eps)^(19/2)/(1+eps)^(11/2) I^ref_{B/sqrt(1-eps)} <= I'_B <= (1+eps)^(19/2)/(1-eps)^(11/2)
    I^ref_{B/sqrt(1+eps)}."""
    e = Iv(eps) if isinstance(eps, Decimal) else Iv.of(eps)
    if e.hi >= 1:
        raise ValueError("eps must be below 1")
    up, dn = 1 + e, 1 - e
    s_up, s_dn = up.sqrt(), dn.sqrt()
    su, sd = (s_dn, s_up) if MUT == "sandwich-swap" else (s_up, s_dn)
    lo, hi = B
    upper = half_power(up, N_VEC + DEGREE) / half_power(dn, N_VEC) * birth.I((scaled(lo, su), scaled(hi, su)))
    lower = half_power(dn, N_VEC + DEGREE) / half_power(up, N_VEC) * birth.I((scaled(lo, sd), scaled(hi, sd)))
    if lower.lo > upper.hi:                       # impossible for a correct bracket; recorded and reported by rule SANDWICH_ORDERED
        SANDWICH_VIOLATIONS.append(str(B))
        return Iv(upper.lo, lower.hi)
    return Iv(lower.lo, upper.hi)


def scaled_exact(birth, B, eta):
    """Exact I'_B for C' = (1 + eta) C'_ref: (1 + eta)^4 I^ref_{B/sqrt(1+eta)} (degree-8 homogeneity of det(A)^2)."""
    s = (1 + Iv.frac(eta)).sqrt()
    lo, hi = B
    return ipow(1 + Iv.frac(eta), DEGREE // 2) * birth.I((scaled(lo, s), scaled(hi, s)))


def frob_eps(Cp, Cref):
    """eps = ||C' - C'_ref||_F / lambda0, entrywise supremum of the deviation over each interval entry."""
    dev2 = Iv(ZERO)
    n = len(Cref)
    for i in range(n):
        for j in range(n):
            r = Iv.frac(Cref[i][j])
            c = Cp[i][j] if isinstance(Cp[i][j], Iv) else Iv.frac(Cp[i][j])
            d = max(abs(CC.subtract(c.hi, r.lo)), abs(CF.subtract(c.lo, r.hi)))
            dev2 = dev2 + Iv(d).sq()
    return (dev2.sqrt() / Iv.frac(LAMBDA0)).hi


class Space5:
    """Every covariance entry of the 5-jet (other than Var f = 1, exact by the normalization) in [ref - E, ref + E]; the odd block
    (G, t) is independent of the even block (f, V, A) for any even kernel ([LP] section 15)."""

    def __init__(self, E, birth):
        E = Iv.of(E)
        self.birth = birth

        def box(M):
            n = len(M)
            Bx = [[None] * n for _ in range(n)]
            for i in range(n):
                for j in range(i, n):
                    Bx[i][j] = Bx[j][i] = Iv.frac(M[i][j]) + Iv(E.hi.copy_negate(), E.hi)
            return Bx
        CE_ = box(C_EVEN_REF)
        CE_[0][0] = Iv(ONE)
        CO = box(C_ODD_REF)
        G = [[CO[i][j] for j in range(5)] for i in range(5)]
        Ginv, detG = inv_n(G)
        c = [CO[5][j] for j in range(5)]
        self.tau2 = CO[5][5] - sum((c[i] * Ginv[i][j] * c[j] for i in range(5) for j in range(5)), Iv(ZERO))
        self.pG = 1 / (4 * PI.sq() * SQRT2PI * detG.sqrt())             # (2 pi)^(-5/2) det(Cov G)^(-1/2)
        self.Cp, detV = schur(CE_, IDX_X, IDX_V)
        self.pV = 1 / (4 * PI.sq() * SQRT2PI * detV.sqrt())
        self.tau43 = root_iv(self.tau2.sq(), 3)
        self.eps = frob_eps(self.Cp, CPRIME_REF)

    def k_integral(self, K):
        lo, hi = K

        def s(k):
            return k if isinstance(k, float) else 72 * Iv.frac(k * k) / self.tau2
        a = SHAPE()
        return self.tau43 * C72 * (gamma_lower(s(hi), a) - gamma_lower(s(lo), a)) / (2 * SQRT2PI)

    def coefficient(self, B, K):
        sphere = 2 * PI.sq() if MUT == "sphere" else 8 * PI.sq() / 3     # |S^4| = 8 pi^2/3
        return 144 * sphere * self.pG * self.pV * sandwich(self.birth, B, self.eps) * self.k_integral(K)


# ---------------- a non-scalar exact test of Lemma S: the tilted pair (f, g)
TILT = (Fr(17, 25), Fr(33, 34), Fr(2261, 86700))                      # v = Var f, mu, sigma^2 with mu^2 v + sigma^2 = 2/3


def tilted_cprime(v, mu, sig2):
    """Covariance of (f, A11, A12, ..., A44) for A = Q - (mu f + sigma Z) I."""
    s = mu * mu * v + sig2
    C = [[Fr(0)] * 11 for _ in range(11)]
    C[0][0] = v
    for i in DIAG_A:
        C[0][i] = C[i][0] = -mu * v
        for j in DIAG_A:
            C[i][j] = s + (2 if i == j else 0)
    for i in OFF_A:
        C[i][i] = Fr(1)
    return C
