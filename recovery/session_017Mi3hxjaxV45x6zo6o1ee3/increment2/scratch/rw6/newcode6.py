

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
    """The inner integrals in closed form, for every m = 2, ..., 5 by one generic layer recursion (NOTE section 2).  On the ordered
    eigenvalue sector mu_m = -a, mu_{m-1} = -a - p_1, ..., mu_1 = -a - p_1 - ... - p_{m-1} put x_0 = T = m a + (m-1) p_1 + ... + p_{m-1}
    (minus the trace) and x_k = (m-k) p_k + ... + p_{m-1}, k = 1, ..., m-1 (Jacobian 1/m!, region T > x_1 > ... > x_{m-1} > 0); the
    integrand is P(x) e^{-sum_k x_k^2 c_k} with c_{m-1} = 1/8 and c_{m-2}, c_{m-3}, c_{m-4} = 1/24, 1/48, 1/80 (eight times the
    triangular numbers), P = prod mu_i^2 times the Vandermonde.  The q = x_{m-1} layer is exact
    (Math-#201 L1); every further layer integrates y^j e^{-c y^2} times a kind ('1', ('e', g), ('E', g), ('eE', al, g)) from 0 to
    the next variable by moment_dict and L_dict.  An Owen-type kind ('K', al, g) entering a further layer is refused (ValueError).
    Returns ({kind: polynomial in T, list by power}, outer gamma 3/(4 m (m+3)), [(c, generated kinds that came out zero) per layer])."""
    nv = m
    xs = [pvar(i, nv) for i in range(nv)]

    def lin(i, j, den):
        co = [Fr(0)] * nv
        co[i] += Fr(1, den)
        if j is not None:
            co[j] -= Fr(1, den)
        return plin(co)
    a = lin(0, 1, m)
    ps = [lin(k, k + 1 if k + 1 < m else None, m - k) for k in range(1, m)]
    mus, run = [pscale(a, Fr(-1))], a
    for p in ps:
        run = padd(run, p)
        mus.append(pscale(run, Fr(-1)))
    P = {(0,) * nv: Fr(1)}
    for mu in mus:
        P = pmul(P, pmul(mu, mu))
    for i in range(m):
        for j in range(i + 1, m):
            P = pmul(P, padd(mus[i], pscale(mus[j], Fr(-1))))                 # mus[i] > mus[j]
    U, V = {}, {}
    for mono, c in P.items():
        Bk, Ck = Jk_closed(mono[-1])                                           # raises on an even power of q
        base = mono[:-1]
        U[base] = U.get(base, Fr(0)) + c * Bk
        for n, x in enumerate(Ck):
            if x:
                mm = list(base)
                mm[-1] += n
                mm = tuple(mm)
                V[mm] = V.get(mm, Fr(0)) + c * x
    pieces = {("1",): {k: v for k, v in U.items() if v}, ("e", Fr(1, 8)): {k: v for k, v in V.items() if v}}
    stages, nvc = [], nv - 1
    for c in (Fr(1, 24), Fr(1, 48), Fr(1, 80))[:m - 2]:
        pieces = layer(pieces, nvc, c)
        nvc -= 1
        if MUT == "layer-dropped" and c == Fr(1, 48):
            pieces.pop(("E", Fr(3, 16)), None)                                # a cancelled layer silently not derived
        stages.append((c, sorted(("%s" % (k,)) for k, p in pieces.items() if not p)))
    out = {}
    for kind, poly in pieces.items():
        lst = []
        for mono, v in poly.items():
            n = mono[0]
            lst += [Fr(0)] * (n + 1 - len(lst))
            lst[n] += v
        out[kind] = lst
    return out, Fr(3, 4 * m * (m + 3)), stages


def layer(pieces, nv, c):
    """Integrate the last variable y of nv variables (T, x_1, ..., y) over (0, x_{nv-2}) against e^{-c y^2}, kind by kind."""
    out = {}
    for kind, poly in pieces.items():
        for mono, coef in poly.items():
            j = mono[-1]
            if kind[0] == "1":
                lay = moment_dict(c, j)
            elif kind[0] == "e":
                lay = moment_dict(c + kind[1], j)
            elif kind[0] == "E":
                lay = L_dict(c, kind[1], j)
            elif kind[0] == "eE":
                lay = L_dict(c + kind[1], kind[2], j)
            else:
                raise ValueError("an Owen-type kind %r survived an inner layer" % (kind,))
            base = mono[:-1]
            for k2, pl in lay.items():
                acc = out.setdefault(k2, {})
                for n, x in enumerate(pl):
                    if x:
                        mm = list(base)
                        mm[-1] += n
                        mm = tuple(mm)
                        acc[mm] = acc.get(mm, Fr(0)) + coef * x
    return {k: {mm: v for mm, v in p.items() if v} for k, p in out.items()}


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
    """g_m(T) = e^{-gamma_out T^2} F_m(T) as terms (const, polynomial, gamma, kappa, K): const * poly(T) e^{-gamma T^2} [erf(kappa T)]
    [K_{al,g}(T)], with K_{al,g}(T) = int_0^T e^{-al w^2} E_g(w) dw the Owen-type kinds (certified cumulative layer, NOTE section 3)."""
    pieces, g_ref, _ = F_pieces(m)
    gout = g_ref if gamma_out is None else gamma_out
    terms = []
    for key, poly in sorted(pieces.items(), key=lambda kv: str(kv[0])):
        poly = trim(poly)
        if not poly:
            continue
        if key[0] == "1":
            terms.append((Iv(ONE), poly, Iv.frac(gout), None, None))
        elif key[0] == "e":
            terms.append((Iv(ONE), poly, Iv.frac(gout + key[1]), None, None))
        elif key[0] == "E":
            terms.append(((PI / Iv.frac(key[1])).sqrt() / 2, poly, Iv.frac(gout), Iv.frac(key[1]).sqrt(), None))
        elif key[0] == "eE":
            terms.append(((PI / Iv.frac(key[2])).sqrt() / 2, poly, Iv.frac(gout + key[1]), Iv.frac(key[2]).sqrt(), None))
        else:
            terms.append((Iv(ONE), poly, Iv.frac(gout), None, (key[1], key[2])))
    return terms


def K_float(al, g, T, n=4000):
    """Float control of K_{al,g}(T) by Simpson's rule."""
    if T == 0:
        return 0.0
    h = T / n
    f = [math.exp(-al * (i * h) ** 2) * 0.5 * math.sqrt(math.pi / g) * math.erf(math.sqrt(g) * i * h) for i in range(n + 1)]
    return h / 3 * (f[0] + f[-1] + 4 * sum(f[1:-1:2]) + 2 * sum(f[2:-1:2]))


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
            tot += pv * K_float(float(key[1]), float(key[2]), T)
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
    """int_0^inf g(T) Phi(alpha - beta T) dT for g = sum const * poly(T) e^{-gamma T^2} [erf(kappa T)] [K_{al,g}(T)] (entire).  On each
    cell of width CELL the point Taylor coefficients of g at the centre are computed once and contracted with the cell moments, so
    that an evaluation for given (alpha, beta) costs one product of coefficient vectors per cell.  The remainder of the order-K
    Taylor polynomial of the product is bounded by Cauchy's estimate at the centre, |c_n| <= M_g M_Phi / rho^n, summed geometrically;
    beyond T_CUT the integrand is bounded by |g| (0 <= Phi <= 1, |erf| <= 1, |K_{al,g}| <= pi/(4 sqrt(al g))) and incomplete gamma
    bounds.  K_{al,g} is a certified cumulative layer (Math-#204 section 3): its value at a centre is the certified integral of
    k = e^{-al w^2} E_g(w) over the full cells before plus the left half of the current cell, its higher Taylor coefficients are those
    of k divided by n, and its disc bound is |K(s0)| + rho M_k."""

    def __init__(self, terms, K=K_ORDER, cell=CELL, T=T_CUT, rho=RHO):
        self.K, self.rho = K, rho
        half = cell / 2
        ncell = int(T / cell)
        if ncell * cell != T or Fr(half) >= Fr(rho):
            raise ValueError("bad quadrature parameters")
        mom = [Iv.frac(2 * half ** (n + 1) / (n + 1)) if n % 2 == 0 else Iv(ZERO) for n in range(K + 1)]
        left = [Iv.frac((-1) ** n * half ** (n + 1) / (n + 1)) for n in range(K + 1)]   # int_{-h}^0 x^n dx
        rem_moment = Iv.frac(2 * half ** (K + 2) / (K + 2) / (1 - Fr(half) / Fr(rho)))
        rho_pow = Iv.frac(Fr(rho) ** (K + 1))
        kinds = sorted({kk for *_, kk in terms if kk is not None})
        Kacc = {kk: Iv(ZERO) for kk in kinds}
        self.Kvals = {kk: [] for kk in kinds}
        self.cells, Rsum = [], Iv(ZERO)
        for c in range(ncell):
            s0 = (2 * c + 1) * half
            Kco, Kbd = {}, {}
            for al, g in kinds:
                Ec = (PI / Iv.frac(g)).sqrt() / 2
                kc = [Ec * x for x in t_mul(gauss_coeffs(Iv.frac(al), s0, K), erf_coeffs(Iv.frac(g).sqrt(), s0, K))]
                Mk = Ec * gauss_bound(Iv.frac(al), s0, rho) * erf_bound(Iv.frac(g).sqrt(), s0, rho)
                Rk = (Mk / rho_pow * rem_moment / 2).hi                          # one-sided Cauchy remainder
                if MUT == "remainder-dropped":
                    Rk = ZERO
                lefti = sum((kc[n] * left[n] for n in range(K + 1)), Iv(ZERO))
                full = sum((kc[n] * mom[n] for n in range(0, K + 1, 2)), Iv(ZERO))
                Ks0 = Kacc[(al, g)] + (Iv(ZERO) if MUT == "cumulative-shift" else lefti) + Iv(Rk.copy_negate(), Rk)
                Kacc[(al, g)] = Kacc[(al, g)] + full + Iv(CC.multiply(Decimal(-2), Rk), CC.multiply(Decimal(2), Rk))
                Kco[(al, g)] = [Ks0] + [kc[n] / (n + 1) for n in range(K)]
                Kbd[(al, g)] = Iv(Ks0.mag()) + Iv.frac(rho) * Mk
                self.Kvals[(al, g)].append((s0, Ks0))
            G, M = [Iv(ZERO)] * (K + 1), Iv(ZERO)
            for const, poly, gam, kap, kk in terms:
                pc = t_mul(poly_coeffs(poly, s0, K), gauss_coeffs(gam, s0, K))
                Mb = poly_bound(poly, s0, rho) * gauss_bound(gam, s0, rho)
                if kap is not None:
                    pc = t_mul(pc, erf_coeffs(kap, s0, K))
                    Mb = Mb * erf_bound(kap, s0, rho)
                if kk is not None:
                    pc = t_mul(pc, Kco[kk])
                    Mb = Mb * Kbd[kk]
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
        self.K_at_cut = Kacc
        tail = Iv(ZERO)
        for const, poly, gam, kap, kk in terms:              # int_T^inf t^j e^{-g t^2} dt = (1/2) g^{-(j+1)/2} Gamma((j+1)/2, g T^2)
            kfac = Iv(ONE) if kk is None else PI / (4 * (Iv.frac(kk[0]) * Iv.frac(kk[1])).sqrt())
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
                tail = tail + Iv(const.mag()) * kfac * Iv.frac(abs(cj)) * up / gpow / 2
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


Z_MEHTA = {2: 8 * SQRT2PI, 3: 48 * SQRT2 * PI, 4: 1536 * PI, 5: 46080 * PI * SQRTPI}   # Mehta's integral (Math-#199 section 1; Z_4, Z_5: Math-#202, #204)


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


def F5_direct_float(T, n=16):
    """int_{T > w' > w > s > q > 0} P e^{-w'^2/80 - w^2/48 - s^2/24 - q^2/8} by nested Gauss-Legendre (float control of the exact
    layers and of the cumulative K layer; P = prod mu_i^2 times the Vandermonde in the gaps)."""
    xs, ws = gl_nodes(n)
    tot = 0.0
    for x1, w1 in zip(xs, ws):
        u = T * (x1 + 1) / 2
        a = (T - u) / 5
        for x2, w2 in zip(xs, ws):
            w = u * (x2 + 1) / 2
            p1 = (u - w) / 4
            for x3, w3 in zip(xs, ws):
                s_ = w * (x3 + 1) / 2
                p2 = (w - s_) / 3
                pre = w1 * w2 * w3 * (T / 2) * (u / 2) * (w / 2) * (s_ / 2) * math.exp(-u * u / 80 - w * w / 48 - s_ * s_ / 24)
                for x4, w4 in zip(xs, ws):
                    q = s_ * (x4 + 1) / 2
                    p3, p4 = (s_ - q) / 2, q
                    m1, m2, m3, m4, m5 = a, a + p1, a + p1 + p2, a + p1 + p2 + p3, a + p1 + p2 + p3 + p4
                    gaps = (p1, p2, p3, p4)
                    vand = 1.0
                    for i in range(4):
                        acc = 0.0
                        for j in range(i, 4):
                            acc += gaps[j]
                            vand *= acc
                    tot += pre * w4 * (m1 * m2 * m3 * m4 * m5) ** 2 * vand * math.exp(-q * q / 8)
    return tot


def I5_float(B, n=48, R=80.0):
    """Float control of the outer T layer: Gauss-Legendre in T of e^{-3T^2/160} F_5(T) W_B(T) with F_5 from the float evaluation of
    the exact pieces (math.erf and Simpson's K), independent of the certified Taylor/Cauchy quadrature and of the cumulative layer."""
    xs, ws = gl_nodes(n)
    pieces = F_pieces(5)[0]
    lo, hi = float(B[0]), float(B[1])

    def Phi_f(x):
        return 0.5 * math.erfc(-x / math.sqrt(2))
    tot = 0.0
    for k in range(4):                                                       # four panels of R/4
        a0 = k * R / 4
        for x, w in zip(xs, ws):
            T = a0 + R / 8 * (x + 1)
            W = Phi_f(2 * (hi - T / 8)) - Phi_f(2 * (lo - T / 8))
            tot += w * R / 8 * math.exp(-3 * T * T / 160) * F_float(pieces, T) * W
    return math.sqrt(3 / 8) / (46080 * math.pi ** 1.5) * tot


# ---------------- the d = 6 jet of the reference kernel exp(-|z|^2/2), frame (u, w1, ..., w5)
def _e(*idx):
    v = [0] * 6
    for i in idx:
        v[i] += 1
    return tuple(v)


A_PAIRS = [(i, j) for i in range(1, 6) for j in range(i, 6)]                          # A11 A12 ... A55 (15 entries)
EVEN = [_e(), _e(0, 0)] + [_e(0, i) for i in range(1, 6)] + [_e(i, j) for i, j in A_PAIRS]   # f, f_uu, f_uw1..5, A (22 entries)
ODD = [_e(0)] + [_e(i) for i in range(1, 6)] + [_e(0, 0, 0)]                                  # f_u, f_w1..5, t = f_uuu
IDX_X, IDX_V = [0] + list(range(7, 22)), [1, 2, 3, 4, 5, 6]
DIAG_A = [1 + k for k, (i, j) in enumerate(A_PAIRS) if i == j]                         # positions of A11, ..., A55 in IDX_X
OFF_A = [1 + k for k, (i, j) in enumerate(A_PAIRS) if i != j]
N_VEC, DEGREE = 16, 10                                                                 # Lemma S: dimension of (f, A), degree of det(A)^2


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
    """Determinant by Laplace expansion along the first row (n <= 6); Fractions or intervals."""
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
    """Inverse by the adjugate (n <= 6); works for Fractions and for intervals."""
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
LAMBDA0 = Fr(23, 100)


def shifted(M, lam):
    return [[M[i][j] - (lam if i == j else 0) for j in range(len(M))] for i in range(len(M))]


# ---------------- image bound for the torus kernel in d = 6
def image_bound6(L):
    """E^(6)_L = 314496 (76 L^6 + 15) exp(-L^2/2): the shell |n|_inf = j carries (2j+1)^6 - (2j-1)^6 = 384 j^5 + 320 j^3 + 24 j <= 728 j^5
    points with |n|^2 <= 6 j^2, so sum_{n != 0} |n|^6 exp(-L^2 |n|^2/2) <= 157248 sum_j j^11 exp(-L^2 j^2/2) <= 314496 exp(-L^2/2)
    (successive terms ratio at most 2^11 exp(-3L^2/2) < 1/2 for L >= 10); the contraction bound 76|x|^6 phi(x) (q <= 6, |x| >= 1),
    |D^q phi(0)| <= 15 and the normalization S >= 1 are those of Math-#197 section 3 (dimension-free); the constant is Math-#204's E_6."""
    shells = 384 if MUT == "image-shells" else 728
    L = Fr(L)
    return Iv.frac(2 * shells * 216 * (76 * L ** 6 + 15)) * (-Iv.frac(L * L / 2)).exp()


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
    """Lemma S with n = 16, degree 10: (1-eps)^13/(1+eps)^8 I^ref_{B/sqrt(1-eps)} <= I'_B <= (1+eps)^13/(1-eps)^8
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
    """Exact I'_B for C' = (1 + eta) C'_ref: (1 + eta)^5 I^ref_{B/sqrt(1+eta)} (degree-10 homogeneity of det(A)^2)."""
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


class Space6:
    """Every covariance entry of the 6-jet (other than Var f = 1, exact by the normalization) in [ref - E, ref + E]; the odd block
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
        G = [[CO[i][j] for j in range(6)] for i in range(6)]
        Ginv, detG = inv_n(G)
        c = [CO[6][j] for j in range(6)]
        self.tau2 = CO[6][6] - sum((c[i] * Ginv[i][j] * c[j] for i in range(6) for j in range(6)), Iv(ZERO))
        self.pG = 1 / (8 * PI.sq() * PI * detG.sqrt())                   # (2 pi)^(-3) det(Cov G)^(-1/2)
        self.Cp, detV = schur(CE_, IDX_X, IDX_V)
        self.pV = 1 / (8 * PI.sq() * PI * detV.sqrt())
        self.tau43 = root_iv(self.tau2.sq(), 3)
        self.eps = frob_eps(self.Cp, CPRIME_REF)

    def k_integral(self, K):
        lo, hi = K

        def s(k):
            return k if isinstance(k, float) else 72 * Iv.frac(k * k) / self.tau2
        a = SHAPE()
        return self.tau43 * C72 * (gamma_lower(s(hi), a) - gamma_lower(s(lo), a)) / (2 * SQRT2PI)

    def coefficient(self, B, K):
        sphere = 8 * PI.sq() / 3 if MUT == "sphere" else PI.sq() * PI     # |S^5| = pi^3
        return 144 * sphere * self.pG * self.pV * sandwich(self.birth, B, self.eps) * self.k_integral(K)


# ---------------- a non-scalar exact test of Lemma S: the tilted pair (f, g)
TILT = (Fr(17, 25), Fr(33, 34), Fr(2261, 86700))                      # v = Var f, mu, sigma^2 with mu^2 v + sigma^2 = 2/3


def tilted_cprime(v, mu, sig2):
    """Covariance of (f, A11, A12, ..., A55) for A = Q - (mu f + sigma Z) I."""
    s = mu * mu * v + sig2
    C = [[Fr(0)] * 16 for _ in range(16)]
    C[0][0] = v
    for i in DIAG_A:
        C[0][i] = C[i][0] = -mu * v
        for j in DIAG_A:
            C[i][j] = s + (2 if i == j else 0)
    for i in OFF_A:
        C[i][i] = Fr(1)
    return C
