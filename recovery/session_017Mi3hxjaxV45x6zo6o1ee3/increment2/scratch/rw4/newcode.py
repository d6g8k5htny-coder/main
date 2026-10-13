

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


def F_pieces(m):
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
    raise ValueError("m must be 2 or 3")


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
        else:
            terms.append(((PI / Iv.frac(key[1])).sqrt() / 2, poly, Iv.frac(gout), Iv.frac(key[1]).sqrt()))
    return terms


def F_float(pieces, T):
    tot = 0.0
    for key, poly in pieces.items():
        pv = sum(float(c) * T ** i for i, c in enumerate(poly))
        if key[0] == "1":
            tot += pv
        elif key[0] == "e":
            tot += pv * math.exp(-float(key[1]) * T * T)
        else:
            g = float(key[1])
            tot += pv * 0.5 * math.sqrt(math.pi / g) * math.erf(math.sqrt(g) * T)
    return tot


def F3_direct_float(T, n=200):
    """int_0^T int_0^s P(T, s, q) e^{-s^2/24 - q^2/8} dq ds by composite Simpson in (s, q): an independent float control of the
    exact layers."""
    hs = T / n
    tot = 0.0
    for i in range(n + 1):
        sv = i * hs
        ws = 1 if i in (0, n) else (4 if i % 2 else 2)
        hq = sv / n
        inner = 0.0
        for j in range(n + 1):
            qv = j * hq
            wq = 1 if j in (0, n) else (4 if j % 2 else 2)
            a, p = (T - sv) / 3, (sv - qv) / 2
            inner += wq * a * a * (a + p) ** 2 * (a + p + qv) ** 2 * p * qv * (p + qv) * math.exp(-sv * sv / 24 - qv * qv / 8)
        tot += ws * inner * hq / 3
    return tot * hs / 3


# ---------------- Taylor quadrature with Cauchy remainders (Math-#199 section 3; factors as functions, plus Phi(alpha - beta T))
def ipow(x, n):
    r = Iv(ONE)
    for _ in range(n):
        r = r * x
    return r


def gauss_coeffs(g, s0, K):
    """Taylor coefficients of exp(-g s^2) at s0: (n+1) y_{n+1} = -2 g (s0 y_n + y_{n-1})."""
    S0 = Iv.frac(s0)
    y = [(-(g * S0.sq())).exp()]
    prev = Iv(ZERO)
    for n in range(K):
        nxt = -(2 * g) * (S0 * y[n] + prev) / (n + 1)
        prev = y[n]
        y.append(nxt)
    return y


def gauss_bound(g, s0, rho):
    inner = max(Fr(0), abs(Fr(s0)) - Fr(rho))
    return (g * Iv.frac(rho * rho) - g * Iv.frac(inner * inner)).exp()   # |e^{-g z^2}| = e^{-g(x^2 - y^2)}, |y| <= rho


def erf_coeffs(kappa, s0, K):
    """erf(kappa s): value at s0, then (2 kappa/sqrt pi) times the Gaussian's coefficients divided by n + 1."""
    y = gauss_coeffs(kappa.sq(), s0, K)
    pref = 2 * kappa / SQRTPI
    return [erf_iv(kappa * Iv.frac(s0))] + [pref * y[n] / (n + 1) for n in range(K)]


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
    return [Phi(w0)] + [-(pref * y[n] / (n + 1)) for n in range(K)]


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
    beyond T_CUT the integrand is bounded by |g| (0 <= Phi <= 1) and incomplete gamma bounds."""

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
        for const, poly, gam, kap in terms:                   # int_T^inf t^j e^{-g t^2} dt = (1/2) g^{-(j+1)/2} Gamma((j+1)/2, g T^2)
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


Z_MEHTA = {2: 8 * SQRT2PI, 3: 48 * SQRT2 * PI}                 # Math-#199 section 1 (Mehta's integral; Z_3 also in closed form there)


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


def M3_float(b, nodes, R=16.0):
    """E[det(bI - Q)^2 1{Q < bI}] for the 3 x 3 GOE, by Gauss-Legendre over the ordered eigenvalues lambda_3 = b - a,
    lambda_2 = b - a - p, lambda_1 = b - a - p - q with Mehta's density 6/Z_3 |Delta| e^{-|lambda|^2/4} (float control)."""
    xs, ws = nodes
    pts = [(R * (x + 1) / 2, w * R / 2) for x, w in zip(xs, ws)]
    tot = 0.0
    for a, wa in pts:
        ea = (b - a) ** 2
        for p, wp in pts:
            ep = ea + (b - a - p) ** 2
            for q, wq in pts:
                e = ep + (b - a - p - q) ** 2
                tot += wa * wp * wq * a * a * (a + p) ** 2 * (a + p + q) ** 2 * p * q * (p + q) * math.exp(-e / 4)
    return 6 / (48 * math.sqrt(2) * math.pi) * tot


def I3_float(B, nb=20, n=28):
    xs, ws = gl_nodes(nb)
    nodes = gl_nodes(n)
    lo, hi = float(B[0]), float(B[1])
    tot = 0.0
    for x, w in zip(xs, ws):
        b = lo + (hi - lo) * (x + 1) / 2
        tot += w * math.sqrt(3 / (4 * math.pi)) * math.exp(-3 * b * b / 4) * M3_float(b, nodes)
    return tot * (hi - lo) / 2


# ---------------- the d = 4 jet of the reference kernel exp(-|z|^2/2), frame (u, w1, w2, w3)
EVEN = [(0, 0, 0, 0), (2, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1),
        (0, 2, 0, 0), (0, 1, 1, 0), (0, 1, 0, 1), (0, 0, 2, 0), (0, 0, 1, 1), (0, 0, 0, 2)]  # f, f_uu, f_uw*, A11 A12 A13 A22 A23 A33
ODD = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (3, 0, 0, 0)]               # f_u, f_w1, f_w2, f_w3, t = f_uuu
IDX_X, IDX_V = [0, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4]
DIAG_A = [1, 4, 6]                                                                     # positions of A11, A22, A33 in IDX_X
N_VEC, DEGREE = 7, 6                                                                   # Lemma S: dimension of (f, A), degree of det(A)^2


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
    if MUT == "tau-cross" and sorted([a, b]) == sorted([(3, 0, 0, 0), (1, 0, 0, 0)]):
        return Fr(0)
    return v


def ref_matrix(idx):
    return [[ref_cov(a, b) for b in idx] for a in idx]


def det_n(M):
    """Determinant by Laplace expansion along the first row (n <= 4); Fractions or intervals."""
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
    """Inverse by the adjugate (n <= 4); works for Fractions and for intervals."""
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
LAMBDA0 = Fr(3, 10)


def shifted(M, lam):
    return [[M[i][j] - (lam if i == j else 0) for j in range(len(M))] for i in range(len(M))]


# ---------------- image bound for the torus kernel in d = 4
def image_bound4(L):
    """E^(4)_L = 10240 (76 L^6 + 15) exp(-L^2/2): the shell |n|_inf = j carries (2j+1)^4 - (2j-1)^4 = 64 j^3 + 16 j <= 80 j^3 points
    with |n|^2 <= 4 j^2, so sum_{n != 0} |n|^6 exp(-L^2 |n|^2/2) <= 5120 sum_j j^9 exp(-L^2 j^2/2) <= 10240 exp(-L^2/2) (successive
    terms ratio at most 2^9 exp(-3L^2/2) < 1/2 for L >= 10); the contraction bound 76|x|^6 phi(x) (q <= 6, |x| >= 1), |D^q phi(0)| <= 15
    and the normalization S >= 1 are those of Math-#197 section 3 (dimension-free)."""
    shells = 64 if MUT == "image-shells" else 80
    L = Fr(L)
    return Iv.frac(2 * shells * 64 * (76 * L ** 6 + 15)) * (-Iv.frac(L * L / 2)).exp()


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
    """Lemma S with n = 7, degree 6: (1-eps)^(13/2)/(1+eps)^(7/2) I^ref_{B/sqrt(1-eps)} <= I'_B <= (1+eps)^(13/2)/(1-eps)^(7/2)
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
    """Exact I'_B for C' = (1 + eta) C'_ref: (1 + eta)^3 I^ref_{B/sqrt(1+eta)} (degree-6 homogeneity of det(A)^2)."""
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


class Space4:
    """Every covariance entry of the 4-jet (other than Var f = 1, exact by the normalization) in [ref - E, ref + E]; the odd block
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
        G = [[CO[i][j] for j in range(4)] for i in range(4)]
        Ginv, detG = inv_n(G)
        c = [CO[4][j] for j in range(4)]
        self.tau2 = CO[4][4] - sum((c[i] * Ginv[i][j] * c[j] for i in range(4) for j in range(4)), Iv(ZERO))
        self.pG = 1 / (4 * PI.sq() * detG.sqrt())                       # (2 pi)^(-2) det(Cov G)^(-1/2)
        self.Cp, detV = schur(CE_, IDX_X, IDX_V)
        self.pV = 1 / (4 * PI.sq() * detV.sqrt())
        self.tau43 = root_iv(self.tau2.sq(), 3)
        self.eps = frob_eps(self.Cp, CPRIME_REF)

    def k_integral(self, K):
        lo, hi = K

        def s(k):
            return k if isinstance(k, float) else 72 * Iv.frac(k * k) / self.tau2
        a = SHAPE()
        return self.tau43 * C72 * (gamma_lower(s(hi), a) - gamma_lower(s(lo), a)) / (2 * SQRT2PI)

    def coefficient(self, B, K):
        sphere = 4 * PI if MUT == "sphere" else 2 * PI.sq()              # |S^3| = 2 pi^2
        return 144 * sphere * self.pG * self.pV * sandwich(self.birth, B, self.eps) * self.k_integral(K)


# ---------------- a non-scalar exact test of Lemma S: the tilted pair (f, g)
TILT = (Fr(17, 25), Fr(33, 34), Fr(2261, 86700))                      # v = Var f, mu, sigma^2 with mu^2 v + sigma^2 = 2/3


def tilted_cprime(v, mu, sig2):
    """Covariance of (f, A11, A12, A13, A22, A23, A33) for A = Q - (mu f + sigma Z) I."""
    s = mu * mu * v + sig2
    C = [[Fr(0)] * 7 for _ in range(7)]
    C[0][0] = v
    for i in DIAG_A:
        C[0][i] = C[i][0] = -mu * v
        for j in DIAG_A:
            C[i][j] = s + (2 if i == j else 0)
    for i in (2, 3, 5):
        C[i][i] = Fr(1)
    return C
