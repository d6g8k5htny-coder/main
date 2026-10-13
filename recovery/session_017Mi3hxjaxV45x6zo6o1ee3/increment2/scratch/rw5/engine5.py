# ---------------- Taylor quadrature with Cauchy remainders (Math-#199 section 3; factors as functions, Phi(alpha - beta T) and a
# cumulative Owen-type factor Kfun(T) = int_0^T e^{-al w^2} E_g(w) dw)
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


def kprime_coeffs(ks, s0, K):
    """Taylor coefficients at s0 of Kfun' = e^{-al T^2} E_g(T) = (1/2) sqrt(pi/g) e^{-al T^2} erf(sqrt(g) T)."""
    al, g = ks
    cg = (PI / Iv.frac(g)).sqrt() / 2
    return [cg * x for x in t_mul(gauss_coeffs(Iv.frac(al), s0, K), erf_coeffs(Iv.frac(g).sqrt(), s0, K))]


def kprime_bound(ks, s0, rho):
    al, g = ks
    return (PI / Iv.frac(g)).sqrt() / 2 * gauss_bound(Iv.frac(al), s0, rho) * erf_bound(Iv.frac(g).sqrt(), s0, rho)


def kfun_sweep(ks, K, cell, ncell, rho):
    """Kfun at every cell centre by a certified cumulative sum: Kfun(0) = 0 and, cell by cell, the exact integral of the order-K Taylor
    polynomial of Kfun' at the centre over the left half (to reach the centre) and over the whole cell (to reach the next left
    end), each with its Cauchy remainder (Kfun' is entire)."""
    half = cell / 2
    rem_half = Iv.frac(half ** (K + 2) / (K + 2) / (1 - Fr(half) / Fr(rho)))       # int_{-h/2}^{0} sum_{n>K} |x|^n/rho^n ... (times rho^-(K+1))
    rho_pow = Iv.frac(Fr(rho) ** (K + 1))
    left, centres = Iv(ZERO), []
    for c in range(ncell):
        s0 = (2 * c + 1) * half
        d = kprime_coeffs(ks, s0, K)
        R = (kprime_bound(ks, s0, rho) / rho_pow * rem_half).hi
        to_centre = Iv(ZERO)
        whole = Iv(ZERO)
        for n in range(K + 1):
            to_centre = to_centre + d[n] * Iv.frac(Fr((-1) ** n) * half ** (n + 1) / (n + 1))
            if n % 2 == 0:
                whole = whole + d[n] * Iv.frac(2 * half ** (n + 1) / (n + 1))
        centre = left + to_centre + Iv(R.copy_negate(), R)
        R2 = CC.multiply(Decimal(2), R)
        left = left + whole + Iv(R2.copy_negate(), R2)
        centres.append(centre)
    return centres


def owen_T_any(h, a):
    """Owen's T(h, a), h >= 0, a > 0: the toolkit series for a <= 1; for a > 1 the reflection
    T(h, a) = Phi(h)/2 + Phi(ah)/2 - Phi(h) Phi(ah) - T(ah, 1/a) (Owen 1956)."""
    if a.hi <= 1:
        return owen_T_eval(h, a)
    if a.lo <= 1:
        raise ValueError("Owen slope interval straddles 1")
    ah = a * h
    refl = Iv(ZERO) if MUT == "owen-reflection" else owen_T_eval(ah, 1 / a)
    return Phi(h) / 2 + Phi(ah) / 2 - Phi(h) * Phi(ah) - refl


def kfun_owen(ks, T):
    """Kfun(T) = (1/2) sqrt(pi/g) sqrt(pi/al) [arctan(a)/pi - 2 T(sqrt(2 al) T, a)], a = sqrt(g/al) (closed form through Owen's T)."""
    al, g = Iv.frac(ks[0]), Iv.frac(ks[1])
    a = (g / al).sqrt()
    at = atan_iv(a) if a.hi <= 1 else PI / 2 - atan_iv(1 / a)
    return (PI / g).sqrt() / 2 * (PI / al).sqrt() * (at / PI - 2 * owen_T_any((2 * al).sqrt() * Iv.frac(T), a))


class Model:
    """int_0^inf g(T) Phi(alpha - beta T) dT for g = sum const * poly(T) e^{-gamma T^2} [erf(kappa T) | Kfun(T)] (entire).  On each
    cell of width CELL the point Taylor coefficients of g at the centre are computed once and contracted with the cell moments, so
    that an evaluation for given (alpha, beta) costs one product of coefficient vectors per cell.  Kfun's value at the centre comes
    from the certified cumulative sweep and its higher coefficients from Kfun'.  The remainder of the order-K Taylor polynomial of
    the product is bounded by Cauchy's estimate at the centre, |c_n| <= M_g M_Phi / rho^n, summed geometrically (on the disc,
    |Kfun(z)| <= |Kfun(s0)| + rho sup|Kfun'|); beyond T_CUT the integrand is bounded by |g| (0 <= Phi <= 1, |erf| <= 1,
    |Kfun| <= (1/2) sqrt(pi/g) (1/2) sqrt(pi/al)) and incomplete gamma bounds."""

    def __init__(self, terms, K=K_ORDER, cell=CELL, T=T_CUT, rho=RHO):
        self.K, self.rho = K, rho
        half = cell / 2
        ncell = int(T / cell)
        if ncell * cell != T or Fr(half) >= Fr(rho):
            raise ValueError("bad quadrature parameters")
        mom = [Iv.frac(2 * half ** (n + 1) / (n + 1)) if n % 2 == 0 else Iv(ZERO) for n in range(K + 1)]
        rem_moment = Iv.frac(2 * half ** (K + 2) / (K + 2) / (1 - Fr(half) / Fr(rho)))
        rho_pow = Iv.frac(Fr(rho) ** (K + 1))
        sweeps = {ks: kfun_sweep(ks, K, cell, ncell, rho) for _, _, _, _, ks in terms if ks is not None}
        self.kcentres = sweeps
        self.cells, Rsum = [], Iv(ZERO)
        for c in range(ncell):
            s0 = (2 * c + 1) * half
            G, M = [Iv(ZERO)] * (K + 1), Iv(ZERO)
            for const, poly, gam, kap, ks in terms:
                pc = t_mul(poly_coeffs(poly, s0, K), gauss_coeffs(gam, s0, K))
                Mb = poly_bound(poly, s0, rho) * gauss_bound(gam, s0, rho)
                if kap is not None:
                    pc = t_mul(pc, erf_coeffs(kap, s0, K))
                    Mb = Mb * erf_bound(kap, s0, rho)
                if ks is not None:
                    d = kprime_coeffs(ks, s0, K)
                    kc = sweeps[ks][c]
                    pc = t_mul(pc, [kc] + [d[n] / (n + 1) for n in range(K)])
                    Mb = Mb * (Iv(kc.mag()) + Iv.frac(rho) * kprime_bound(ks, s0, rho))
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
        for const, poly, gam, kap, ks in terms:              # int_T^inf t^j e^{-g t^2} dt = (1/2) g^{-(j+1)/2} Gamma((j+1)/2, g T^2)
            fmax = Iv(ONE) if ks is None else (PI / Iv.frac(ks[1])).sqrt() / 2 * (PI / Iv.frac(ks[0])).sqrt() / 2
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
                tail = tail + Iv(const.mag()) * fmax * Iv.frac(abs(cj)) * up / gpow / 2
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


'''
