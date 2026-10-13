

def phi_pt(x):
    """phi(x) for rational x (interval)."""
    x = Fr(x)
    return exp_neg_point(x * x / 2) / SQRT2PI


def Phibar_pt(x):
    """1 - Phi(x) for rational x (interval)."""
    p = phi_point(Fr(x)) if abs(Fr(x)) <= 12 else None
    if p is None:
        if x > 0:
            # Mills: Phibar(x) <= phi(x)/x
            return IV(0, (phi_pt(x) / x).hi)
        return IV(1 - (phi_pt(-x) / (-x)).hi, 1)
    return IV(1 - p.hi, 1 - p.lo)


def upper_moments(x, J):
    """[M_0(x), ..., M_J(x)], M_j(x) = integral_x^inf t^j phi(t) dt, for rational x >= 0 (intervals)."""
    x = Fr(x)
    require(x >= 0, 'upper moment at negative point')
    ph = phi_pt(x)
    M = [Phibar_pt(x), ph]
    for j in range(2, J + 1):
        M.append(ph * x ** (j - 1) + M[j - 2] * (j - 1))
    return M


def dfact(n):
    """(n)!! for n >= -1."""
    return math.prod(range(n, 0, -2)) if n > 0 else 1


def rup_rel(x, bits=200):
    """Round a rational x >= 0 up to a dyadic with `bits` significant bits (upper bounds stay upper bounds)."""
    x = Fr(x)
    if x <= 0:
        return x
    e = x.numerator.bit_length() - x.denominator.bit_length()
    sh = bits - e
    if sh >= 0:
        return Fr(-((-x.numerator << sh) // x.denominator), 1 << sh)
    return Fr(-((-x.numerator) // (x.denominator << -sh)) << -sh)


def sqrt_up(x):
    """Rational upper bound of sqrt(x), x >= 0 rational, relative error below 2^-200 (no absolute rounding floor)."""
    x = Fr(x)
    if x == 0:
        return Fr(0)
    n, d = x.numerator, x.denominator
    nd = n * d
    m = max(0, (420 - nd.bit_length()) // 2 + 1)
    return rup_rel(Fr(math.isqrt(nd << (2 * m)) + 1, d << m))


def iroot(x, p):
    """Upper bound (rational) of x^(1/p) for rational x >= 0 and p a power of two."""
    v = Fr(x)
    while p > 1:
        v = sqrt_up(v)
        p //= 2
    return v


def pow2(p):
    q = 1
    while q < p:
        q *= 2
    return q


ZN = {}


def znorm(p):
    """Upper bound of ||N(0,1)||_p for p a power of two (p = 1 uses sqrt(2/pi) < 4/5)."""
    if p == 1:
        return Fr(4, 5)
    if p not in ZN:
        ZN[p] = iroot(Fr(dfact(p - 1)), p)
    return ZN[p]


def habs_moment(j):
    """E|zeta|^j (interval)."""
    if j % 2 == 0:
        return IV(dfact(j - 1))
    return SQRT2 / isqrt_iv(PI) * (2 ** ((j - 1) // 2) * math.factorial((j - 1) // 2))


# ----------------------------------------------------------------------------- functionals and exact covariance series
HALF = Fr(1, 2)
XM, XS, X0PT = -HALF, HALF, Fr(0)


def term(c, p, a, b, x0):
    """c r^p d_x^a d_y^b f(x0 r, 0)."""
    return (Fr(c), p, a, b, Fr(x0))


def functionals():
    pins = {
        'p1': [term(HALF, 0, 0, 0, XM), term(HALF, 0, 0, 0, XS)],
        'p2': [term(-1, -1, 0, 0, XM), term(1, -1, 0, 0, XS)],
        'p3p': [term(6, -2, 1, 0, XM), term(6, -2, 1, 0, XS), term(12, -3, 0, 0, XM), term(-12, -3, 0, 0, XS)],
        'p4': [term(-1, -1, 1, 0, XM), term(1, -1, 1, 0, XS)],
        'q1': [term(HALF, 0, 0, 1, XM), term(HALF, 0, 0, 1, XS)],
        'q2': [term(-1, -1, 0, 1, XM), term(1, -1, 0, 1, XS)],
    }
    targets = {
        'w': [term(1, 0, 0, 2, XM)],
        'v': [term(1, -1, 1, 1, XM), term(1, -2, 0, 1, XM), term(-1, -2, 0, 1, XS)],
        'om': [term(1, -1, 0, 2, XS), term(-1, -1, 0, 2, XM)],
        'Y': [term(1, 0, 0, 3, X0PT)],
        'T': [term(1, -2, 2, 0, XM), term(4, -3, 1, 0, XM), term(2, -3, 1, 0, XS), term(6, -4, 0, 0, XM), term(-6, -4, 0, 0, XS)],
        'tau': [term(1, -3, 2, 0, XS), term(-1, -3, 2, 0, XM), term(-6, -4, 1, 0, XM), term(-6, -4, 1, 0, XS),
                term(-12, -5, 0, 0, XM), term(12, -5, 0, 0, XS)],
        'nu': [term(1, -2, 1, 1, XM), term(1, -2, 1, 1, XS), term(2, -3, 0, 1, XM), term(-2, -3, 0, 1, XS)],
        'X40': [term(1, 0, 4, 0, X0PT)], 'X31': [term(1, 0, 3, 1, X0PT)], 'X22': [term(1, 0, 2, 2, X0PT)],
        'X13': [term(1, 0, 1, 3, X0PT)], 'X04': [term(1, 0, 0, 4, X0PT)],
    }
    return pins, targets


PIN_NAMES = ('p1', 'p2', 'p3p', 'p4', 'q1', 'q2')
TARGET_NAMES = ('w', 'v', 'om', 'Y', 'T', 'tau', 'nu', 'X40', 'X31', 'X22', 'X13', 'X04')
JET4 = ('X40', 'X31', 'X22', 'X13', 'X04')
JET4_IDX = {'X40': (4, 0), 'X31': (3, 1), 'X22': (2, 2), 'X13': (1, 3), 'X04': (0, 4)}


def weight(F):
    """Scaling weight a - (power of r), common to all terms of a preconditioned functional."""
    ws = {t[2] - t[1] for t in F}
    require(len(ws) == 1, 'functional not homogeneous')
    return ws.pop()


def hcoef(n):
    """d^n/dx^n exp(-x^2/2) at 0."""
    if n % 2:
        return 0
    return (-1) ** (n // 2) * dfact(n - 1)


PAIR_CACHE = {}


def pair_series(F, G, key):
    """Exact data of Cov_K(F, G) = sum_gamma kappa_gamma pi_gamma r^(gx - wF - wG):
    returns (reference series {power: rational}, weight w, list of (|c c'|, a+a', b+b') for the remainder bounds)."""
    if key in PAIR_CACHE:
        return PAIR_CACHE[key]
    wF, wG = weight(F), weight(G)
    w = wF + wG
    pi = {}
    absterms = []
    for (cf, pf, af, bf, xf) in F:
        for (cg, pg, ag, bg, xg) in G:
            gy = bf + bg
            sgn = (-1) ** (ag + bg)
            d = xf - xg
            absterms.append((abs(cf * cg), af + ag, gy))
            for gx in range(af + ag, NX + 1):
                m = gx - af - ag
                val = cf * cg * sgn * d ** m / math.factorial(m)
                if val != 0:
                    pi[(gx, gy)] = pi.get((gx, gy), Fr(0)) + val
    for (gx, gy), v in pi.items():
        require(v == 0 or gx - w >= 0, 'negative power of r in a covariance (regularity fails)')
    ser = {}
    for (gx, gy), v in pi.items():
        kap = hcoef(gx) * hcoef(gy)
        if kap and v:
            ser[gx - w] = ser.get(gx - w, Fr(0)) + kap * v
    out = ({p: c for p, c in ser.items() if c != 0}, w, absterms)
    PAIR_CACHE[key] = out
    return out


def torus_M1(L):
    """Bound of |sum_{n != 0} G(z + p_n)| on the polydisc |z_1|, |z_2| <= 1 (every frame, |p_n| = |n| L):
    exp(1) * sum_{j >= 1} 8 j exp(-(j L - 3/2)^2 / 2)."""
    L = Fr(L)
    tot = Fr(0)
    for j in range(1, 6):
        tot += 8 * j * exp_neg_point((j * L - Fr(3, 2)) ** 2 / 2).hi
    tail = 8 * 7 * exp_neg_point((6 * L - Fr(3, 2)) ** 2 / 2).hi * 4      # j >= 6, very crude
    return Fr(27183, 10000) * (tot + tail)


def theta_minus_one(L):
    L = Fr(L)
    tot = Fr(0)
    for j in range(1, 6):
        tot += 8 * j * exp_neg_point((j * L) ** 2 / 2).hi
    return tot + 8 * 7 * exp_neg_point((6 * L) ** 2 / 2).hi * 4


def cov_band(F, G, key, r0, r1, M1):
    """Enclosure of the covariance (unnormalized torus kernel sum_n G(z + p_n)) of F and G, for r in [r0, r1]."""
    ser, w, absterms = pair_series(F, G, key)
    val = IV(0)
    for p, c in ser.items():
        val = val + IV(r0 ** p, r1 ** p) * c
    A = max((t[1] for t in absterms), default=0)
    # Taylor truncation tail of the reference kernel (orders gx > NX)
    tail = Fr(0)
    for (acc, aa, gy) in absterms:
        tail += acc * abs(hcoef(gy))
    tail *= 2 * Fr(NX + 1) ** A * r1 ** (NX + 1 - w)
    # torus remainder: Cauchy estimates |rho_gamma| <= gx! gy! M1 on the unit polydisc
    tor = Fr(0)
    if True:
        for (acc, aa, gy) in absterms:
            tor += acc * math.factorial(gy) * Fr(max(w, 1)) ** aa
        tor = tor * M1 / (1 - Fr(27183, 10000) * r1)
    err = tail + tor
    return val + IV(-err, err)


# ----------------------------------------------------------------------------- interval linear algebra
def solve_iv(A, B):
    """Interval Gaussian elimination with partial pivoting on midpoints: X[c] solves A X[c] = B[c]."""
    n = len(A)
    M = [[A[i][j] for j in range(n)] + [B[c][i] for c in range(len(B))] for i in range(n)]
    for col in range(n):
        p = max(range(col, n), key=lambda i: abs(M[i][col].mid()))
        M[col], M[p] = M[p], M[col]
        piv = M[col][col]
        for i in range(col + 1, n):
            f = M[i][col] / piv
            for j in range(col, len(M[i])):
                M[i][j] = M[i][j] - f * M[col][j]
    X = []
    for c in range(len(B)):
        x = [None] * n
        for i in reversed(range(n)):
            s = M[i][n + c]
            for j in range(i + 1, n):
                s = s - M[i][j] * x[j]
            x[i] = s / M[i][i]
        X.append(x)
    return X


def cholesky3(C):
    """Interval Cholesky of a 3 x 3 SPD interval matrix."""
    L = [[IV(0)] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i + 1):
            s = C[i][j] - sum((L[i][l] * L[j][l] for l in range(j)), IV(0))
            if i == j:
                require(s.lo > 0, 'Cholesky pivot not positive')
                L[i][j] = isqrt_iv(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def absup(x):
    return max(abs(x.lo), abs(x.hi))


# ----------------------------------------------------------------------------- the conditional law on an r-band
class Law:
    pass


def band_law(r0, r1, L=L_MIN):
    """Interval enclosures, uniform over r in [r0, r1], every frame and every torus side >= L, of the Q-law of the
    targets: mean = b alpha + k beta, covariance C (normalized kernel), and the pin energy coefficients."""
    pins, targets = functionals()
    P = [pins[n] for n in PIN_NAMES]
    T = [targets[n] for n in TARGET_NAMES]
    M1 = torus_M1(L)
    th1 = theta_minus_one(L)
    nt = len(T)
    Spp = [[cov_band(P[i], P[j], ('p', i, j), r0, r1, M1) for j in range(6)] for i in range(6)]
    Stp = [[cov_band(T[i], P[j], ('tp', i, j), r0, r1, M1) for j in range(6)] for i in range(nt)]
    Stt = [[cov_band(T[i], T[j], ('tt', i, j), r0, r1, M1) for j in range(nt)] for i in range(nt)]
    r = IV(r0, r1)
    kvec = [-(r * r * r) * HALF, -(r * r), IV(12), IV(0), IV(0), IV(0)]
    e1 = [IV(1), IV(0), IV(0), IV(0), IV(0), IV(0)]
    W = solve_iv(Spp, Stp)
    xs = solve_iv(Spp, [e1, kvec])
    law = Law()
    law.r0, law.r1, law.L, law.M1, law.th1 = r0, r1, L, M1, th1
    law.alpha = {n: W[i][0] for i, n in enumerate(TARGET_NAMES)}
    law.beta = {n: sum((W[i][j] * kvec[j] for j in range(6)), IV(0)) for i, n in enumerate(TARGET_NAMES)}
    scale = IV(1 / (1 + th1), 1)          # K_L = (unnormalized) / Theta, Theta in [1, 1 + th1]
    C = {}
    for i, a in enumerate(TARGET_NAMES):
        for j, c in enumerate(TARGET_NAMES):
            if j < i:
                continue
            v = Stt[i][j] - sum((W[i][l] * Stp[j][l] for l in range(6)), IV(0))
            if i != j:
                v2 = Stt[j][i] - sum((W[j][l] * Stp[i][l] for l in range(6)), IV(0))
                v = v.meet(v2)
            C[(a, c)] = C[(c, a)] = v * scale
    law.C = C
    escale = IV(1, 1 + th1)
    law.e_bb = xs[0][0] * escale
    law.e_bk = sum((kvec[j] * xs[0][j] for j in range(6)), IV(0)) * escale
    law.e_kk = sum((kvec[j] * xs[1][j] for j in range(6)), IV(0)) * escale
    # regression on w and the g-parts
    sw2 = C[('w', 'w')]
    require(sw2.lo > 0, 'Var w not positive')
    law.sw2 = sw2
    law.sw = isqrt_iv(sw2)
    law.bw = {n: C[(n, 'w')] / sw2 for n in TARGET_NAMES if n != 'w'}
    Cg = {}
    for a in TARGET_NAMES:
        for c in TARGET_NAMES:
            if a != 'w' and c != 'w':
                Cg[(a, c)] = C[(a, c)] - C[(a, 'w')] * C[(c, 'w')] / sw2
    law.Cg = Cg
    law.L3 = cholesky3([[Cg[(a, c)] for c in ('v', 'om', 'Y')] for a in ('v', 'om', 'Y')])
    law.sdg = {n: isqrt_iv(IV(max(Cg[(n, n)].lo, Fr(0)), Cg[(n, n)].hi)) if Cg[(n, n)].lo > 0 else IV(0, isqrt_iv(IV(Cg[(n, n)].hi)).hi)
               for n in TARGET_NAMES if n != 'w'}
    law.sdQ = {n: isqrt_iv(C[(n, n)]) for n in TARGET_NAMES}
    # crude higher jets: Taylor-tail coefficients S_alpha(r1) = sum_{beta != 0} sd0_{alpha+beta} (2 r1)^{|beta|} / beta!
    law.S = {n: taylor_tail_coefficient(JET4_IDX[n], r1, M1) for n in JET4}
    return law


def sd0_bound(gx, gy, M1):
    """Upper bound of sqrt(Var d^gamma f(0)) = sqrt(|d^{2 gamma} K_L(0)|) (normalized, every frame, L >= L_MIN)."""
    v = dfact(2 * gx - 1) * dfact(2 * gy - 1) + math.factorial(2 * gx) * math.factorial(2 * gy) * M1
    return isqrt_iv(IV(v)).hi


def taylor_tail_coefficient(alpha, r1, M1):
    ax, ay = alpha
    tot = Fr(0)
    for n in range(1, NB + 1):
        for bx in range(n + 1):
            by = n - bx
            tot += sd0_bound(ax + bx, ay + by, M1) * (2 * r1) ** n / (math.factorial(bx) * math.factorial(by))
    # tail |beta| > NB: sd0_gamma <= 2^|gamma| gx! gy! sqrt(1 + M1) (binom(2m, m) <= 4^m), so a term is at most
    # 2^|alpha| (4 + n)^4 (4 r1)^n 2 per multi-index; n + 1 multi-indices of order n
    q = 4 * r1
    require(q <= Fr(1, 16), 'Taylor tail ratio too large')
    tail = Fr(0)
    base = 2 ** (ax + ay) * 2
    n = NB + 1
    # sum_{n > NB} (n + 1)(n + 4)^8 q^n <= (NB + 2)(NB + 5)^8 q^(NB+1) / (1 - 2 q)   (consecutive ratio <= 2q for these n)
    tail = base * (NB + 2) * Fr(NB + 5) ** 8 * q ** (NB + 1) / (1 - 2 * q)
    return tot + tail
