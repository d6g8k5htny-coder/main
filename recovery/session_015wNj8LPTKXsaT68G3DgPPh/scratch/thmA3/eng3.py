

# ----------------------------------------------------------------------------- functionals and exact covariance series (d = 3)
HALF = Fr(1, 2)
XM, XS, X0PT = -HALF, HALF, Fr(0)
JET4_IDX = {'X%d%d%d' % (a, b1, 4 - a - b1): (a, b1, 4 - a - b1) for a in range(5) for b1 in range(5 - a)}
JET4 = tuple(sorted(JET4_IDX, key=lambda n: (-JET4_IDX[n][0], -JET4_IDX[n][1])))
BLOCKS = ((4, 0), (3, 1), (2, 2), (1, 3), (0, 4))           # partial blocks (a, c) of the fourth derivatives, a + c = 4


def block_members(a, c):
    """[(name, Frobenius multiplicity c!/beta!)] of the 4-jet block d_x^a D_y^c f(0)."""
    out = []
    for b1 in range(c, -1, -1):
        b2 = c - b1
        out.append(('X%d%d%d' % (a, b1, b2), math.factorial(c) // (math.factorial(b1) * math.factorial(b2))))
    return out


def term(c, p, a, b1, b2, x0):
    """c r^p d_x^a d_y1^b1 d_y2^b2 f(x0 r, 0, 0)."""
    return (Fr(c), p, a, b1, b2, Fr(x0))


def functionals():
    pins = {
        'p1': [term(HALF, 0, 0, 0, 0, XM), term(HALF, 0, 0, 0, 0, XS)],
        'p2': [term(-1, -1, 0, 0, 0, XM), term(1, -1, 0, 0, 0, XS)],
        'p3p': [term(6, -2, 1, 0, 0, XM), term(6, -2, 1, 0, 0, XS), term(12, -3, 0, 0, 0, XM), term(-12, -3, 0, 0, 0, XS)],
        'p4': [term(-1, -1, 1, 0, 0, XM), term(1, -1, 1, 0, 0, XS)],
    }
    for j, (e1, e2) in ((1, (1, 0)), (2, (0, 1))):
        pins['q1%d' % j] = [term(HALF, 0, 0, e1, e2, XM), term(HALF, 0, 0, e1, e2, XS)]
        pins['q2%d' % j] = [term(-1, -1, 0, e1, e2, XM), term(1, -1, 0, e1, e2, XS)]
    targets = {
        'a11': [term(1, 0, 0, 2, 0, XM)], 'a22': [term(1, 0, 0, 0, 2, XM)], 'a12': [term(1, 0, 0, 1, 1, XM)],
        'o11': [term(1, -1, 0, 2, 0, XS), term(-1, -1, 0, 2, 0, XM)],
        'o22': [term(1, -1, 0, 0, 2, XS), term(-1, -1, 0, 0, 2, XM)],
        'o12': [term(1, -1, 0, 1, 1, XS), term(-1, -1, 0, 1, 1, XM)],
        'T': [term(1, -2, 2, 0, 0, XM), term(4, -3, 1, 0, 0, XM), term(2, -3, 1, 0, 0, XS), term(6, -4, 0, 0, 0, XM),
              term(-6, -4, 0, 0, 0, XS)],
        'tau': [term(1, -3, 2, 0, 0, XS), term(-1, -3, 2, 0, 0, XM), term(-6, -4, 1, 0, 0, XM), term(-6, -4, 1, 0, 0, XS),
                term(-12, -5, 0, 0, 0, XM), term(12, -5, 0, 0, 0, XS)],
    }
    for j, (e1, e2) in ((1, (1, 0)), (2, (0, 1))):
        targets['v%d' % j] = [term(1, -1, 1, e1, e2, XM), term(1, -2, 0, e1, e2, XM), term(-1, -2, 0, e1, e2, XS)]
        targets['nu%d' % j] = [term(1, -2, 1, e1, e2, XM), term(1, -2, 1, e1, e2, XS), term(2, -3, 0, e1, e2, XM),
                               term(-2, -3, 0, e1, e2, XS)]
    for b1 in range(3, -1, -1):
        targets['y%s' % ('1' * b1 + '2' * (3 - b1))] = [term(1, 0, 0, b1, 3 - b1, X0PT)]
    for n, (a, b1, b2) in JET4_IDX.items():
        targets[n] = [term(1, 0, a, b1, b2, X0PT)]
    return pins, targets


PIN_NAMES = ('p1', 'p2', 'p3p', 'p4', 'q11', 'q21', 'q12', 'q22')
BNAMES = ('a11', 'a22', 'a12')                              # the transverse Hessian A = D_y^2 f(M) = -B
VNAMES = ('v1', 'v2')
ONAMES = ('o11', 'o22', 'o12')
YNAMES = ('y111', 'y112', 'y122', 'y222')
FREE = VNAMES + ONAMES + YNAMES                              # the free third-derivative blocks, in Cholesky order
FREE_W = (1, 1, 1, 1, 2, 1, 3, 3, 1)                         # Frobenius multiplicities (Euclidean norms in these weights)
TARGET_NAMES = BNAMES + FREE + ('T', 'tau', 'nu1', 'nu2') + JET4
NP = len(PIN_NAMES)


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
    """Exact data of Cov_K(F, G) = sum_gamma kappa_gamma pi_gamma r^(gx - wF - wG), gamma = (gx, gy1, gy2):
    returns (reference series {power: rational}, weight w, list of (|c c'|, a+a', gy1, gy2) for the remainder bounds)."""
    if key in PAIR_CACHE:
        return PAIR_CACHE[key]
    wF, wG = weight(F), weight(G)
    w = wF + wG
    pi = {}
    absterms = []
    for (cf, pf, af, bf1, bf2, xf) in F:
        for (cg, pg, ag, bg1, bg2, xg) in G:
            g1, g2 = bf1 + bg1, bf2 + bg2
            sgn = (-1) ** (ag + bg1 + bg2)
            d = xf - xg
            absterms.append((abs(cf * cg), af + ag, g1, g2))
            for gx in range(af + ag, NX + 1):
                m = gx - af - ag
                val = cf * cg * sgn * d ** m / math.factorial(m)
                if val != 0:
                    pi[(gx, g1, g2)] = pi.get((gx, g1, g2), Fr(0)) + val
    for (gx, g1, g2), v in pi.items():
        require(v == 0 or gx - w >= 0, 'negative power of r in a covariance (regularity fails)')
    ser = {}
    for (gx, g1, g2), v in pi.items():
        kap = hcoef(gx) * hcoef(g1) * hcoef(g2)
        if kap and v:
            ser[gx - w] = ser.get(gx - w, Fr(0)) + kap * v
    out = ({p: c for p, c in ser.items() if c != 0}, w, absterms)
    PAIR_CACHE[key] = out
    return out


E_UP = Fr(27183, 10000)                                      # > e
SQRT3_UP = Fr(7, 4)                                          # > sqrt(3)


def torus_M1(L):
    """Bound of |sum_{n != 0} G(z + p_n)| on the polydisc |z_1|, |z_2|, |z_3| <= 1 (every frame, |p_n| = |n| L):
    exp(3/2) sum_{j >= 1} (24 j^2 + 2) exp(-(j L - sqrt 3)^2 / 2); 24 j^2 + 2 lattice points have sup-norm j."""
    L = Fr(L)
    e32 = exp_neg_point(Fr(3, 2)).inv().hi
    tot = Fr(0)
    for j in range(1, 6):
        tot += (24 * j * j + 2) * exp_neg_point((j * L - SQRT3_UP) ** 2 / 2).hi
    # j >= 6: (24 j^2 + 2) exp(-(jL - 7/4)^2/2) decreases by more than 1/2 per step for L >= 10
    tail = 2 * (24 * 36 + 2) * exp_neg_point((6 * L - SQRT3_UP) ** 2 / 2).hi
    return e32 * (tot + tail)


def theta_minus_one(L):
    L = Fr(L)
    tot = Fr(0)
    for j in range(1, 6):
        tot += (24 * j * j + 2) * exp_neg_point((j * L) ** 2 / 2).hi
    return tot + 2 * (24 * 36 + 2) * exp_neg_point((6 * L) ** 2 / 2).hi


def cov_band(F, G, key, r0, r1, M1):
    """Enclosure of the covariance (unnormalized torus kernel sum_n G(z + p_n)) of F and G, for r in [r0, r1]."""
    ser, w, absterms = pair_series(F, G, key)
    val = IV(0)
    for p, c in ser.items():
        val = val + IV(r0 ** p, r1 ** p) * c
    A = max((t[1] for t in absterms), default=0)
    # Taylor truncation tail of the reference kernel (orders gx > NX)
    tail = Fr(0)
    for (acc, aa, g1, g2) in absterms:
        tail += acc * abs(hcoef(g1) * hcoef(g2))
    tail *= 2 * Fr(NX + 1) ** A * r1 ** (NX + 1 - w)
    # torus remainder: Cauchy estimates |rho_gamma| <= gx! gy1! gy2! M1 on the unit polydisc
    tor = Fr(0)
    for (acc, aa, g1, g2) in absterms:
        tor += acc * math.factorial(g1) * math.factorial(g2) * Fr(max(w, 1)) ** aa
    tor = tor * M1 / (1 - E_UP * r1)
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


def cholesky(C):
    """Interval Cholesky of an n x n SPD interval matrix (pivots must be positive)."""
    n = len(C)
    L = [[IV(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = C[i][j] - sum((L[i][l] * L[j][l] for l in range(j)), IV(0))
            if i == j:
                require(s.lo > 0, 'Cholesky pivot not positive')
                L[i][j] = isqrt_iv(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def opnorm_lower_tri(L, rows, cols):
    """Upper bound of the operator norm of the block L[rows][cols] of an interval lower-triangular matrix:
    max |diagonal| + Frobenius norm of the rest (square diagonal blocks), Frobenius norm otherwise."""
    if rows == cols:
        dg = max(absup(L[i][i]) for i in rows)
        off = sum(absup(L[i][j]) ** 2 for i in rows for j in cols if j < i)
        return rup_rel(dg + sqrt_up(off))
    return sqrt_up(sum(absup(L[i][j]) ** 2 for i in rows for j in cols))


def sym3_eig_bounds(C):
    """(lower bound of lambda_min, upper bound of lambda_max) of a symmetric 3 x 3 interval matrix (Gershgorin)."""
    lo = min(C[i][i].lo - sum(absup(C[i][j]) for j in range(3) if j != i) for i in range(3))
    hi = max(C[i][i].hi + sum(absup(C[i][j]) for j in range(3) if j != i) for i in range(3))
    return lo, hi


def det3(C):
    a, b, c = C[0]
    d, e, f = C[1]
    g, h, i = C[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


# ----------------------------------------------------------------------------- the conditional law on an r-band
class Law:
    pass


def band_law(r0, r1, L=L_MIN):
    """Interval enclosures, uniform over r in [r0, r1], every frame and every torus side >= L, of the Q-law of the
    targets: mean = b alpha + k beta, covariance C (normalized kernel), and the pin energy coefficients; then the
    regression on the transverse Hessian (three entries) and the g-law of the remaining targets."""
    pins, targets = functionals()
    P = [pins[n] for n in PIN_NAMES]
    T = [targets[n] for n in TARGET_NAMES]
    M1 = torus_M1(L)
    th1 = theta_minus_one(L)
    nt = len(T)
    Spp = [[None] * NP for _ in range(NP)]
    for i in range(NP):
        for j in range(i, NP):
            Spp[i][j] = Spp[j][i] = cov_band(P[i], P[j], ('p', i, j), r0, r1, M1)
    Stp = [[cov_band(T[i], P[j], ('tp', i, j), r0, r1, M1) for j in range(NP)] for i in range(nt)]
    Stt = [[None] * nt for _ in range(nt)]
    for i in range(nt):
        for j in range(i, nt):
            Stt[i][j] = Stt[j][i] = cov_band(T[i], T[j], ('tt', i, j), r0, r1, M1)
    r = IV(r0, r1)
    kvec = [-(r * r * r) * HALF, -(r * r), IV(12)] + [IV(0)] * (NP - 3)
    e1 = [IV(1)] + [IV(0)] * (NP - 1)
    W = solve_iv(Spp, Stp)
    xs = solve_iv(Spp, [e1, kvec])
    law = Law()
    law.r0, law.r1, law.L, law.M1, law.th1 = r0, r1, L, M1, th1
    law.alpha = {n: W[i][0] for i, n in enumerate(TARGET_NAMES)}
    law.beta = {n: sum((W[i][j] * kvec[j] for j in range(NP)), IV(0)) for i, n in enumerate(TARGET_NAMES)}
    scale = IV(1 / (1 + th1), 1)          # K_L = (unnormalized) / Theta, Theta in [1, 1 + th1]
    C = {}
    for i, a in enumerate(TARGET_NAMES):
        for j in range(i, nt):
            c = TARGET_NAMES[j]
            v = Stt[i][j] - sum((W[i][l] * Stp[j][l] for l in range(NP)), IV(0))
            C[(a, c)] = C[(c, a)] = v * scale
    law.C = C
    escale = IV(1, 1 + th1)
    law.e_bb = xs[0][0] * escale
    law.e_bk = sum((kvec[j] * xs[0][j] for j in range(NP)), IV(0)) * escale
    law.e_kk = sum((kvec[j] * xs[1][j] for j in range(NP)), IV(0)) * escale
    regress_on_B(law)
    law.S = {blk: taylor_tail_coefficient(blk, r1, M1) for blk in BLOCKS}
    law.net_var = {blk: net_max_variance(law.C, blk) for blk in BLOCKS if blk[1] >= 2}
    return law


NET_M = 128                                # rational unit vectors ((1 - t^2), 2 t)/(1 + t^2), t = j/NET_M, and their rotations


def circle_net():
    """Rational unit vectors u_j covering angles [0, pi) with consecutive angular gaps <= 2/NET_M (theta = 2 arctan t has
    d theta/dt <= 2), together with the rotated copies (-u2, u1)."""
    pts = []
    for j in range(NET_M + 1):
        t = Fr(j, NET_M)
        pts.append(((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)))
    return pts + [(-u2, u1) for (u1, u2) in pts]


NET = circle_net()


def block_form_weights(blk, u):
    """w_beta = (c!/beta!) u1^beta1 u2^beta2: d_x^a d_u^c f(0) = sum_beta w_beta X_(a,beta)."""
    a, c = blk
    return [(n, m * u[0] ** int(n[2]) * u[1] ** int(n[3])) for n, m in block_members(a, c)]


def net_max_variance(C, blk):
    """Upper bound of max_j Var_Q(d_x^a d_(u_j)^c f(0)) over the net."""
    best = Fr(0)
    for u in NET:
        w = block_form_weights(blk, u)
        v = sum(abs(wi) * abs(wj) * absup(C[(ni, nj)]) if ni != nj else wi * wi * C[(ni, ni)].hi
                for ni, wi in w for nj, wj in w)
        best = max(best, v)
    return best


FROB_B = (IV(1), IV(1), SQRT2)               # (B11, B22, B12) -> Frobenius-isometric coordinates (B11, B22, sqrt2 B12)


def regress_on_B(law):
    """Regression of every target on the three entries of the transverse Hessian, in the Frobenius-isometric coordinates
    b~ = (B11, B22, sqrt2 B12) of B = -A: F = F_g + beta~_F . (b~ - mu~), with F_g independent of B."""
    C = law.C
    # covariance of b~ (B = -A: signs cancel in the covariance)
    Cb = [[C[(BNAMES[i], BNAMES[j])] * FROB_B[i] * FROB_B[j] for j in range(3)] for i in range(3)]
    law.Cb = Cb
    law.detCb = det3(Cb)
    require(law.detCb.lo > 0, 'transverse Hessian covariance not positive')
    lo, hi = sym3_eig_bounds(Cb)
    require(lo > 0, 'transverse Hessian covariance eigenvalue bound')
    law.sB_lo2, law.sB_hi2 = lo, hi
    law.sB_lo = sqrt_down(lo)
    # regression coefficients of F on b~: beta~_F = Cov(F, b~) Cb^-1, Cov(F, b~_i) = -Cov(F, A_i) FROB_B[i]
    others = [n for n in TARGET_NAMES if n not in BNAMES]
    rhs = [[-C[(n, BNAMES[i])] * FROB_B[i] for i in range(3)] for n in others]
    sol = solve_iv(Cb, rhs)
    law.bt = {n: sol[i] for i, n in enumerate(others)}
    law.bnorm = {n: sqrt_up(sum(absup(x) ** 2 for x in law.bt[n])) for n in others}
    Cg = {}
    for a in others:
        for c in others:
            Cg[(a, c)] = C[(a, c)] - sum((law.bt[a][i] * (-C[(c, BNAMES[i])] * FROB_B[i]) for i in range(3)), IV(0))
    law.Cg = Cg
    # the free blocks in Frobenius coordinates: v, omega = (o11, o22, sqrt2 o12), Y~ = (y111, sqrt3 y112, sqrt3 y122, y222)
    fw = [isqrt_iv(IV(w)) if w != 1 else IV(1) for w in FREE_W]
    law.Lfree = cholesky([[Cg[(FREE[i], FREE[j])] * fw[i] * fw[j] for j in range(9)] for i in range(9)])
    law.fw = fw
    law.sdg = {n: sqrt_up(max(Cg[(n, n)].hi, Fr(0))) for n in others}
    law.sdQ = {n: sqrt_up(C[(n, n)].hi) for n in TARGET_NAMES}


def sd0_sq(g, M1):
    """Upper bound of Var d^g f(0) = |d^(2g) K_L(0)| (normalized kernel, every frame, L >= L_MIN), g = (gx, gy1, gy2):
    the reference value prod (2 g_i - 1)!! plus the Cauchy bound (2g)! M1 of the torus remainder."""
    return math.prod(dfact(2 * x - 1) for x in g) + math.prod(math.factorial(2 * x) for x in g) * M1


def taylor_tail_coefficient(blk, r1, M1):
    """S_(a,c)(r1) = sum_{gamma != 0} sigma_(a,c),gamma (2 r1)^|gamma| / gamma!, sigma^2 = sum_beta (c!/beta!) s^2_(a+gx, beta+gy):
    the Frobenius norm of the Taylor tail of the block d_x^a D_y^c f at the midpoint over D."""
    a, c = blk
    tot = Fr(0)
    for n in range(1, NB + 1):
        for gx in range(n + 1):
            for g1 in range(n - gx + 1):
                g2 = n - gx - g1
                s2 = Fr(0)
                for b1 in range(c + 1):
                    b2 = c - b1
                    mult = math.factorial(c) // (math.factorial(b1) * math.factorial(b2))
                    s2 += mult * sd0_sq((a + gx, b1 + g1, b2 + g2), M1)
                tot += sqrt_up(s2) * (2 * r1) ** n / (math.factorial(gx) * math.factorial(g1) * math.factorial(g2))
    # tail |gamma| = n > NB: s_delta <= 2^|delta| delta! sqrt(1 + M1) and (alpha + gamma)!/gamma! <= (n + 4)^4, so
    # sigma (2 r1)^n / gamma! <= 2^(c/2) 2^(4 + n) (n + 4)^4 (2 r1)^n sqrt(1 + M1); (n+1)(n+2)/2 multi-indices of order n;
    # the ratio of consecutive bounds is below 2 q for these n (q = 4 r1 <= 1/16)
    q = 4 * r1
    require(q <= Fr(1, 16), 'Taylor tail ratio too large')
    n = NB + 1
    base = 2 * 2 ** 4 * sqrt_up(Fr(2) ** c * (1 + M1))
    tail = base * Fr((n + 1) * (n + 2), 2) * Fr(n + 4) ** 4 * q ** n / (1 - 2 * q)
    return rup_rel(tot + tail)
