
# ---------------- the d = 3 jet of the reference kernel exp(-|z|^2/2), frame (u, w1, w2)
# Cov(d^a f, d^b f) = (-1)^|a| d^(a+b) K(0), and d^al exp(-|z|^2/2) at 0 = prod_i (-1)^(al_i/2) (al_i - 1)!! for even al, else 0.
EVEN = [(0, 0, 0), (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]   # f, f_uu, f_uw1, f_uw2, A11, A12, A22
ODD = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (3, 0, 0)]                                    # f_u, f_w1, f_w2, t = f_uuu
IDX_X, IDX_V = [0, 4, 5, 6], [1, 2, 3]                                                 # (f, A11, A12, A22) and V = (f_uu, f_uw1, f_uw2)


def dfact2(n):
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def ref_cov(a, b):
    al = [x + y for x, y in zip(a, b)]
    if any(x % 2 for x in al):
        return Fr(0)
    v = Fr((-1) ** sum(a))
    for x in al:
        v *= (-1) ** (x // 2) * dfact2(x - 1)
    if MUT == "tau-cross" and sorted([a, b]) == sorted([(3, 0, 0), (1, 0, 0)]):
        return Fr(0)
    return v


def ref_matrix(idx):
    return [[ref_cov(a, b) for b in idx] for a in idx]


def det3(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


def inv3(M):
    """Inverse by the adjugate; works for Fractions and for intervals."""
    d = det3(M)
    cof = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            r = [x for x in range(3) if x != i]
            c = [y for y in range(3) if y != j]
            minor = M[r[0]][c[0]] * M[r[1]][c[1]] - M[r[0]][c[1]] * M[r[1]][c[0]]
            cof[i][j] = minor if (i + j) % 2 == 0 else -minor
    return [[cof[j][i] / d for j in range(3)] for i in range(3)], d


def schur(C, X, V):
    """Cov(X | V) = C_XX - C_XV C_VV^-1 C_VX for index lists into C (Fractions or intervals)."""
    CVV = [[C[i][j] for j in V] for i in V]
    Vinv, detV = inv3(CVV)
    out = []
    for i in X:
        row = []
        for j in X:
            s = None
            for p in range(3):
                for q in range(3):
                    t = C[i][V[p]] * Vinv[p][q] * C[V[q]][j]
                    s = t if s is None else s + t
            row.append(C[i][j] + s if MUT == "schur-sign" else C[i][j] - s)
        out.append(row)
    return out, detV


def sylvester_pd(M):
    """Exact positive definiteness of a symmetric rational matrix by its leading principal minors (fraction-free elimination)."""
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


# exact reference: C'_ref = Cov((f, A11, A12, A22) | V = 0) and its smallest-eigenvalue floor
C_EVEN_REF = ref_matrix(EVEN)
C_ODD_REF = ref_matrix(ODD)
CPRIME_REF, DETV_REF = schur(C_EVEN_REF, IDX_X, IDX_V)
LAMBDA0 = Fr(1) if MUT == "lambda-min" else Fr(1, 3)


def lambda_floor_ok():
    """C'_ref - LAMBDA0 I is positive definite (exact), so C'_ref >= LAMBDA0 I; the smallest eigenvalue is 2 - sqrt(8/3)."""
    M = [[CPRIME_REF[i][j] - (LAMBDA0 if i == j else 0) for j in range(4)] for i in range(4)]
    return sylvester_pd(M)


# ---------------- image bound for the torus kernel in d = 3
def image_bound3(L):
    """E^(3)_L = 1404 (76 L^6 + 15) exp(-L^2/2): the shell |n|_inf = j carries (2j+1)^3 - (2j-1)^3 = 24 j^2 + 2 <= 26 j^2 points
    with |n|^2 <= 3 j^2, so sum_{n != 0} |n|^6 exp(-L^2 |n|^2/2) <= 702 sum_j j^8 exp(-L^2 j^2/2) <= 1404 exp(-L^2/2) (successive
    terms ratio at most 2^8 exp(-3L^2/2) < 1/2 for L >= 10); the contraction bound 76|x|^6 phi(x) and the normalization S >= 1
    are those of Math-#197 section 3."""
    shells = 13 if MUT == "image-shells" else 26
    L = Fr(L)
    return Iv.frac(2 * shells * 27 * (76 * L ** 6 + 15)) * (-Iv.frac(L * L / 2)).exp()


# ---------------- the d = 3 pipeline over an entry box
class Space3:
    """Every covariance entry of the 3-jet (other than Var f = 1, exact by the normalization) in [ref - E, ref + E].
    Odd block (G, t) independent of the even block (f, V, A) for any even kernel ([LP] section 15)."""

    def __init__(self, E):
        E = Iv.of(E)
        def box(M):
            n = len(M)
            B = [[None] * n for _ in range(n)]
            for i in range(n):
                for j in range(i, n):
                    v = Iv.frac(M[i][j]) + Iv(-E.hi, E.hi)
                    B[i][j] = B[j][i] = v
            return B
        CE_ = box(C_EVEN_REF)
        CE_[0][0] = Iv(ONE)
        CO = box(C_ODD_REF)
        G = [[CO[i][j] for j in range(3)] for i in range(3)]
        Ginv, detG = inv3(G)
        c = [CO[3][j] for j in range(3)]
        self.tau2 = CO[3][3] - sum((c[i] * Ginv[i][j] * c[j] for i in range(3) for j in range(3)), Iv(ZERO))
        self.pG = 1 / (2 * PI * SQRT2PI * detG.sqrt())                        # (2 pi)^(-3/2) det(Cov G)^(-1/2)
        self.Cp, detV = schur(CE_, IDX_X, IDX_V)
        self.pV = 1 / (2 * PI * SQRT2PI * detV.sqrt())
        self.tau43 = root_iv(self.tau2.sq(), 3)
        # epsilon: ||C_ref'^(-1/2) (C' - C_ref') C_ref'^(-1/2)|| <= ||C' - C_ref'||_F / LAMBDA0, entrywise sup of |C'_ij - ref|
        dev2 = Iv(ZERO)
        for i in range(4):
            for j in range(4):
                r = Iv.frac(CPRIME_REF[i][j])
                d = max(abs(CC.subtract(self.Cp[i][j].hi, r.lo)), abs(CF.subtract(self.Cp[i][j].lo, r.hi)))
                dev2 = dev2 + Iv(d).sq()
        self.eps = (dev2.sqrt() / Iv.frac(LAMBDA0)).hi

    def k_integral(self, K):
        lo, hi = K
        def s(k):
            return k if isinstance(k, float) else 72 * Iv.frac(k * k) / self.tau2
        a = SHAPE()
        return self.tau43 * C72 * (gamma_lower(s(hi), a) - gamma_lower(s(lo), a)) / (2 * SQRT2PI)

    def b_integral(self, B):
        return sandwich(B, self.eps)

    def coefficient(self, B, K):
        sphere = 2 * PI if MUT == "sphere" else 4 * PI
        return 144 * sphere * self.pG * self.pV * self.b_integral(B) * self.k_integral(K)


# ---------------- the d = 3 reference birth integral with interval endpoints (Math-#197 (4.1) and its three primitives)
def endpoint(b):
    return b if isinstance(b, (float, Iv)) else Iv.frac(b)


def I3_ref(lo, hi):
    """int_lo^hi phi_{2/3}(b) m_{3,b} db, m_{3,b} = (b^3 + b) phi(b) + (b^4 + 2 b^2 + 7) Phi(b) - 4 sqrt2 exp(-b^2/4) Phi(b/sqrt2);
    lo, hi are rationals, intervals or +-INF."""
    sig = SIGMA_REF
    def term_i(b):
        if isinstance(b, float):
            return Iv(ZERO)
        return -(b.sq() * 2 / 5 + Iv.frac(Fr(18, 25))) * (-(b.sq()) * 5 / 4).exp() / (2 * PI * sig)
    def term_ii(b):
        G = G_at(b if isinstance(b, float) else b / sig, sig)
        return sig.sq().sq() * G[4] + 2 * sig.sq() * G[2] + 7 * G[0]
    def term_iii(b):
        G = G_at(b if isinstance(b, float) else b * SQRT2, Iv.frac(Fr(1, 2)))
        return 4 / sig * G[0]
    lo, hi = endpoint(lo), endpoint(hi)
    cross = Iv.frac(Fr(1, 2)) if MUT == "d3-cross" else Iv(ONE)
    return (term_i(hi) - term_i(lo)) + (term_ii(hi) - term_ii(lo)) - cross * (term_iii(hi) - term_iii(lo))


def scaled(b, s):
    if isinstance(b, float):
        return b
    b = endpoint(b)
    return b * s if MUT == "window-scale" else b / s


def sandwich(B, eps):
    """I'_B for every covariance C' of (f, A) | V=0 with (1-eps) C_ref' <= C' <= (1+eps) C_ref':
    (1-eps)^4/(1+eps)^2 I^ref_{B/sqrt(1-eps)} <= I'_B <= (1+eps)^4/(1-eps)^2 I^ref_{B/sqrt(1+eps)}  (NOTE section 2)."""
    e = Iv(eps) if isinstance(eps, Decimal) else Iv.frac(eps)
    if e.hi >= 1:
        raise ValueError("eps must be below 1")
    up, dn = 1 + e, 1 - e
    s_up, s_dn = up.sqrt(), dn.sqrt()
    lo, hi = B
    su, sd = (s_dn, s_up) if MUT == "sandwich-swap" else (s_up, s_dn)
    upper = up.sq().sq() / dn.sq() * I3_ref(scaled(lo, su), scaled(hi, su))
    lower = dn.sq().sq() / up.sq() * I3_ref(scaled(lo, sd), scaled(hi, sd))
    if lower.lo > upper.hi:
        raise ValueError("sandwich bounds out of order")
    return Iv(lower.lo, upper.hi)


def scaled_exact(B, eta):
    """Exact I'_B for C' = (1 + eta) C_ref': (1 + eta)^2 I^ref_{B/sqrt(1+eta)} (degree-4 homogeneity of det(A)^2)."""
    s = (1 + Iv.frac(eta)).sqrt()
    lo, hi = B
    return (1 + Iv.frac(eta)).sq() * I3_ref(lo if isinstance(lo, float) else endpoint(lo) / s, hi if isinstance(hi, float) else endpoint(hi) / s)


# ---------------- float control for a non-scalar perturbation (not part of the certificate)
def gauss_legendre(n):
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


def cone_moment_float(mu, Sig, b, nodes):
    """E[det(A)^2 1{A < 0}] for A = [[a, c], [c, d]], (a, c, d) ~ N(b mu, Sig): the c-integral over |c| < sqrt(ad) in closed form
    (truncated normal moments of orders 0, 2, 4), then Gauss-Legendre over a, d in (-R, 0)."""
    xs, ws = nodes
    ma, mc, md = (b * m for m in mu)
    saa, sac, sad = Sig[0]
    scc, scd, sdd = Sig[1][1], Sig[1][2], Sig[2][2]
    det2 = saa * sdd - sad * sad
    ia, iad, idd = sdd / det2, -sad / det2, saa / det2
    ka, kd = (sac * sdd - scd * sad) / det2, (scd * saa - sac * sad) / det2           # regression of c on (a, d)
    vc = scc - (ka * sac + kd * scd)
    sv = math.sqrt(vc)
    R = 12 * math.sqrt(max(saa, sdd)) + abs(ma) + abs(md)
    tot = 0.0
    for xa, wa in zip(xs, ws):
        a = -R * (xa + 1) / 2
        for xd, wd in zip(xs, ws):
            d = -R * (xd + 1) / 2
            ya, yd = a - ma, d - md
            q = ia * ya * ya + 2 * iad * ya * yd + idd * yd * yd
            dens = math.exp(-q / 2) / (2 * math.pi * math.sqrt(det2))
            RR = a * d
            r = math.sqrt(RR)
            m = mc + ka * ya + kd * yd
            z1, z2 = (-r - m) / sv, (r - m) / sv
            p1, p2 = fphi(z1), fphi(z2)
            mom = [fPhi(z2) - fPhi(z1), p1 - p2]
            for j in range(2, 5):
                mom.append((j - 1) * mom[j - 2] + z1 ** (j - 1) * p1 - z2 ** (j - 1) * p2)
            def M(k):
                return sum(math.comb(k, j) * m ** (k - j) * sv ** j * mom[j] for j in range(k + 1))
            tot += wa * wd * dens * (RR * RR * M(0) - 2 * RR * M(2) + M(4))
    return tot * (R / 2) ** 2


def I_float(Cp, B, nb=40, na=48):
    """int_B p_f(b) E[det(A)^2 1{A<0} | f = b] db for the covariance Cp of (f, A11, A12, A22) (floats)."""
    s2 = Cp[0][0]
    mu = [Cp[i][0] / s2 for i in (1, 2, 3)]
    Sig = [[Cp[i][j] - Cp[i][0] * Cp[0][j] / s2 for j in (1, 2, 3)] for i in (1, 2, 3)]
    nodes = gauss_legendre(na)
    xs, ws = gauss_legendre(nb)
    lo, hi = float(B[0]), float(B[1])
    tot = 0.0
    for x, w in zip(xs, ws):
        b = lo + (hi - lo) * (x + 1) / 2
        tot += w * math.exp(-b * b / (2 * s2)) / math.sqrt(2 * math.pi * s2) * cone_moment_float(mu, Sig, b, nodes)
    return tot * (hi - lo) / 2


CONTROL_DELTA = {(0, 5): Fr(1, 1000), (4, 4): Fr(2, 1000), (4, 6): Fr(1, 1000), (1, 5): Fr(-1, 1000), (2, 4): Fr(1, 1000)}


def control_matrices():
    C = [list(r) for r in C_EVEN_REF]
    for (i, j), v in CONTROL_DELTA.items():
        C[i][j] += v
        if i != j:
            C[j][i] += v
    Cp, _ = schur(C, IDX_X, IDX_V)
    dev2 = sum((Cp[i][j] - CPRIME_REF[i][j]) ** 2 for i in range(4) for j in range(4))
    eps = (Iv.frac(dev2).sqrt() / Iv.frac(LAMBDA0)).hi
    return Cp, eps
