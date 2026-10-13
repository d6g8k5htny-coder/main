"""Planar near cluster coefficients alpha_1, alpha_2 of Theorem N / #158 Theorem F: numerical evaluation (stdlib only).
Field: d=2, torus side L, K_L(z)=sum_n a_n exp(2 pi i n.z/L), a_n ∝ exp(-2 pi^2 |n|^2/L^2) (parent section 2).
alpha_j = (p_b(0)/z_0) * int q_k(a,beta,c) I_j(a,beta,c) da dbeta dc,   I_j = int_{s<-|B|/2} 9k^2(4s^2-B^2) 1{n(s,a,beta,c)=j} ds.
"""
import math, random, sys, json

def lattice_moments(L, pmax=6, N=None):
    """M[p][q] = sum_n a_n n1^p n2^q (normalized weights), p,q even."""
    if N is None:
        N = int(math.ceil(L * 2.6)) + 3   # exp(-2 pi^2 N^2/L^2) ~ exp(-133): negligible
    c = 2 * math.pi ** 2 / L ** 2
    Z = 0.0
    M = [[0.0] * (pmax + 1) for _ in range(pmax + 1)]
    for n1 in range(-N, N + 1):
        for n2 in range(-N, N + 1):
            w = math.exp(-c * (n1 * n1 + n2 * n2))
            Z += w
            for p in range(0, pmax + 1, 2):
                for q in range(0, pmax + 1, 2):
                    M[p][q] += w * n1 ** p * n2 ** q
    for p in range(pmax + 1):
        for q in range(pmax + 1):
            M[p][q] /= Z
    return M

# jets: index -> (alpha_x, alpha_z)
JETS = {'f': (0, 0), 'fx': (1, 0), 'fz': (0, 1), 'fxx': (2, 0), 'fxz': (1, 1), 'fzz': (0, 2),
        'fxxx': (3, 0), 'fxxz': (2, 1), 'fxzz': (1, 2), 'fzzz': (0, 3)}

def jet_cov(L, M):
    """Cov(d^alpha f(0), d^beta f(0)) = Re[(2 pi i/L)^|alpha| (-2 pi i/L)^|beta|] sum a_n n^(alpha+beta)."""
    names = list(JETS)
    C = {}
    for u in names:
        for v in names:
            au, av = JETS[u], JETS[v]
            p, q = au[0] + av[0], au[1] + av[1]
            if p % 2 or q % 2:
                C[(u, v)] = 0.0
                continue
            coef = (2j * math.pi / L) ** (au[0] + au[1]) * (-2j * math.pi / L) ** (av[0] + av[1])
            val = coef * M[p][q]
            assert abs(val.imag) < 1e-12 * (1 + abs(val.real))
            C[(u, v)] = val.real
    return C

def mat(C, rows, cols):
    return [[C[(r, c)] for c in cols] for r in rows]

def solve(A, B):
    """Gaussian elimination: A X = B (A n x n, B n x m)."""
    n = len(A); m = len(B[0])
    Ab = [A[i][:] + B[i][:] for i in range(n)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(Ab[r][i]))
        Ab[i], Ab[piv] = Ab[piv], Ab[i]
        p = Ab[i][i]
        for j in range(i, n + m):
            Ab[i][j] /= p
        for r in range(n):
            if r != i and Ab[r][i] != 0.0:
                f = Ab[r][i]
                for j in range(i, n + m):
                    Ab[r][j] -= f * Ab[i][j]
    return [row[n:] for row in Ab]

def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

def transpose(A):
    return [list(r) for r in zip(*A)]

def regression(C, targets, target_vals, outputs):
    """Conditional mean and covariance of outputs given targets = target_vals (Gaussian)."""
    S_tt = mat(C, targets, targets); S_ot = mat(C, outputs, targets); S_oo = mat(C, outputs, outputs)
    K = transpose(solve(S_tt, transpose(S_ot)))        # S_ot S_tt^-1
    mean = [sum(K[i][j] * target_vals[j] for j in range(len(targets))) for i in range(len(outputs))]
    KS = matmul(K, transpose(S_ot))
    cov = [[S_oo[i][j] - KS[i][j] for j in range(len(outputs))] for i in range(len(outputs))]
    return mean, cov

def cholesky(A):
    n = len(A); Lm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(Lm[i][k] * Lm[j][k] for k in range(j))
            Lm[i][j] = math.sqrt(s) if i == j else s / Lm[j][j]
    return Lm

def phi(x): return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
def Phi(x): return 0.5 * math.erfc(-x / math.sqrt(2))

# ---------------------------------------------------------------- classifier in s
def cubic_roots(B, k, D):
    """real roots of g(s) = -2 s^3 + 3 B s^2 - B^3 - 12 k D^2 (sorted)."""
    g = lambda s: -2 * s ** 3 + 3 * B * s ** 2 - B ** 3 - 12 * k * D * D
    crit = sorted({0.0, B})
    # bracket: g -> +inf at -inf, -inf at +inf
    span = 10 * (1 + abs(B) + abs(D) + k) + 10 * (abs(12 * k * D * D) ** (1 / 3))
    pts = [-span] + crit + [span]
    while g(pts[0]) <= 0: pts[0] *= 2
    while g(pts[-1]) >= 0: pts[-1] *= 2
    roots = []
    for lo, hi in zip(pts[:-1], pts[1:]):
        glo, ghi = g(lo), g(hi)
        if glo == 0: roots.append(lo); continue
        if glo * ghi < 0:
            for _ in range(200):
                mid = 0.5 * (lo + hi); gm = g(mid)
                if gm == 0: lo = hi = mid; break
                if glo * gm < 0: hi, ghi = mid, gm
                else: lo, glo = mid, gm
            roots.append(0.5 * (lo + hi))
    return sorted(set(roots))

def n_of(s, a, b, c, k):
    """count of extra window critical points via classifier: typed iff s < -|B|/2."""
    B = b - a * a / (12 * k); D = (c - a * b / (4 * k) + a ** 3 / (72 * k * k)) / 2
    if not s < -abs(B) / 2: return 0
    g = -2 * s ** 3 + 3 * B * s ** 2 - B ** 3 - 12 * k * D * D
    if s > B:
        return 2 if g > 0 else 1
    return 1 if g < 0 else 0

def n_direct(s, a, b, c, k):
    """direct count: critical points of G(u,Z) = C(u) + (s+Bu) Z^2/2 + (D/3) Z^3 with Z != 0 and height in (-k,0)."""
    B = b - a * a / (12 * k); D = (c - a * b / (4 * k) + a ** 3 / (72 * k * k)) / 2
    Cu = lambda u: 2 * k * u ** 3 - 1.5 * k * u - k / 2
    pts = []
    if abs(D) > 1e-14:
        # (6k + B^3/(2D^2)) u^2 + (B^2 s/D^2) u + (B s^2/(2 D^2) - 3k/2) = 0
        A2 = 6 * k + B ** 3 / (2 * D * D); A1 = B * B * s / (D * D); A0 = B * s * s / (2 * D * D) - 1.5 * k
        if abs(A2) < 1e-14:
            us = [-A0 / A1] if abs(A1) > 1e-14 else []
        else:
            disc = A1 * A1 - 4 * A2 * A0
            us = [] if disc < 0 else [(-A1 + sg * math.sqrt(disc)) / (2 * A2) for sg in (1, -1)]
        for u in us:
            w = s + B * u; Z = -w / D
            if abs(Z) < 1e-12: continue
            h = Cu(u) + w * Z * Z / 2 + D * Z ** 3 / 3
            pts.append(h)
    else:
        if abs(B) > 1e-14:
            u = -s / B; Z2 = 3 * k * (1 - 4 * u * u) / B
            if Z2 > 1e-14:
                h = Cu(u)
                pts.append(h); pts.append(h)
    return sum(1 for h in pts if -k < h < 0)

def I_j(a, b, c, k):
    """(I_1, I_2): int 9k^2(4s^2 - B^2) 1{n=j} ds over s < -|B|/2, via exact interval structure."""
    B = b - a * a / (12 * k); D = (c - a * b / (4 * k) + a ** 3 / (72 * k * k)) / 2
    top = -abs(B) / 2
    F = lambda s: 12 * k * k * s ** 3 - 9 * k * k * B * B * s      # antiderivative of 9k^2(4s^2-B^2)
    roots = cubic_roots(B, k, D)
    g = lambda s: -2 * s ** 3 + 3 * B * s ** 2 - B ** 3 - 12 * k * D * D
    # sign structure by evaluation at interior points (robust to double roots / tangencies)
    bps = [-math.inf] + roots + [math.inf]
    pos, neg = [], []
    for lo, hi in zip(bps[:-1], bps[1:]):
        if lo == -math.inf and hi == math.inf: mid = 0.0
        elif lo == -math.inf: mid = hi - 1.0 - abs(hi)
        elif hi == math.inf: mid = lo + 1.0 + abs(lo)
        else: mid = 0.5 * (lo + hi)
        (pos if g(mid) > 0 else neg).append((lo, hi))
    def clip_int(lo, hi, a_, b_):
        lo2, hi2 = max(lo, a_), min(hi, b_)
        return (F(hi2) - F(lo2)) if hi2 > lo2 else 0.0
    # n=1 : g<0 on s<top (both s<=B and s>B branches give 1 when g<0)
    I1 = sum(clip_int(lo, hi, -1e300, top) for lo, hi in neg)
    # n=2 : g>0 and B < s < top
    I2 = sum(clip_int(lo, hi, B, top) for lo, hi in pos) if B < top else 0.0
    return I1, I2

def setup(L, k, b):
    M = lattice_moments(L)
    C = jet_cov(L, M)
    # even block: A = fzz given (f, fxx, fxz) = (b, 0, 0)
    muA, covA = regression(C, ['f', 'fxx', 'fxz'], [b, 0.0, 0.0], ['fzz'])
    muA = muA[0]; sA = math.sqrt(covA[0][0])
    pb0 = phi((0 - muA) / sA) / sA
    t = -muA / sA
    m2b = muA * muA * Phi(t) - 2 * muA * sA * phi(t) + sA * sA * (Phi(t) - t * phi(t))
    z0 = 36 * k * k * m2b
    # odd block: (fxxz, fxzz, fzzz) given (fx, fz, fxxx) = (0, 0, 12k)
    muJ, covJ = regression(C, ['fx', 'fz', 'fxxx'], [0.0, 0.0, 12 * k], ['fxxz', 'fxzz', 'fzzz'])
    return dict(C=C, muA=muA, sA=sA, pb0=pb0, m2b=m2b, z0=z0, muJ=muJ, covJ=covJ, Lch=cholesky(covJ))

def gauss_hermite(n):
    """nodes/weights for weight exp(-x^2/2)/sqrt(2 pi) via Golub-Welsch-free Newton on Hermite polynomials."""
    nodes, weights = [], []
    # initial guesses (Szego) then Newton
    for i in range(1, n + 1):
        if i == 1: x = math.sqrt(2 * n + 1) - 1.85575 * (2 * n + 1) ** (-1 / 6)
        elif i == 2: x = nodes[-1] - 1.14 * n ** 0.426 / nodes[-1]
        elif i == 3: x = 1.86 * nodes[-1] - 0.86 * nodes[0]
        elif i == 4: x = 1.91 * nodes[-1] - 0.91 * nodes[-2]
        else: x = 2 * nodes[-1] - nodes[-2]
        for _ in range(100):
            # physicists' Hermite via recurrence, normalized (probabilists' scaling handled after)
            p0, p1 = 1.0 / math.pi ** 0.25, 0.0
            p1 = math.sqrt(2) * x * p0
            pm, pc = p0, p1
            for j in range(2, n + 1):
                pn = math.sqrt(2 / j) * x * pc - math.sqrt((j - 1) / j) * pm
                pm, pc = pc, pn
            dp = math.sqrt(2 * n) * pm
            dx = pc / dp; x -= dx
            if abs(dx) < 1e-15: break
        nodes.append(x); weights.append(2.0 / (dp * dp))
    # symmetric completion: nodes are the positive half (approximately); build full set robustly by symmetry
    full = sorted(set([round(v, 14) for v in nodes] + [round(-v, 14) for v in nodes]))
    # recompute weights for full set exactly via Newton refinement at each node
    out = []
    for x in full:
        p0 = 1.0 / math.pi ** 0.25; pm, pc = p0, math.sqrt(2) * x * p0
        for j in range(2, n + 1):
            pn = math.sqrt(2 / j) * x * pc - math.sqrt((j - 1) / j) * pm
            pm, pc = pc, pn
        dp = math.sqrt(2 * n) * pm
        w = 2.0 / (dp * dp)
        out.append((x * math.sqrt(2), w / math.sqrt(math.pi)))   # probabilists' scaling: sum w = 1, nodes sqrt(2) x
    return out

def alphas_gh(S, k, order):
    gh = gauss_hermite(order)
    Lch, mu = S['Lch'], S['muJ']
    I1 = I2 = 0.0
    for x1, w1 in gh:
        for x2, w2 in gh:
            for x3, w3 in gh:
                y = (x1, x2, x3)
                a = mu[0] + Lch[0][0] * y[0]
                bb = mu[1] + Lch[1][0] * y[0] + Lch[1][1] * y[1]
                c = mu[2] + Lch[2][0] * y[0] + Lch[2][1] * y[1] + Lch[2][2] * y[2]
                i1, i2 = I_j(a, bb, c, k)
                w = w1 * w2 * w3
                I1 += w * i1; I2 += w * i2
    f = S['pb0'] / S['z0']
    return f * I1, f * I2

def alphas_mc(S, k, nsamp, seed=1):
    rng = random.Random(seed)
    Lch, mu = S['Lch'], S['muJ']
    s1 = s2 = q1 = q2 = 0.0
    for _ in range(nsamp):
        y = (rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))
        a = mu[0] + Lch[0][0] * y[0]
        bb = mu[1] + Lch[1][0] * y[0] + Lch[1][1] * y[1]
        c = mu[2] + Lch[2][0] * y[0] + Lch[2][1] * y[1] + Lch[2][2] * y[2]
        i1, i2 = I_j(a, bb, c, k)
        s1 += i1; s2 += i2; q1 += i1 * i1; q2 += i2 * i2
    f = S['pb0'] / S['z0']
    m1, m2 = s1 / nsamp, s2 / nsamp
    e1 = math.sqrt(max(q1 / nsamp - m1 * m1, 0) / nsamp); e2 = math.sqrt(max(q2 / nsamp - m2 * m2, 0) / nsamp)
    return f * m1, f * m2, f * e1, f * e2

if __name__ == '__main__':
    # 1. classifier vs direct root count
    rng = random.Random(3); bad = 0; tot = 0
    for _ in range(20000):
        k = rng.choice([0.5, 1.0, 2.0]); a, b, c, s = (rng.uniform(-4, 4) for _ in range(4))
        n1, n2 = n_of(s, a, b, c, k), n_direct(s, a, b, c, k)
        tot += 1
        if n1 != n2: bad += 1
    print('classifier vs direct: mismatches', bad, 'of', tot)
    # 2. I_j vs brute-force s-quadrature at random jets
    for trial in range(3):
        k = 1.0; a, b, c = (rng.uniform(-3, 3) for _ in range(3))
        B = b - a * a / (12 * k)
        top = -abs(B) / 2; lo = top - 60
        N = 400000; h = (top - lo) / N; b1 = b2 = 0.0
        for i in range(N):
            s = lo + (i + 0.5) * h; n = n_of(s, a, b, c, k); w = 9 * k * k * (4 * s * s - B * B)
            if n == 1: b1 += w * h
            elif n == 2: b2 += w * h
        print('I_j exact vs quadrature', I_j(a, b, c, k), (b1, b2))
    # 3. coefficients
    for L in (24.0, 12.0, 6.0):
        for k in (1.0,):
            for b in (0.0, 1.0):
                S = setup(L, k, b)
                print(f'\nL={L} k={k} b={b}: muA={S["muA"]:.6g} sA={S["sA"]:.6g} pb0={S["pb0"]:.6g} z0={S["z0"]:.6g} muJ={[round(x,6) for x in S["muJ"]]}')
                print('  covJ diag', [round(S['covJ'][i][i], 6) for i in range(3)], 'cov(a,c)', round(S['covJ'][0][2], 6))
                for order in (20, 40):
                    a1, a2 = alphas_gh(S, k, order)
                    print(f'  GH{order}: alpha1={a1:.6g} alpha2={a2:.6g} ratio2={a2/(a1+a2):.5f} palm_near={2*a2/(a1+2*a2):.5f}')
                a1, a2, e1, e2 = alphas_mc(S, k, 200000)
                print(f'  MC2e5: alpha1={a1:.6g}±{e1:.2g} alpha2={a2:.6g}±{e2:.2g}')
