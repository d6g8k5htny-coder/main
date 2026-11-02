"""Independent numerical check of the finite-L binding (Math- frontiers/original_finite_l_gaussian_binding_20261017).

Route A builds raw jet covariances from direct d-dimensional lattice sums and applies Gaussian
conditioning numerically, then evaluates GAD equation (28) as quoted in the binding's (17).
Route B evaluates formula (1) from the binding's closed forms (10), (11), (13), (14).
Standard library only. Usage: python3 -I -B check_binding.py
"""
import itertools
import math

def lattice(L, K, d):
    q1 = [2 * math.pi * k / L for k in range(-K, K + 1)]
    w1 = [math.exp(-x * x / 2) for x in q1]
    z = sum(w1)
    w1 = [x / z for x in w1]
    pts = []
    for idx in itertools.product(range(len(q1)), repeat=d):
        w = 1.0
        for i in idx:
            w *= w1[i]
        pts.append((tuple(q1[i] for i in idx), w))
    return pts, q1, w1

def moment(pts, gamma):
    s = 0.0
    for p, w in pts:
        t = w
        for pi, g in zip(p, gamma):
            if g:
                t *= pi ** g
        s += t
    return s

def jet_cov(pts, alpha, beta):
    """Cov(d^alpha F(x), d^beta F(x)) = (-1)^|beta| Re(i^(|alpha|+|beta|)) sum_m w_m p_m^(alpha+beta)."""
    n = sum(alpha) + sum(beta)
    re = [1, 0, -1, 0][n % 4]
    if re == 0:
        return 0.0
    gamma = tuple(a + b for a, b in zip(alpha, beta))
    return (-1) ** sum(beta) * re * moment(pts, gamma)

def mi(d, *axes):
    counts = [0] * d
    for a in axes:
        counts[a] += 1
    return tuple(counts)

def combo_cov(pts, x, y, cache):
    s = 0.0
    for a, ca in x.items():
        for b, cb in y.items():
            key = (a, b)
            if key not in cache:
                cache[key] = jet_cov(pts, a, b)
            s += ca * cb * cache[key]
    return s

def add(dct, k, v):
    dct[k] = dct.get(k, 0.0) + v

def variables(d, e, U):
    D = [{mi(d, i): 1.0} for i in range(d)]
    He = []
    for i in range(d):
        v = {}
        for j in range(d):
            add(v, mi(d, i, j), e[j])
        He.append(v)
    T = {}
    for i, j, k in itertools.product(range(d), repeat=3):
        add(T, mi(d, i, j, k), e[i] * e[j] * e[k])
    A = []
    pairs = [(a, b) for a in range(d - 1) for b in range(a, d - 1)]
    for a, b in pairs:
        v = {}
        for i, j in itertools.product(range(d), repeat=2):
            add(v, mi(d, i, j), U[i][a] * U[j][b])
        A.append(v)
    return D, He, T, A, pairs

def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

def transpose(A):
    return [list(r) for r in zip(*A)]

def inverse(M):
    n = len(M)
    a = [list(r) + [float(i == j) for j in range(n)] for i, r in enumerate(M)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        a[c], a[p] = a[p], a[c]
        pv = a[c][c]
        a[c] = [x / pv for x in a[c]]
        for r in range(n):
            if r != c and a[r][c]:
                f = a[r][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [r[n:] for r in a]

def det(M):
    n = len(M)
    a = [list(r) for r in M]
    s = 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        if p != c:
            a[c], a[p] = a[p], a[c]
            s = -s
        s *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return s

def route_a(pts, d, e, U, cache):
    """Raw-sum joint law: p_(D,He)(0,0) and conditional covariance of (T, A) given D=0, He=0."""
    D, He, T, A, pairs = variables(d, e, U)
    X = D + He
    Y = [T] + A
    Sxx = [[combo_cov(pts, u, v, cache) for v in X] for u in X]
    Sxy = [[combo_cov(pts, u, v, cache) for v in Y] for u in X]
    Syy = [[combo_cov(pts, u, v, cache) for v in Y] for u in Y]
    S = [[Syy[i][j] - v for j, v in enumerate(r)] for i, r in enumerate(matmul(matmul(transpose(Sxy), inverse(Sxx)), Sxy))]
    p0 = (2 * math.pi) ** (-len(X) / 2) / math.sqrt(det(Sxx))
    return p0, S, pairs

def route_b_parts(a, b, c, d, e, U):
    """Closed forms of the binding: B (11), V (10), Gamma_e (12)-(13), p_(D,He)(0,0) (18)."""
    k4, k6 = b - 3 * a * a, c - 15 * a * b + 30 * a ** 3
    s4, s6 = sum(x ** 4 for x in e), sum(x ** 6 for x in e)
    V = 6 * a ** 3 + 9 * a * k4 * s4 + (k6 - k4 * k4 / a) * s6
    B = [[a * a * ((i == j) + 2 * e[i] * e[j]) + (k4 * e[i] ** 2 if i == j else 0.0) for j in range(d)] for i in range(d)]
    pairs = [(x, y) for x in range(d - 1) for y in range(x, d - 1)]
    R = [[a * a * (x == y) * e[i] + k4 * U[i][x] * U[i][y] * e[i] for i in range(d)] for x, y in pairs]
    C = [[a * a * ((x == y) * (z == w) + (x == z) * (y == w) + (x == w) * (y == z))
          + k4 * sum(U[i][x] * U[i][y] * U[i][z] * U[i][w] for i in range(d)) for z, w in pairs] for x, y in pairs]
    G = [[C[i][j] - v for j, v in enumerate(r)] for i, r in enumerate(matmul(matmul(R, inverse(B)), transpose(R)))]
    p0 = (2 * math.pi) ** (-d) * a ** (-d / 2) / math.sqrt(det(B))
    return V, B, G, p0

def abs_moment_43(V):
    return 2 ** (2 / 3) * math.gamma(7 / 6) * V ** (2 / 3) / math.sqrt(math.pi)

def abs_moment_43_quadrature(V, n=200000, cut=40.0):
    """Midpoint rule for E|N(0,V)|^(4/3), independent of the Gamma closed form (19)."""
    s, h = 0.0, cut / n
    for i in range(n):
        t = (i + 0.5) * h
        s += t ** (4 / 3) * math.exp(-t * t / 2)
    return 2 * s * h / math.sqrt(2 * math.pi) * V ** (2 / 3)

def check_d2(L, K, nodes=720):
    pts, q1, w1 = lattice(L, K, 2)
    a = sum(w * x ** 2 for x, w in zip(q1, w1))
    b = sum(w * x ** 4 for x, w in zip(q1, w1))
    c = sum(w * x ** 6 for x, w in zip(q1, w1))
    gad = alg = 0.0
    cache = {}
    worst = {'p0': 0.0, 'V': 0.0, 'G': 0.0, 'rho': 0.0}
    for n in range(nodes):
        th = 2 * math.pi * n / nodes
        e = (math.cos(th), math.sin(th))
        U = [[-math.sin(th)], [math.cos(th)]]
        p0a, S, _ = route_a(pts, 2, e, U, cache)
        V, B, G, p0b = route_b_parts(a, b, c, 2, e, U)
        rho = S[0][1] / math.sqrt(S[0][0] * S[1][1])
        worst['rho'] = max(worst['rho'], abs(rho))
        worst['p0'] = max(worst['p0'], abs(p0a / p0b - 1))
        worst['V'] = max(worst['V'], abs(S[0][0] / V - 1))
        worst['G'] = max(worst['G'], abs(S[1][1] / G[0][0] - 1))
        # Route A: GAD (28) integrand with the raw conditional law; T,A independence checked via rho.
        gad += p0a * abs_moment_43(S[0][0]) * S[1][1] / 2
        # Route B: formula (1) integrand, F(e) = Gamma_e / 2 in d = 2.
        alg += (G[0][0] / 2) * V ** (2 / 3) / math.sqrt(det(B))
    h = 2 * math.pi / nodes
    C_gad = gad * h / (2 * 12 ** (1 / 3))
    C_alg = math.gamma(7 / 6) / (24 ** (1 / 3) * math.sqrt(math.pi) * (2 * math.pi) ** 2 * a) * alg * h
    return {'L': L, 'K': K, 'a': a, 'b': b, 'c': c, 'C_GAD_route_A': C_gad, 'C_alg_route_B': C_alg,
            'rel_diff': C_gad / C_alg - 1, 'worst_rel': worst}

def check_d3(L, K, directions=6):
    pts, q1, w1 = lattice(L, K, 3)
    a = sum(w * x ** 2 for x, w in zip(q1, w1))
    b = sum(w * x ** 4 for x, w in zip(q1, w1))
    c = sum(w * x ** 6 for x, w in zip(q1, w1))
    out = []
    cache = {}
    for n in range(directions):
        th, ph = 0.37 + 1.1 * n, 0.9 + 0.7 * n
        e = (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))
        u1 = (math.cos(th) * math.cos(ph), math.cos(th) * math.sin(ph), -math.sin(th))
        u2 = (-math.sin(ph), math.cos(ph), 0.0)
        U = [[u1[i], u2[i]] for i in range(3)]
        p0a, S, pairs = route_a(pts, 3, e, U, cache)
        V, B, G, p0b = route_b_parts(a, b, c, 3, e, U)
        cross = max(abs(S[0][j]) for j in range(1, 4))
        gdiff = max(abs(S[i + 1][j + 1] - G[i][j]) for i in range(3) for j in range(3))
        out.append({'e': [round(x, 6) for x in e], 'p0_rel': p0a / p0b - 1, 'V_rel': S[0][0] / V - 1,
                    'max_|Cov(T,A)|cond': cross, 'max_|Gamma_raw-Gamma_formula|': gdiff,
                    'Gamma_scale': max(abs(x) for r in G for x in r)})
    return {'L': L, 'K': K, 'kappa4': b - 3 * a * a, 'directions': out}

if __name__ == '__main__':
    import json
    print('E|N(0,1)|^(4/3): closed form', abs_moment_43(1.0), 'quadrature', abs_moment_43_quadrature(1.0))
    for L, K in ((3.0, 8), (6.0, 14), (24.0, 60)):
        print(json.dumps(check_d2(L, K)))
    for L, K in ((3.0, 6), (5.0, 8)):
        print(json.dumps(check_d3(L, K)))
