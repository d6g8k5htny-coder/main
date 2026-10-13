

# ----------------------------------------------------------------------------- floating control (not part of the certificate)
def mc_event(r, b, k, n, seed, deg=7, nang=24):
    """End-to-end Monte Carlo of Q^W(G_r^c) / r^3 and Z_r / r^2 for the reference kernel in d = 3 (floating point): the
    Taylor jet of order deg at the midpoint is sampled under Q (exact pin series evaluated in floating point), W_r is
    evaluated from the Hessians at M and S, and M_3, M_4 are the partial-block operator norms ([CAP] (1)) maximized over a
    grid of the cylinder D (binary forms maximized over nang angles; floating, so slightly below the true suprema)."""
    import random
    pins, _ = functionals()
    P = [pins[nm] for nm in PIN_NAMES]
    jets = [(a, b1, m - a - b1) for m in range(deg + 1) for a in range(m, -1, -1) for b1 in range(m - a, -1, -1)]
    J = [[term(1, 0, a, b1, b2, X0PT)] for (a, b1, b2) in jets]
    nj = len(J)

    def ev(F, G, key):
        ser, w, _ = pair_series(F, G, key)
        return sum(float(c) * r ** p for p, c in ser.items())
    Spp = [[ev(P[i], P[j], ('p', i, j)) for j in range(NP)] for i in range(NP)]
    Sjp = [[ev(J[i], P[j], ('jp3', deg, i, j)) for j in range(NP)] for i in range(nj)]
    Sjj = [[ev(J[i], J[j], ('jj3', deg, i, j)) for j in range(nj)] for i in range(nj)]

    def solve(A, bv):
        m = len(A)
        M = [A[i][:] + [bv[i]] for i in range(m)]
        for c in range(m):
            p = max(range(c, m), key=lambda i: abs(M[i][c]))
            M[c], M[p] = M[p], M[c]
            for i in range(c + 1, m):
                f = M[i][c] / M[c][c]
                for j in range(c, m + 1):
                    M[i][j] -= f * M[c][j]
        x = [0.0] * m
        for i in reversed(range(m)):
            x[i] = (M[i][m] - sum(M[i][j] * x[j] for j in range(i + 1, m))) / M[i][i]
        return x
    vr = [b - k * r ** 3 / 2, -k * r * r, 12 * k] + [0.0] * (NP - 3)
    Wt = [solve(Spp, Sjp[i]) for i in range(nj)]
    mu = [sum(Wt[i][j] * vr[j] for j in range(NP)) for i in range(nj)]
    C = [[Sjj[i][j] - sum(Wt[i][l] * Sjp[j][l] for l in range(NP)) for j in range(nj)] for i in range(nj)]
    # Cholesky with clamped pivots (the conditional covariance is positive semidefinite up to rounding)
    L = [[0.0] * nj for _ in range(nj)]
    for i in range(nj):
        for j in range(i + 1):
            s = C[i][j] - sum(L[i][l] * L[j][l] for l in range(j))
            if i == j:
                L[i][i] = math.sqrt(s) if s > 1e-300 else 0.0
            else:
                L[i][j] = s / L[j][j] if L[j][j] > 0 else 0.0
    index = {al: i for i, al in enumerate(jets)}
    fact = [math.factorial(i) for i in range(deg + 2)]

    def deriv(cf, be, p):
        """d^be f(p) from the jet (Taylor at 0)."""
        tot = 0.0
        for al, i in index.items():
            if al[0] >= be[0] and al[1] >= be[1] and al[2] >= be[2]:
                e = (al[0] - be[0], al[1] - be[1], al[2] - be[2])
                tot += cf[i] * (p[0] ** e[0] * p[1] ** e[1] * p[2] ** e[2]) / (fact[e[0]] * fact[e[1]] * fact[e[2]])
        return tot
    angs = [(math.cos(math.pi * t / nang), math.sin(math.pi * t / nang)) for t in range(nang)]
    pts = [(x, 0.0, 0.0) for x in (-2 * r, -r, 0.0, r, 2 * r)]
    pts += [(x, 2 * r * math.cos(2 * math.pi * t / 6), 2 * r * math.sin(2 * math.pi * t / 6)) for x in (-2 * r, 0.0, 2 * r)
            for t in range(6)]

    def block_norm(vals, c):
        """Operator norm of the symmetric c-form with components vals[(b1, b2)] (b1 + b2 = c) over unit vectors."""
        if c == 0:
            return abs(vals[(0, 0)])
        best = 0.0
        for (u1, u2) in angs:
            q = sum(math.comb(c, b1) * vals[(b1, c - b1)] * u1 ** b1 * u2 ** (c - b1) for b1 in range(c + 1))
            best = max(best, abs(q))
        return best

    def Mj(cf, j):
        best = 0.0
        for p in pts:
            for a in range(j + 1):
                c = j - a
                vals = {(b1, c - b1): deriv(cf, (a, b1, c - b1), p) for b1 in range(c + 1)}
                best = max(best, block_norm(vals, c))
        return best

    def hess(cf, p):
        return [[deriv(cf, tuple(int(x == i) + int(x == j) for x in range(3)), p) for j in range(3)] for i in range(3)]

    def det3f(H):
        return (H[0][0] * (H[1][1] * H[2][2] - H[1][2] * H[2][1]) - H[0][1] * (H[1][0] * H[2][2] - H[1][2] * H[2][0])
                + H[0][2] * (H[1][0] * H[2][1] - H[1][1] * H[2][0]))

    def neg_def(H):
        return H[0][0] < 0 and H[0][0] * H[1][1] - H[0][1] ** 2 > 0 and det3f(H) < 0

    def index2(H):
        """Number of negative eigenvalues of a symmetric 3 x 3 matrix is 2: det > 0 and not positive definite."""
        d = det3f(H)
        pd = H[0][0] > 0 and H[0][0] * H[1][1] - H[0][1] ** 2 > 0 and d > 0
        return d > 0 and not pd
    rng = random.Random(seed)
    num = den = 0.0
    acap = 4 / (3 * k)
    for _ in range(n):
        z = [rng.gauss(0, 1) for _ in range(nj)]
        cf = [mu[i] + sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(nj)]
        HM, HS = hess(cf, (-r / 2, 0.0, 0.0)), hess(cf, (r / 2, 0.0, 0.0))
        if not (neg_def(HM) and index2(HS)):
            continue
        Wv = abs(det3f(HM) * det3f(HS))
        den += Wv
        a11, a22, a12 = -HM[1][1], -HM[2][2], -HM[1][2]
        lam1 = (a11 + a22) / 2 - math.sqrt(((a11 - a22) / 2) ** 2 + a12 * a12)
        M3 = Mj(cf, 3)
        if lam1 <= acap * r * M3 * M3:
            num += Wv
            continue
        if r * Mj(cf, 4) > 3 * k / 10:
            num += Wv
    return {'r': r, 'b': b, 'k': k, 'n': n, 'seed': seed, 'QW_Gc_over_r3': num / den / r ** 3 if den else None,
            'Z_over_r2': den / n / r ** 2}


if __name__ == '__main__':
    main()
