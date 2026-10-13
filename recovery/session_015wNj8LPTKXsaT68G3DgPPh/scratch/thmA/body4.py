

# ----------------------------------------------------------------------------- floating control (not part of the certificate)
def mc_event(r, b, k, n, seed, deg=8, grid=9):
    """End-to-end Monte Carlo of Q^W(G_r^c) / r^3 and Z_r / r^2 for the reference kernel (floating point): the Taylor jet of
    order deg at the midpoint is sampled under Q, the field near D is rebuilt from it, W_r and G_r (M_3, M_4 on a grid of D)
    are evaluated directly from the definitions."""
    import random
    pins, _ = functionals()
    P = [pins[nm] for nm in PIN_NAMES]
    jets = [(i, m - i) for m in range(deg + 1) for i in range(m, -1, -1)]
    J = [[term(1, 0, a, bb, X0PT)] for (a, bb) in jets]

    def ev(F, G, key):
        ser, w, _ = pair_series(F, G, key)
        return sum(float(c) * r ** p for p, c in ser.items())
    Spp = [[ev(P[i], P[j], ('p', i, j)) for j in range(6)] for i in range(6)]
    Sjp = [[ev(J[i], P[j], ('jp', deg, i, j)) for j in range(6)] for i in range(len(J))]
    Sjj = [[ev(J[i], J[j], ('jj', deg, i, j)) for j in range(len(J))] for i in range(len(J))]

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
    vr = [b - k * r ** 3 / 2, -k * r * r, 12 * k, 0, 0, 0]
    Wt = [solve(Spp, Sjp[i]) for i in range(len(J))]
    mu = [sum(Wt[i][j] * vr[j] for j in range(6)) for i in range(len(J))]
    C = [[Sjj[i][j] - sum(Wt[i][l] * Sjp[j][l] for l in range(6)) for j in range(len(J))] for i in range(len(J))]
    # symmetric square root (Jacobi)
    m = len(J)
    A = [row[:] for row in C]
    V = [[float(i == j) for j in range(m)] for i in range(m)]
    for _ in range(100):
        off = sum(A[p][q] ** 2 for p in range(m) for q in range(p + 1, m))
        if off < 1e-26:
            break
        for p in range(m - 1):
            for q in range(p + 1, m):
                if abs(A[p][q]) < 1e-300:
                    continue
                th = 0.5 * math.atan2(2 * A[p][q], A[q][q] - A[p][p])
                c, s = math.cos(th), math.sin(th)
                for kk in range(m):
                    akp, akq = A[kk][p], A[kk][q]
                    A[kk][p], A[kk][q] = c * akp - s * akq, s * akp + c * akq
                for kk in range(m):
                    apk, aqk = A[p][kk], A[q][kk]
                    A[p][kk], A[q][kk] = c * apk - s * aqk, s * apk + c * aqk
                for kk in range(m):
                    vkp, vkq = V[kk][p], V[kk][q]
                    V[kk][p], V[kk][q] = c * vkp - s * vkq, s * vkp + c * vkq
    lam = [max(A[i][i], 0.0) for i in range(m)]
    S = [[sum(V[i][kk] * math.sqrt(lam[kk]) * V[j][kk] for kk in range(m)) for j in range(m)] for i in range(m)]
    fact = [math.factorial(i) for i in range(deg + 2)]
    pts = [(-2 * r + 4 * r * i / (grid - 1), -2 * r + 4 * r * j / (grid - 1)) for i in range(grid) for j in range(grid)]

    def deriv(cf, a0, b0, x, y):
        tot = 0.0
        for idx, (a, bb) in enumerate(jets):
            if a >= a0 and bb >= b0:
                tot += cf[idx] * x ** (a - a0) * y ** (bb - b0) / (fact[a - a0] * fact[bb - b0])
        return tot
    rng = random.Random(seed)
    num = den = 0.0
    acap = 4 / (3 * k)
    for _ in range(n):
        z = [rng.gauss(0, 1) for _ in jets]
        cf = [mu[i] + sum(S[i][j] * z[j] for j in range(m)) for i in range(m)]
        hM = [deriv(cf, 2, 0, -r / 2, 0.0), deriv(cf, 1, 1, -r / 2, 0.0), deriv(cf, 0, 2, -r / 2, 0.0)]
        hS = [deriv(cf, 2, 0, r / 2, 0.0), deriv(cf, 1, 1, r / 2, 0.0), deriv(cf, 0, 2, r / 2, 0.0)]
        dM = hM[0] * hM[2] - hM[1] ** 2
        dS = hS[0] * hS[2] - hS[1] ** 2
        if not (hM[0] < 0 and dM > 0 and dS < 0):
            continue
        Wv = abs(dM * dS)
        den += Wv
        lamv = -hM[2]
        M3 = max(abs(deriv(cf, a, 3 - a, x, y)) for (x, y) in pts for a in range(4))
        M4 = max(abs(deriv(cf, a, 4 - a, x, y)) for (x, y) in pts for a in range(5))
        if not (lamv > acap * r * M3 * M3 and r * M4 <= 3 * k / 10):
            num += Wv
    return {'r': r, 'b': b, 'k': k, 'n': n, 'seed': seed, 'QW_Gc_over_r3': num / den / r ** 3, 'Z_over_r2': den / n / r ** 2}
