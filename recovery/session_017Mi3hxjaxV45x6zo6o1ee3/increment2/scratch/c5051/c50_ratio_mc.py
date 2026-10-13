#!/usr/bin/env python3
"""Float (NOT certified) estimate of C_U(1)/C_R for C50, reference planar kernel.

Model (Math-#217 NOTE §1): (a, beta, c) ~ N(0, diag(2, 2, 6)); B = beta - a^2/(12k),
D = (c - a beta/(4k) + a^3/(72k^2))/2; failure in s is the interval (s1, -|B|/2) where s1 is the root of
g(s) = -2s^3 + 3Bs^2 - B^3 - 12kD^2 on (-inf, min(B, 0)]; weight w = 9k^2(4s^2 - B^2).
k-weight on K: exp(-12k^2) (pi_0); b and sigma factor out of the ratio.

C_R   ~ ((C0^5-c0^5)/5) int_K e^{-12k^2} J_fail(k) dk
C_U(1)= C_R - (1/2) int_K e^{-12k^2} int t^4 E[ int w 1_fail 1{D_(t,k)=2} ds ] dt dk
1{D=2} only on Delta<0 (lemma): Y' (the extra saddle that is not Y_*) admissible:
   c0 <= t|Y'-M0| <= C0 and -P(Y')/|Y'-M0|^3 in K, original coordinates, M0=(-1/2,0).
"""
import math, random, sys
from multiprocessing import Pool

def gauss_legendre(n):
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for j in range(2, n + 1):
                p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < 1e-15:
                break
        xs.append(x); ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws

def s1_root(B, D, k):
    hi = min(B, 0.0)
    g = lambda s: -2 * s ** 3 + 3 * B * s * s - B ** 3 - 12 * k * D * D
    if g(hi) >= 0:
        return hi
    lo = hi - 1.0
    while g(lo) < 0:
        lo = hi - 2 * (hi - lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def F(s, B, k):
    return 12 * k * k * s ** 3 - 9 * k * k * B * B * s

def second_saddle(a, B, D, s, k):
    """Delta<0 branch: both extras are saddles; return (dist, gap) of the one that is not Y_*, or None."""
    Dl = 12 * k * D * D + B ** 3
    A1 = 2 * B * B * s; A0 = B * s * s - 3 * k * D * D
    disc = A1 * A1 - 4 * Dl * A0
    if disc <= 0 or D == 0:
        return None
    r = math.sqrt(disc)
    pts = []
    for u in ((-A1 + r) / (2 * Dl), (-A1 - r) / (2 * Dl)):
        Z = -(s + B * u) / D
        P = -k * (u + 0.5) + (s / 6) * Z * Z
        X = u - a * Z / (12 * k)
        dist = math.hypot(X + 0.5, Z)
        pts.append((P, dist))
    win = [p for p in pts if -k < p[0] < 0]
    if not win:
        return None
    ystar = max(win, key=lambda p: p[0])
    other = pts[1] if pts[0] is ystar else pts[0]
    P, dist = other
    return dist, -P / dist ** 3

def worker(args):
    seed, nsamp, knodes, bands, m_s = args
    rng = random.Random(seed)
    nk = len(knodes)
    Jf = [0.0] * nk; Jm = [0.0] * nk
    ex = {bk: [0.0] * nk for bk in bands}
    for _ in range(nsamp):
        a = rng.gauss(0, math.sqrt(2)); beta = rng.gauss(0, math.sqrt(2)); c = rng.gauss(0, math.sqrt(6))
        for i, k in enumerate(knodes):
            B = beta - a * a / (12 * k)
            D = (c - a * beta / (4 * k) + a ** 3 / (72 * k * k)) / 2
            smax = -abs(B) / 2
            s1 = s1_root(B, D, k)
            if s1 >= smax:
                continue
            I = F(smax, B, k) - F(s1, B, k)
            Jf[i] += I
            if 12 * k * D * D + B ** 3 < 0:
                Jm[i] += I
                L = smax - s1
                for (c0, C0, km, kp) in bands:
                    if not (km <= k <= kp):
                        continue
                    acc = 0.0
                    for _ in range(m_s):
                        s = s1 + L * rng.random()
                        res = second_saddle(a, B, D, s, k)
                        if res is None:
                            continue
                        dist, gap = res
                        if not (km <= gap <= kp):
                            continue
                        # t-average of 1{c0 <= t dist <= C0} under density 5t^4/(C0^5-c0^5) on [c0,C0]
                        lo = max(c0, c0 / dist); hi = min(C0, C0 / dist)
                        if hi > lo:
                            frac = (hi ** 5 - lo ** 5) / (C0 ** 5 - c0 ** 5)
                            acc += 9 * k * k * (4 * s * s - B * B) * frac
                    ex[(c0, C0, km, kp)][i] += L * acc / m_s
    return Jf, Jm, ex

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    nsamp = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
    if mode == "validate":
        knodes = [0.5, 1.0, 2.0]; bands = []
    else:
        km, kp = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.5, 2.0)
        xs, ws = gauss_legendre(16)
        knodes = [km + (kp - km) * (x + 1) / 2 for x in xs]
        kw = [(kp - km) / 2 * w for w in ws]
        bands = [(0.5, 2.0, km, kp), (0.5, 4.0, km, kp), (0.25, 4.0, km, kp)]
    nproc = 4
    with Pool(nproc) as pool:
        outs = pool.map(worker, [(1000 + j, nsamp, knodes, bands, 4) for j in range(nproc)])
    N = nproc * nsamp
    Jf = [sum(o[0][i] for o in outs) / N for i in range(len(knodes))]
    Jm = [sum(o[1][i] for o in outs) / N for i in range(len(knodes))]
    if mode == "validate":
        for k, jf, jm in zip(knodes, Jf, Jm):
            print(f"k={k}: J_fail~{jf:.3f}  J^-~{jm:.3f}  J^-/J_fail~{jm/jf:.4f}")
    else:
        ew = [w * math.exp(-12 * k * k) for k, w in zip(knodes, kw)]
        CR = sum(e * j for e, j in zip(ew, Jf)); CRm = sum(e * j for e, j in zip(ew, Jm))
        print(f"K=[{km},{kp}]  C_R^-/C_R ~ {CRm/CR:.4f}  -> rigorous-form bound C_U(1)/C_R >= {1-CRm/CR/2:.4f}")
        for bk in bands:
            E2 = sum(e * sum(o[2][bk][i] for o in outs) / N for i, e in enumerate(ew))
            print(f"  band [c0,C0]=[{bk[0]},{bk[1]}]:  C_U(1)/C_R ~ {1 - 0.5 * E2 / CR:.5f}   (D=2 mass share {E2/CR:.5f})")

def per_worker_spread(outs, knodes, kw, bands):
    """Ratio estimates from each worker separately (crude error proxy)."""
    ew = [w * math.exp(-12 * k * k) for k, w in zip(knodes, kw)]
    rows = []
    for o in outs:
        CR = sum(e * j for e, j in zip(ew, o[0])); CRm = sum(e * j for e, j in zip(ew, o[1]))
        rows.append((CRm / CR, [sum(e * o[2][bk][i] for i, e in enumerate(ew)) / CR for bk in bands]))
    return rows
