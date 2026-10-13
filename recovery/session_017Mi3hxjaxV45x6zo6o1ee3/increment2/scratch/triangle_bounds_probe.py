"""Nonauthor falsification probe for the Math-#103 short-edge triangle lemma
and two deterministic extensions (angle-dependent and angle-free).

Standard library only. For random rational cubics f on R^2 with three
prescribed rational critical points A, B, C (exact nullspace solve), the
Hessian is affine, so its Lipschitz constant is L = max_{|w|=1} ||D^3f[w]||_op.
L is estimated from below by dense direction sampling; since every bound is
increasing in L, a pass with the lower estimate is a pass for the true L.

Checked, with Pi = |det H_A det H_B det H_C|, r=|B-A|, d=|C-A|,
sigma=|sin angle(B-A,C-A)|, ell=|B-C|:
  T1  (Math-#103)  d<=r/4, sigma>=1/2  =>  Pi <= (2025/1024) L^6 r^4 d^2
  T2  (any sigma)  Pi <= [L^2 r d/(4 sigma)] [L(r+(r+d)/(2 sigma))]^2
                         [L^2 d ell^2/(4 r sigma)]
  T3  (angle-free) Pi <= (1/8) L^3 r d^2 ||H_A|| ||H_B|| ||H_C||
"""
import math
import random
from fractions import Fraction as F

# monomials x^i y^j with 1 <= i+j <= 3
MONO = [(i, j) for s in (1, 2, 3) for i in range(s + 1) for j in [s - i]]


def grad_rows(pt):
    x, y = pt
    rx, ry = [], []
    for i, j in MONO:
        rx.append(i * x ** (i - 1) * y ** j if i else F(0))
        ry.append(j * x ** i * y ** (j - 1) if j else F(0))
    return rx, ry


def nullspace(rows, n):
    m = [list(r) for r in rows]
    piv, r = [], 0
    for c in range(n):
        p = next((k for k in range(r, len(m)) if m[k][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        inv = 1 / m[r][c]
        m[r] = [v * inv for v in m[r]]
        for k in range(len(m)):
            if k != r and m[k][c] != 0:
                fac = m[k][c]
                m[k] = [a - fac * b for a, b in zip(m[k], m[r])]
        piv.append(c)
        r += 1
    free = [c for c in range(n) if c not in piv]
    basis = []
    for fc in free:
        v = [F(0)] * n
        v[fc] = F(1)
        for i, pc in enumerate(piv):
            v[pc] = -m[i][fc]
        basis.append(v)
    return basis


def hessian(coef, pt):
    x, y = pt
    hxx = hxy = hyy = F(0)
    for a, (i, j) in zip(coef, MONO):
        if a == 0:
            continue
        if i >= 2:
            hxx += a * i * (i - 1) * x ** (i - 2) * y ** j
        if i >= 1 and j >= 1:
            hxy += a * i * j * x ** (i - 1) * y ** (j - 1)
        if j >= 2:
            hyy += a * j * (j - 1) * x ** i * y ** (j - 2)
    return hxx, hxy, hyy


def third(coef):
    d = dict(zip(MONO, coef))
    return (6 * d[(3, 0)], 2 * d[(2, 1)], 2 * d[(1, 2)], 6 * d[(0, 3)])


def opnorm(a, b, c):
    a, b, c = float(a), float(b), float(c)
    return abs((a + c) / 2) + math.hypot((a - c) / 2, b)


def lip_lower(t, n=4096):
    fxxx, fxxy, fxyy, fyyy = map(float, t)
    best = 0.0
    for k in range(n):
        th = math.pi * k / n
        w1, w2 = math.cos(th), math.sin(th)
        best = max(best, opnorm(fxxx * w1 + fxxy * w2, fxxy * w1 + fxyy * w2,
                                fxyy * w1 + fyyy * w2))
    return best


def rat(scale, den=97):
    return F(random.randint(-scale * den, scale * den), den)


def trial(short_edge, cone, noncritical=False):
    A = (rat(2), rat(2))
    th = random.uniform(0, 2 * math.pi)
    r0 = random.uniform(0.3, 2.0)
    B = (A[0] + F(round(r0 * math.cos(th) * 997), 997),
         A[1] + F(round(r0 * math.sin(th) * 997), 997))
    if cone == "collinear":  # nearly axial approach: sigma down to ~1e-4
        phi = th + random.choice((0.0, math.pi)) + random.choice((1, -1)) * 10 ** random.uniform(-4, -1)
    elif cone:  # angle at A in [30, 150] degrees => sigma >= 1/2
        phi = th + random.choice((1, -1)) * random.uniform(math.pi / 6, 5 * math.pi / 6)
    else:
        phi = random.uniform(0, 2 * math.pi)
    dd = r0 * (random.uniform(0.002, 0.25) if short_edge else random.uniform(0.05, 3.0))
    C = (A[0] + F(round(dd * math.cos(phi) * 99991), 99991),
         A[1] + F(round(dd * math.sin(phi) * 99991), 99991))
    if C == A or C == B:
        return None
    rows = []
    for P in (A, B, C):
        rows.extend(grad_rows(P))
    ns = nullspace(rows, len(MONO))
    if not ns:
        return None
    weights = [random.randint(-9, 9) for _ in ns]
    coef = [sum(w * v[k] for w, v in zip(weights, ns)) for k in range(len(MONO))]
    if noncritical:  # negative control: drop the critical-point hypothesis
        coef = [F(random.randint(-9, 9)) for _ in MONO]
    if all(c == 0 for c in coef):
        return None
    for P in (A, B, C):  # exact criticality check (not an assert: survives -O)
        rx, ry = grad_rows(P)
        crit = (sum(a * b for a, b in zip(rx, coef)) == 0
                and sum(a * b for a, b in zip(ry, coef)) == 0)
        if crit == noncritical:
            if noncritical:
                continue
            raise RuntimeError("constructed field is not critical at a vertex")
    HA, HB, HC = (hessian(coef, P) for P in (A, B, C))
    dets = [h[0] * h[2] - h[1] ** 2 for h in (HA, HB, HC)]
    Pi = abs(float(dets[0] * dets[1] * dets[2]))
    L = lip_lower(third(coef))
    vr = (float(B[0] - A[0]), float(B[1] - A[1]))
    vd = (float(C[0] - A[0]), float(C[1] - A[1]))
    r, d = math.hypot(*vr), math.hypot(*vd)
    ell = math.hypot(float(B[0] - C[0]), float(B[1] - C[1]))
    sigma = abs(vr[0] * vd[1] - vr[1] * vd[0]) / (r * d)
    if sigma < 1e-9 or L == 0:
        return None
    nA, nB, nC = (opnorm(*h) for h in (HA, HB, HC))
    out = {}
    if d <= r / 4 and sigma >= 0.5:
        out["T1"] = Pi / ((2025 / 1024) * L ** 6 * r ** 4 * d ** 2)
    g = (L * L * r * d / (4 * sigma)) * (L * (r + (r + d) / (2 * sigma))) ** 2 \
        * (L * L * d * ell * ell / (4 * r * sigma))
    out["T2"] = Pi / g
    out["T3"] = Pi / (L ** 3 * r * d * d * nA * nB * nC / 8) if nA * nB * nC else 0.0
    return out


def main(argv):
    noncritical = "--mutant-noncritical" in argv
    random.seed(20260928)
    worst = {"T1": 0.0, "T2": 0.0, "T3": 0.0}
    count = {"T1": 0, "T2": 0, "T3": 0}
    modes = ([(True, True)] * 400 + [(True, False)] * 400 + [(False, False)] * 400
             + [(True, "collinear")] * 400)
    for mode in modes:
        res = trial(*mode, noncritical=noncritical)
        if not res:
            continue
        for k, v in res.items():
            count[k] += 1
            worst[k] = max(worst[k], v)
    for k in worst:
        status = "PASS" if worst[k] <= 1 + 1e-9 else "FAIL"
        print(f"{k}: {count[k]} triangles, max(Pi/bound) = {worst[k]:.6f}  {status}")
    return 0 if all(w <= 1 + 1e-9 for w in worst.values()) else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
