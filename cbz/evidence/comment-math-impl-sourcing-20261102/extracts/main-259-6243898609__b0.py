#!/usr/bin/env python3
"""Exact check of the section 3 moving-corner witness in main#259/6004452810.

Polynomial arithmetic over Fraction only. Run with --mutant NAME to apply one
negative control; every mutant must make some check fail (exit 1).
Prints one summary line and exits nonzero on failure.
"""
import sys
from fractions import Fraction as F

MUTANTS = ("none", "transverse-slope-7", "quartic-sign", "drop-cubic", "gamma-2t")


def poly_add(a, b):
    c = dict(a)
    for k, v in b.items():
        c[k] = c.get(k, 0) + v
    return {k: v for k, v in c.items() if v != 0}


def poly_mul(a, b):
    c = {}
    for (i, j), u in a.items():
        for (k, l), v in b.items():
            c[(i + k, j + l)] = c.get((i + k, j + l), 0) + u * v
    return {k: v for k, v in c.items() if v != 0}


def const(c):
    return {(0, 0): F(c)} if c != 0 else {}


X = {(1, 0): F(1)}
Y = {(0, 1): F(1)}


def d(p, ax, n=1):
    for _ in range(n):
        q = {}
        for (i, j), v in p.items():
            e = i if ax == 0 else j
            if e:
                key = (i - 1, j) if ax == 0 else (i, j - 1)
                q[key] = q.get(key, 0) + v * e
        p = q
    return p


def ev(p, x, y):
    return sum((v * F(x) ** i * F(y) ** j for (i, j), v in p.items()), F(0))


def field(t, b, mutant):
    r, k, lam, gam = t ** 3, t ** 4, t ** 2, t
    if mutant == "gamma-2t":
        gam = 2 * t
    slope = 7 if mutant == "transverse-slope-7" else 8
    xr2 = poly_add(poly_mul(X, X), const(-r * r / 4))           # x^2 - r^2/4
    cubic = poly_mul(poly_mul(poly_add(X, const(r / 2)), poly_add(X, const(r / 2))), poly_add(X, const(-r)))
    quart = poly_mul(xr2, xr2)
    p = const(b)
    if mutant != "drop-cubic":
        p = poly_add(p, poly_mul(const(2 * k), cubic))
    qs = F(1, 8) if mutant == "quartic-sign" else F(-1, 8)
    p = poly_add(p, poly_mul(const(qs), quart))
    g = poly_mul(const(gam / 2), xr2)
    trans = poly_add(const(-lam), poly_mul(const(slope), X))     # -lambda + 8x
    f = poly_add(poly_add(p, poly_mul(g, Y)), poly_mul(poly_mul(trans, poly_mul(Y, Y)), const(F(1, 2))))
    return f, p, g, trans, dict(r=r, k=k, lam=lam, kappa=t, ell=t ** 13)


def run(mutant):
    fails = []

    def ck(cond, msg):
        if not cond:
            fails.append(msg)
    ts = [F(1, 10), F(1, 17), F(2, 41), F(1, 100)]
    Xs = [F(-1, 2), F(-1, 3), F(0), F(1, 5), F(1, 2), F(1), F(-2), F(3, 2)]
    for t in ts:
        b = F(3, 7)
        f, p, g, trans, q = field(t, b, mutant)
        r, k, lam, kap, ell = q["r"], q["k"], q["lam"], q["kappa"], q["ell"]
        fx, fy = d(f, 0), d(f, 1)
        M, S = (-r / 2, 0), (r / 2, 0)
        # pins: critical points and heights (b, b - k r^3)
        for P in (M, S):
            ck(ev(fx, *P) == 0 and ev(fy, *P) == 0, "critical point t=%s" % t)
        ck(ev(f, *M) == b and ev(f, *S) == b - k * r ** 3, "pin heights t=%s" % t)
        ck(ell == k * r ** 3 == kap * r ** 4 and kap == k / r, "scales t=%s" % t)
        # midpoint data
        A = ev(d(f, 1, 2), 0, 0)
        B = ev(d(d(f, 1, 2), 0), 0, 0)
        f4 = ev(d(f, 0, 4), 0, 0)
        f5 = ev(d(f, 0, 5), 0, 0)
        gm = ev(d(d(f, 0, 2), 1), 0, 0)
        eta = ev(d(d(f, 0, 3), 1), 0, 0)
        ck(A == -lam and B == 8 and f4 == -3 and f5 == 0 and gm == t and eta == 0, "midpoint jets t=%s" % t)
        z = f4 - 3 * gm * gm / A
        Q1 = f5 / 120 - eta * gm / A / 12 + gm * B * gm / (A * A) / 8
        ck(all(type(v) is F for v in (A, B, f4, f5, gm, eta, z, Q1)), "exact Fraction types t=%s" % t)
        ck(z == 0, "z=0 t=%s" % t)
        ck(Q1 == 1 / lam and r * Q1 / kap == 1, "rQ1/kappa=1 t=%s" % t)
        # typed Hessians and rescaled Schur complements
        hxx, hxy, hyy = d(f, 0, 2), d(d(f, 0), 1), d(f, 1, 2)
        HM = (ev(hxx, *M), ev(hxy, *M), ev(hyy, *M))
        HS = (ev(hxx, *S), ev(hxy, *S), ev(hyy, *S))
        ck(HM[0] < 0 and HM[0] * HM[2] - HM[1] ** 2 > 0, "M is a nondegenerate maximum t=%s" % t)
        ck(HS[0] * HS[2] - HS[1] ** 2 < 0 and HS[2] < 0 and HM[2] < 0, "S saddle, negative transverse blocks t=%s" % t)
        def schur(H):
            a, bb, AA = H[0] / r ** 2, H[1] / r, H[2]
            return a - bb * bb / AA
        ck(schur(HM) == -6 * t - t / (1 + 4 * t), "Schur at M t=%s" % t)
        ck(schur(HS) == 6 * t + t / (1 - 4 * t), "Schur at S t=%s" % t)
        # eliminated y and the quintic limit form
        for Xv in Xs:
            x = r * Xv
            den = ev(trans, x, 0)
            if not (den < 0):          # need -lambda + 8x < 0 for a strict max in y
                continue
            gx = ev(g, x, 0)
            ystar = gx / (-den)
            mx = ev(p, x, 0) + gx * gx / (2 * (-den))
            ck(ev(f, x, ystar) == mx and ev(fy, x, ystar) == 0, "max over y t=%s X=%s" % (t, Xv))
            lhs = (mx - b) / ell
            rhs = 2 * (Xv + F(1, 2)) ** 2 * (Xv - 1) + Xv * (Xv * Xv - F(1, 4)) ** 2 / (1 - 8 * t * Xv)
            ck(lhs == rhs, "eliminated-y identity t=%s X=%s" % (t, Xv))
            ck(ystar == t ** 5 * (Xv * Xv - F(1, 4)) / (2 * (1 - 8 * t * Xv)), "ridge displacement r^(5/3) t=%s X=%s" % (t, Xv))
        # transverse curvature on the segment
        mag = max(-ev(hyy, s, 0) for s in (-r / 2, r / 2))
        ck(mag == lam + 4 * r and lam == r / kap, "max transverse magnitude t=%s" % t)
    return fails


def main():
    mutant = "none"
    if "--mutant" in sys.argv:
        mutant = sys.argv[sys.argv.index("--mutant") + 1]
    if mutant == "all":
        bad = []
        for m in MUTANTS[1:]:
            if not run(m):
                bad.append(m)
        base = run("none")
        ok = not base and not bad
        print("witness259 %s: baseline %d failures; mutants caught %d/%d%s" % (
            "PASS" if ok else "FAIL", len(base), len(MUTANTS) - 1 - len(bad), len(MUTANTS) - 1,
            "" if not bad else "; NOT caught: " + ", ".join(bad)))
        sys.exit(0 if ok else 1)
    fails = run(mutant)
    print("witness259 mutant=%s: %s (%d failing checks%s)" % (
        mutant, "PASS" if not fails else "FAIL", len(fails), "" if not fails else "; first: " + fails[0]))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
