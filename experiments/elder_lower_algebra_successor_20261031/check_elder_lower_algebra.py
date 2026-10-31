"""Exact successor check of the elementary algebra in ELDER_LOWER_AND_DENSITY_GAP.md.

Object: ELDER-LOWER-ALGEBRA-SUCCESSOR-01CZ-20261031. Credit 0 · scientific effect NONE · OBL OPEN.

The original packet's `algebra.py` (Math- frontiers/window_multiplicity_laws_20260928,
Git blob 00140088, sha256 16ed713d...3350) was blocked from publication and is not
recoverable from any store this author can read. This file is NOT that file and does
not reproduce it. It independently re-derives, in exact rational arithmetic, the
finite algebraic identities and inequalities that the source text displays in §2-§3
(E5)-(E7) and the two Hessians, plus the extra-saddle height and two README identities.

Everything here is a polynomial identity or an inequality on a compact interval,
checked exactly with fractions.Fraction. There are no floats. Because G_k = k * G_1 and
all checked statements are homogeneous in k > 0, the checks are done at k = 1.

What this does NOT establish: the probabilistic steps of the source (positive
conditional density of the jet box, the conditional C^4 moment, W_r >= c_2 r^4,
Z_r <= C_Z r^2, hence (E2)/(E8)); the upper half of (E4); any constant c or r_*;
correctness of the definition (E1) as the elder death height; anything about the
density loss (E9)-(E11). A pass here means the displayed algebra is right, not that
Theorem E1 holds.

Usage:  python3 check_elder_lower_algebra.py [--mutate NAME]
Prints one summary line; exits 0 on success, 1 on any failed check.
"""
from __future__ import annotations

import argparse
import sys
from fractions import Fraction as F

# ---------------------------------------------------------------- polynomials
# A polynomial is a dict {exponent tuple: Fraction}. Variables are positional.


def padd(*ps):
    out = {}
    for p in ps:
        for e, c in p.items():
            out[e] = out.get(e, F(0)) + c
    return {e: c for e, c in out.items() if c != 0}


def pmul(p, q):
    out = {}
    for e1, c1 in p.items():
        for e2, c2 in q.items():
            e = tuple(a + b for a, b in zip(e1, e2))
            out[e] = out.get(e, F(0)) + c1 * c2
    return {e: c for e, c in out.items() if c != 0}


def pscale(p, c):
    return {e: c * v for e, v in p.items() if c * v != 0}


def ppow(p, n, nvars):
    out = {(0,) * nvars: F(1)}
    for _ in range(n):
        out = pmul(out, p)
    return out


def const(c, nvars):
    return {(0,) * nvars: F(c)} if F(c) != 0 else {}


def var(i, nvars):
    e = [0] * nvars
    e[i] = 1
    return {tuple(e): F(1)}


def pderiv(p, i):
    out = {}
    for e, c in p.items():
        if e[i]:
            e2 = list(e)
            e2[i] -= 1
            out[tuple(e2)] = out.get(tuple(e2), F(0)) + c * e[i]
    return {e: c for e, c in out.items() if c != 0}


def peval(p, point):
    total = F(0)
    for e, c in p.items():
        term = c
        for x, k in zip(point, e):
            term *= F(x) ** k
        total += term
    return total


def compose(p, subs, nvars_out):
    """Substitute polynomial subs[i] (in nvars_out variables) for variable i of p."""
    out = {}
    for e, c in p.items():
        term = const(c, nvars_out)
        for i, k in enumerate(e):
            if k:
                term = pmul(term, ppow(subs[i], k, nvars_out))
        out = padd(out, term)
    return out


# ------------------------------------------------------------- the source data
X2 = (var(0, 2), var(1, 2))  # (X, Z)


def G1(mut=None):
    """G_k / k = 2X^3 - 3X/2 - 1/2 - 3Z^2/4 - X Z^2."""
    X, Z = X2
    cx3 = F(2) if mut != "cubic_coeff" else F(3)
    return padd(pscale(ppow(X, 3, 2), cx3), pscale(X, F(-3, 2)), const(F(-1, 2), 2),
                pscale(ppow(Z, 2, 2), F(-3, 4)), pscale(pmul(X, ppow(Z, 2, 2)), F(-1)))


PATH = [(F(-1, 2), F(0)), (F(-3, 4), F(0)), (F(-3, 4), F(9, 4)), (F(-1), F(9, 4))]  # (E5)
V = var(0, 1)
G_STATED = [  # (E6), polynomials in v
    padd(pscale(ppow(V, 2, 1), F(-3, 16)), pscale(ppow(V, 3, 1), F(-1, 32))),
    const(F(-7, 32), 1),
    padd(const(F(-7, 32), 1), pscale(V, F(51, 64)), pscale(ppow(V, 2, 1), F(-9, 32)),
         pscale(ppow(V, 3, 1), F(-1, 32))),
]


def restrict(g, a, b):
    """g along the segment a -> b, parameterized by v in [0, 1]."""
    sub = [padd(const(a[i], 1), pscale(V, b[i] - a[i])) for i in range(2)]
    return compose(g, sub, 1)


def min_on_unit_interval_monotone(p, sign):
    """For p monotone on [0,1] (sign=+1 nondecreasing, -1 nonincreasing) return its min."""
    return peval(p, [F(0)]) if sign > 0 else peval(p, [F(1)])


def poly_nonneg_on_unit(p, lower):
    """Exact check that p(v) >= lower on [0,1] for p of degree <= 2 (vertex + endpoints)."""
    deg = max((e[0] for e in p), default=0)
    assert deg <= 2, "only quadratics supported"
    pts = [F(0), F(1)]
    a2 = p.get((2,), F(0))
    a1 = p.get((1,), F(0))
    if a2 != 0:
        vtx = -a1 / (2 * a2)
        if 0 <= vtx <= 1:
            pts.append(vtx)
    return min(peval(p, [t]) for t in pts) >= lower


# ---------------------------------------------------------------------- checks
def run(mut=None):
    fails = []

    def ok(name, cond):
        if not cond:
            fails.append(name)

    g = G1(mut)
    gx, gz = pderiv(g, 0), pderiv(g, 1)

    # Pins: M=(-1/2,0) at height 0 (= b), S=(1/2,0) at height -1 (= b - k r^3); gradients zero.
    M, S = (F(-1, 2), F(0)), (F(1, 2), F(0))
    ok("pin_M_height_0", peval(g, M) == 0)
    ok("pin_S_height_-1", peval(g, S) == -1)
    ok("pin_gradients_zero", all(peval(d, P) == 0 for d in (gx, gz) for P in (M, S)))

    # Hessians: diag(-6,-1/2) at M (a maximum), diag(6,-5/2) at S (index one).
    gxx, gxz, gzz = pderiv(gx, 0), pderiv(gx, 1), pderiv(gz, 1)
    hM = (peval(gxx, M), peval(gxz, M), peval(gzz, M))
    hS = (peval(gxx, S), peval(gxz, S), peval(gzz, S))
    ok("hessian_M", hM == (F(-6), F(0), F(-1, 2)))
    ok("hessian_S", hS == (F(6), F(0), F(-5, 2)))

    # (E5): every path vertex in B(0,3); largest squared norm 97/16 < 9.
    sq = [x * x + z * z for x, z in PATH]
    ok("path_in_ball", max(sq) == F(97, 16) and max(sq) < 9)

    # (E6): the exact restricted polynomials.
    restr = [restrict(g, PATH[i], PATH[i + 1]) for i in range(3)]
    stated = list(G_STATED)
    if mut == "e6_g3_linear":
        stated[2] = padd(stated[2], pscale(V, F(1, 64)))
    for i in range(3):
        ok(f"e6_g{i+1}_identity", restr[i] == stated[i])

    # g1 nonincreasing on [0,1]: g1' = -3v/8 - 3v^2/32 <= 0 there.
    d1 = pderiv(restr[0], 0)
    ok("g1_nonincreasing", poly_nonneg_on_unit(pscale(d1, F(-1)), F(0)))
    # g2 constant.
    ok("g2_constant", pderiv(restr[1], 0) == {})
    # g3' = (51-36v-6v^2)/64 >= 9/64 on [0,1].
    d3 = pderiv(restr[2], 0)
    bound = F(9, 64) if mut != "g3_prime_bound" else F(10, 64)
    ok("g3_prime_formula", d3 == padd(const(F(51, 64), 1), pscale(V, F(-36, 64)),
                                      pscale(ppow(V, 2, 1), F(-6, 64))))
    ok("g3_prime_lower_bound", poly_nonneg_on_unit(d3, bound))

    # Path minimum -7/32 and final value 17/64.
    path_min = min(min_on_unit_interval_monotone(restr[0], -1), peval(restr[1], [F(0)]),
                   min_on_unit_interval_monotone(restr[2], +1))
    ok("path_min_-7/32", path_min == F(-7, 32))
    ok("path_end_17/64", peval(restr[2], [F(1)]) == F(17, 64))

    # Perturbation margins with ||F - G|| < 1/64: path > -15/64 > -1/4, end > 16/64 = 1/4 > 0.
    eps = F(1, 64) if mut != "margin_eps" else F(1, 16)
    ok("margin_path_above_-1/4", path_min - eps >= F(-15, 64) and F(-15, 64) > F(-1, 4))
    ok("margin_end_above_1/4", peval(restr[2], [F(1)]) - eps >= F(1, 4) and F(1, 4) > 0)
    # (E3): d_f(M) - f(S) >= 3/4, using d_f(M) >= -1/4 and f(S) = -1.
    ok("e3_gap_3/4", F(-1, 4) - peval(g, S) == F(3, 4))

    # (E7) at the targets s=-3/2, a=0, beta=-2, c=0 recovers G_1 exactly.
    X, Z = X2
    s, a, beta, c = F(-3, 2), F(0), F(-2), F(0)
    if mut == "e7_target_beta":
        beta = F(-1)
    e7 = padd(pscale(ppow(X, 3, 2), F(2)), pscale(X, F(-3, 2)), const(F(-1, 2), 2),
              pscale(ppow(Z, 2, 2), s / 2),
              pscale(pmul(padd(ppow(X, 2, 2), const(F(-1, 4), 2)), Z), a / 2),
              pscale(pmul(X, ppow(Z, 2, 2)), beta / 2), pscale(ppow(Z, 3, 2), c / 6))
    ok("e7_targets_recover_G", e7 == G1(None))

    # Extra saddles on X = -3/4 with Z^2 = 15/8, at height -7/32, index one (det < 0).
    Zsq = F(15, 8)
    ok("extra_saddle_line_level", restrict(g, (F(-3, 4), F(0)), (F(-3, 4), F(1))) ==
       const(F(-7, 32), 1))
    gx_line = restrict(gx, (F(-3, 4), F(0)), (F(-3, 4), F(1)))  # = 15/8 - v^2
    ok("extra_saddle_gx_zero", gx_line == padd(const(Zsq, 1), pscale(ppow(V, 2, 1), F(-1))))
    gzz_line = peval(gzz, (F(-3, 4), F(0)))
    gxx_line = peval(gxx, (F(-3, 4), F(0)))
    # det = gxx*gzz - gxz^2 with gxz = -2Z, so det = gxx*gzz - 4 Z^2.
    ok("extra_saddle_index_one", gxx_line * gzz_line - 4 * Zsq < 0)
    ok("extra_saddle_inside_path_segment", Zsq < F(9, 4) ** 2)

    # Jet-box volume 16 eps^4 k^4 r: widths 2 eps k r, 2 eps k, 2 eps k, 2 eps k.
    ok("jet_box_volume", F(2) ** 4 == 16)

    # README: integral_0^inf s (k - s^3 t)_+ ds = (3/10) k^(5/3) t^(-2/3), at exact cube points.
    for kk, tt in ((F(8), F(1)), (F(1), F(1, 27)), (F(27, 8), F(1, 8))):
        sstar_cubed = kk / tt
        sstar = round_cube_root(sstar_cubed)
        lhs = kk * sstar ** 2 / 2 - tt * sstar ** 5 / 5
        rhs = F(3, 10) * kk * sstar ** 2  # k^(5/3) t^(-2/3) = k * s*^2 exactly
        coeff = F(3, 10) if mut != "overlap_coeff" else F(1, 3)
        ok(f"overlap_integral_{kk}_{tt}", lhs == rhs and rhs == coeff * kk * sstar ** 2)

    # README: Beta(2/3, 2) mean 1/4 and variance 9/176.
    al, be = F(2, 3), F(2)
    ok("beta_mean_1/4", al / (al + be) == F(1, 4))
    ok("beta_var_9/176", al * be / ((al + be) ** 2 * (al + be + 1)) == F(9, 176))

    return fails


def icbrt(n):
    """Integer cube root by bisection (no floats)."""
    lo, hi = 0, max(1, n)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid ** 3 <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


def round_cube_root(q):
    """Exact rational cube root of a positive perfect rational cube."""
    a, b = icbrt(q.numerator), icbrt(q.denominator)
    if b == 0 or F(a, b) ** 3 != q:
        raise ValueError(f"{q} is not a perfect rational cube")
    return F(a, b)


MUTANTS = ("cubic_coeff", "e6_g3_linear", "g3_prime_bound", "margin_eps",
           "e7_target_beta", "overlap_coeff")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutate", choices=MUTANTS)
    args = ap.parse_args(argv)
    fails = run(args.mutate)
    tag = f" [mutant {args.mutate}]" if args.mutate else ""
    if fails:
        print(f"elder-lower algebra successor{tag}: FAIL ({len(fails)}): " + ", ".join(fails))
        return 1
    print(f"elder-lower algebra successor{tag}: OK (exact; identity checks only, "
          "no probabilistic step checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
