"""Exact checks for CL-D5-COLLAR-20260928-v1 (author-side, Claude).

Python standard library only. Verifies, as exact identities or exact
rational inequalities:

  1. the degree-five pin solution satisfies all six pins;
  2. in collar coordinates X = M + r(p,q), Y = (f_x(X)/r^2, f_z(X)/r) has
     deterministic mean (6kp(p-1), 0), and every random-jet entry of its
     coefficient matrix carries a factor q or r (so tr(BB^T) <= C(q^2+r^2)
     on bounded (p,q));
  3. three exact 2x2 minors:
        (S0,C)  = -q^3/2
        (R4,T)  =  r^2 p^2 (p-1)^2 (2p-1)/24
        (R5,T)  =  r^3 p^2 (p-1)^2 (5p^2+5p-4)/240
  4. 100(2p-1)^2 + (5p^2+5p-4)^2 >= 1/70 for every real p, so the last two
     minors never vanish together off the pins;
  5. on the axis strip {|q|<=1/8, |X-M|,|X-S|>=r/4} one has |p(p-1)|>=1/6;
  6. the symmetric three-vertex bound (T4)
        |det H_A det H_B det H_C| <= L^6 (abc)^4 / (512 K^3)
     on seeded random exact-rational cubics with three prescribed critical
     points (finite evidence; L from below by direction sampling).

Not a continuum or Gaussian proof; the covariance, density and moment
arguments are in NOTE.md.
"""
import argparse
import json
import math
import random
import sys
from fractions import Fraction as F
from math import factorial as fa

VARS = ("r", "p", "q", "k", "S0", "T", "C", "D", "R4", "E1", "E2", "E3", "F4",
        "R5", "F41", "F32", "F23", "F14", "F05")
IDX = {v: i for i, v in enumerate(VARS)}
RANDOM = VARS[4:]


class Poly:
    __slots__ = ("t",)

    def __init__(self, terms=None):
        self.t = {m: c for m, c in (terms or {}).items() if c != 0}

    @staticmethod
    def const(c):
        return Poly({(0,) * len(VARS): F(c)})

    def _c(self, o):
        return o if isinstance(o, Poly) else Poly.const(o)

    def __add__(self, o):
        o = self._c(o)
        t = dict(self.t)
        for m, c in o.t.items():
            t[m] = t.get(m, 0) + c
        return Poly(t)

    __radd__ = __add__

    def __neg__(self):
        return Poly({m: -c for m, c in self.t.items()})

    def __sub__(self, o):
        return self + (-self._c(o))

    def __rsub__(self, o):
        return self._c(o) - self

    def __mul__(self, o):
        o = self._c(o)
        t = {}
        for m1, c1 in self.t.items():
            for m2, c2 in o.t.items():
                m = tuple(a + b for a, b in zip(m1, m2))
                t[m] = t.get(m, 0) + c1 * c2
        return Poly(t)

    __rmul__ = __mul__

    def __pow__(self, n):
        out = Poly.const(1)
        for _ in range(n):
            out = out * self
        return out

    def is_zero(self):
        return not self.t

    def coeff(self, name):
        i = IDX[name]
        t = {}
        for m, c in self.t.items():
            if m[i] > 1:
                raise ValueError(name + " nonlinear")
            if m[i] == 1:
                mm = list(m)
                mm[i] = 0
                t[tuple(mm)] = c
        return Poly(t)

    def without(self, names):
        idx = [IDX[n] for n in names]
        return Poly({m: c for m, c in self.t.items() if all(m[i] == 0 for i in idx)})

    def shift_r(self, n):
        i = IDX["r"]
        return Poly({tuple(e - n if j == i else e for j, e in enumerate(m)): c for m, c in self.t.items()})


def V(name):
    m = [0] * len(VARS)
    m[IDX[name]] = 1
    return Poly({tuple(m): F(1)})


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


THIRD = {(2, 1): "T", (1, 2): "C", (0, 3): "D"}
FOURTH = {(4, 0): "R4", (3, 1): "E1", (2, 2): "E2", (1, 3): "E3", (0, 4): "F4"}
FIFTH = {(5, 0): "R5", (4, 1): "F41", (3, 2): "F32", (2, 3): "F23", (1, 4): "F14", (0, 5): "F05"}


def model(mutant=None):
    r, k = V("r"), V("k")
    J = {n: V(n) for n in RANDOM}
    fxxx = 12 * k - F(1, 2) * r * J["R4"] - F(3, 20) * r ** 2 * J["R5"]
    fxx = -6 * k * r + F(1, 12) * r ** 2 * J["R4"] + F(1, 30) * r ** 3 * J["R5"]
    fxz = -F(1, 2) * r * J["T"] - F(1, 6) * r ** 2 * J["E1"] - F(1, 24) * r ** 3 * J["F41"]
    if mutant == "pin-r5":
        fxx = fxx + F(1, 30) * r ** 3 * J["R5"]
    coef = {(2, 0): fxx, (1, 1): fxz, (0, 2): J["S0"], (3, 0): fxxx}
    for tab in (THIRD, FOURTH, FIFTH):
        coef.update({ij: J[n] for ij, n in tab.items()})

    def grad(x, z):
        fx, fz = Poly(), Poly()
        for (i, j), c in coef.items():
            if i:
                fx = fx + c * F(1, fa(i - 1) * fa(j)) * x ** (i - 1) * z ** j
            if j:
                fz = fz + c * F(1, fa(i) * fa(j - 1)) * x ** i * z ** (j - 1)
        return fx, fz

    def height(x, z):
        out = Poly()
        for (i, j), c in coef.items():
            out = out + c * F(1, fa(i) * fa(j)) * x ** i * z ** j
        return out

    return grad, height


def check_algebra(mutant=None):
    r, p, q, k = V("r"), V("p"), V("q"), V("k")
    grad, height = model(mutant)
    fxS, fzS = grad(r, Poly())
    require(fxS.is_zero(), "pin f_x(S)=0")
    require(fzS.is_zero(), "pin f_z(S)=0")
    require((height(r, Poly()) + k * r ** 3).is_zero(), "pin f(S)=b-kr^3")
    fx, fz = grad(r * p, r * q)
    Y1, Y2 = fx.shift_r(2), fz.shift_r(1)
    require((Y1.without(RANDOM) - 6 * k * p * (p - 1)).is_zero(), "mean Y1")
    require(Y2.without(RANDOM).is_zero(), "mean Y2")
    for Y in (Y1, Y2):
        for n in RANDOM:
            for m in Y.coeff(n).t:
                require(m[IDX["r"]] >= 0, "negative r power in " + n)
                require(m[IDX["q"]] + m[IDX["r"]] >= 1, "entry without q or r factor: " + n)
    row = {n: (Y1.coeff(n), Y2.coeff(n)) for n in RANDOM}

    def minor(a, b):
        return row[a][0] * row[b][1] - row[b][0] * row[a][1]

    require((minor("S0", "C") + F(1, 2) * q ** 3).is_zero(), "minor (S0,C)")
    require((minor("R4", "T") - F(1, 24) * r ** 2 * p ** 2 * (p - 1) ** 2 * (2 * p - 1)).is_zero(), "minor (R4,T)")
    require((minor("R5", "T") - F(1, 240) * r ** 3 * p ** 2 * (p - 1) ** 2 * (5 * p ** 2 + 5 * p - 4)).is_zero(),
            "minor (R5,T)")
    return len(RANDOM)


def g(pv):
    return 100 * (2 * pv - 1) ** 2 + (5 * pv ** 2 + 5 * pv - 4) ** 2


def check_separation():
    # |2p-1| >= 1/40  =>  100(2p-1)^2 >= 1/16 > 1/70.
    require(100 * F(1, 40) ** 2 >= F(1, 70), "outer case")
    # p in [39/80, 41/80]: 5p^2+5p-4 is increasing (10p+5>0) and negative at 41/80.
    lo, hi = F(39, 80), F(41, 80)
    require(10 * lo + 5 > 0, "monotone")
    v_hi = 5 * hi ** 2 + 5 * hi - 4
    require(v_hi < 0 and v_hi ** 2 >= F(1, 70), "inner case")
    require(g(F(1, 2)) == F(1, 16), "value at midpoint")
    return str(F(1, 70))


def check_strip():
    # |q|<=1/8 and p^2+q^2>=1/16, (p-1)^2+q^2>=1/16  =>  min(|p|,|p-1|) >= sqrt(3)/8 >= 433/2000.
    s = F(433, 2000)
    require(s ** 2 <= F(1, 16) - F(1, 64), "radius bound")
    # |p(p-1)| minimized at the boundary p=s or p=1-s (inner segment) or p=-s (outer).
    require(s * (1 - s) >= F(1, 6), "inner segment")
    require(s * (1 + s) >= F(1, 6), "outer rays")
    return str(F(1, 6))


# ---- T4 finite evidence -----------------------------------------------------
MONO = [(i, s - i) for s in (1, 2, 3) for i in range(s + 1)]


def _grad_rows(pt):
    x, y = pt
    rx = [i * x ** (i - 1) * y ** j if i else F(0) for i, j in MONO]
    ry = [j * x ** i * y ** (j - 1) if j else F(0) for i, j in MONO]
    return rx, ry


def _nullspace(rows, n):
    m = [list(rw) for rw in rows]
    piv, rk = [], 0
    for c in range(n):
        pr = next((i for i in range(rk, len(m)) if m[i][c] != 0), None)
        if pr is None:
            continue
        m[rk], m[pr] = m[pr], m[rk]
        inv = 1 / m[rk][c]
        m[rk] = [v * inv for v in m[rk]]
        for i in range(len(m)):
            if i != rk and m[i][c] != 0:
                fac = m[i][c]
                m[i] = [a - fac * b for a, b in zip(m[i], m[rk])]
        piv.append(c)
        rk += 1
    basis = []
    for fc in (c for c in range(n) if c not in piv):
        v = [F(0)] * n
        v[fc] = F(1)
        for i, pc in enumerate(piv):
            v[pc] = -m[i][fc]
        basis.append(v)
    return basis


def _opnorm(a, b, c):
    a, b, c = float(a), float(b), float(c)
    return abs((a + c) / 2) + math.hypot((a - c) / 2, b)


def check_t4(noncritical=False, trials=600):
    rnd = random.Random(20260928)
    worst, count = 0.0, 0
    for _ in range(trials):
        pts = [(F(rnd.randint(-300, 300), 97), F(rnd.randint(-300, 300), 97)) for _ in range(3)]
        if len(set(pts)) < 3:
            continue
        rows = []
        for pt in pts:
            rows.extend(_grad_rows(pt))
        ns = _nullspace(rows, len(MONO))
        w = [rnd.randint(-9, 9) for _ in ns]
        coef = [sum(wi * v[i] for wi, v in zip(w, ns)) for i in range(len(MONO))]
        if noncritical:
            coef = [F(rnd.randint(-9, 9)) for _ in MONO]
        d = dict(zip(MONO, coef))
        if not noncritical:
            for pt in pts:
                rx, ry = _grad_rows(pt)
                if sum(a * b for a, b in zip(rx, coef)) or sum(a * b for a, b in zip(ry, coef)):
                    raise ValueError("constructed field not critical")
        t = (6 * d[(3, 0)], 2 * d[(2, 1)], 2 * d[(1, 2)], 6 * d[(0, 3)])
        L = 0.0
        for kk in range(4096):
            th = math.pi * kk / 4096
            c1, s1 = math.cos(th), math.sin(th)
            L = max(L, _opnorm(float(t[0]) * c1 + float(t[1]) * s1, float(t[1]) * c1 + float(t[2]) * s1,
                               float(t[2]) * c1 + float(t[3]) * s1))
        if L == 0:
            continue

        def hess(pt):
            x, y = pt
            hxx = 2 * d[(2, 0)] + 6 * d[(3, 0)] * x + 2 * d[(2, 1)] * y
            hxy = d[(1, 1)] + 2 * d[(2, 1)] * x + 2 * d[(1, 2)] * y
            hyy = 2 * d[(0, 2)] + 2 * d[(1, 2)] * x + 6 * d[(0, 3)] * y
            return hxx * hyy - hxy ** 2

        Pi = abs(float(hess(pts[0]) * hess(pts[1]) * hess(pts[2])))
        (ax, ay), (bx, by), (cx, cy) = [(float(x), float(y)) for x, y in pts]
        a = math.hypot(bx - cx, by - cy)
        b = math.hypot(ax - cx, ay - cy)
        c = math.hypot(ax - bx, ay - by)
        K = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2
        if K < 1e-12:
            continue
        worst = max(worst, Pi / (L ** 6 * (a * b * c) ** 4 / (512 * K ** 3)))
        count += 1
    return count, worst


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", choices=["pin-r5", "t4-noncritical"])
    args = ap.parse_args(argv)
    try:
        jets = check_algebra("pin-r5" if args.mutant == "pin-r5" else None)
        gmin = check_separation()
        pmin = check_strip()
        n, worst = check_t4(noncritical=args.mutant == "t4-noncritical")
        require(worst <= 1 + 1e-9, "T4 bound violated")
    except ValueError as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps({
        "pins_verified": 3,
        "random_jets_through_degree_5": jets,
        "mean": "(6kp(p-1), 0) exact in the degree-five model",
        "every_entry_has_q_or_r_factor": True,
        "minors": ["(S0,C)=-q^3/2", "(R4,T)=r^2 p^2(p-1)^2(2p-1)/24", "(R5,T)=r^3 p^2(p-1)^2(5p^2+5p-4)/240"],
        "separation_lower_bound": gmin,
        "axis_strip_min_abs_p_p_minus_1": pmin,
        "t4_triangles": n,
        "t4_max_ratio": "%.6f" % worst,
        "scope": "exact identities and finite evidence only; Gaussian arguments are in NOTE.md",
        "scientific_effect": "NONE",
        "passed": True,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
