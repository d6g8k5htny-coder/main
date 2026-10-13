"""Certified enclosure of the planar cusp integral J_hat(k) and of C_*(k, 0) (standard library only).

Rigorous interval arithmetic over decimal with directed rounding; exp and sqrt are correctly rounded by the decimal
module and are widened by two units in the last place; pi is bracketed by Machin's alternating series; the integral is
enclosed cell by cell with order-K Taylor arithmetic (point expansion at the cell centre, interval remainder from the
Taylor coefficient over the whole cell), plus an exact Gaussian tail bound beyond the truncation point T.

  J_hat(k)  = int_R (1 + a^2/(144 k^2))^(11/2) exp(-a^2/4 - a^4/(576 k^2) - a^6/(248832 k^4)) da
  J_cusp(k) = J_hat(k) / sqrt(192 pi^3)                      (p_odd = N(0, diag(2, 2, 6)) normalisation)
  C_*(k, 0) = (92448/385) k^5 J_hat(k) / (sqrt(3) pi^2)     ((216/11) k^7 [p_b(0)/z_0] I J_cusp at b = 0, m_(2,0) = 1)
"""
import argparse, json, sys, time
from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN
from fractions import Fraction

PREC = 48
CF = Context(prec=PREC, rounding=ROUND_FLOOR, Emin=-9999999, Emax=9999999)
CC = Context(prec=PREC, rounding=ROUND_CEILING, Emin=-9999999, Emax=9999999)
CE = Context(prec=PREC, rounding=ROUND_HALF_EVEN, Emin=-9999999, Emax=9999999)
ZERO, ONE = Decimal(0), Decimal(1)


def ulp(x):
    if x == 0:
        return Decimal(1).scaleb(-PREC - 30)
    return Decimal(1).scaleb(x.adjusted() - PREC + 1)


class Iv:
    """Closed interval [lo, hi] of Decimals; every operation encloses the exact result."""
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        if hi is None:
            hi = lo
        if lo > hi:
            raise ValueError("empty interval")
        self.lo, self.hi = lo, hi

    @staticmethod
    def frac(q):
        q = Fraction(q)
        n, d = Decimal(q.numerator), Decimal(q.denominator)
        return Iv(CF.divide(n, d), CC.divide(n, d))

    def __add__(a, b):
        b = b if isinstance(b, Iv) else Iv.frac(b)
        return Iv(CF.add(a.lo, b.lo), CC.add(a.hi, b.hi))
    __radd__ = __add__

    def __neg__(a):
        return Iv(-a.hi, -a.lo)

    def __sub__(a, b):
        b = b if isinstance(b, Iv) else Iv.frac(b)
        return Iv(CF.subtract(a.lo, b.hi), CC.subtract(a.hi, b.lo))

    def __mul__(a, b):
        b = b if isinstance(b, Iv) else Iv.frac(b)
        c = [(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
        return Iv(min(CF.multiply(x, y) for x, y in c), max(CC.multiply(x, y) for x, y in c))
    __rmul__ = __mul__

    def __truediv__(a, b):
        b = b if isinstance(b, Iv) else Iv.frac(b)
        if b.lo <= 0 <= b.hi:
            raise ZeroDivisionError("interval division by an interval containing 0")
        c = [(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
        return Iv(min(CF.divide(x, y) for x, y in c), max(CC.divide(x, y) for x, y in c))

    def exp(a):
        lo, hi = CE.exp(a.lo), CE.exp(a.hi)
        return Iv(max(ZERO, CF.subtract(lo, 2 * ulp(lo))), CC.add(hi, 2 * ulp(hi)))

    def sqrt(a):
        if a.lo < 0:
            raise ValueError("sqrt of an interval with negative part")
        lo, hi = CE.sqrt(a.lo), CE.sqrt(a.hi)
        return Iv(max(ZERO, CF.subtract(lo, 2 * ulp(lo))), CC.add(hi, 2 * ulp(hi)))

    def width(a):
        return CC.subtract(a.hi, a.lo)

    def __repr__(self):
        return "[%s, %s]" % (self.lo, self.hi)


# ---------------- Taylor arithmetic: series = list of K+1 intervals (coefficients of t^0..t^K)
def t_const(c, K):
    return [c if isinstance(c, Iv) else Iv.frac(c)] + [Iv(ZERO)] * K


def t_var(c, K):                       # a = c + t
    s = t_const(c, K)
    if K >= 1:
        s[1] = Iv(ONE)
    return s


def t_add(u, v):
    return [x + y for x, y in zip(u, v)]


def t_scale(u, c):
    return [x * c for x in u]


def t_mul(u, v):
    K = len(u) - 1
    out = []
    for k in range(K + 1):
        acc = Iv(ZERO)
        for j in range(k + 1):
            acc = acc + u[j] * v[k - j]
        out.append(acc)
    return out


def t_exp(u):
    K = len(u) - 1
    e = [u[0].exp()]
    for k in range(1, K + 1):
        acc = Iv(ZERO)
        for j in range(1, k + 1):
            acc = acc + (u[j] * Iv.frac(j)) * e[k - j]
        e.append(acc / Iv.frac(k))
    return e


def t_sqrt(u):
    K = len(u) - 1
    s = [u[0].sqrt()]
    two_s0 = s[0] * Iv.frac(2)
    for k in range(1, K + 1):
        acc = u[k]
        for j in range(1, k):
            acc = acc - s[j] * s[k - j]
        s.append(acc / two_s0)
    return s


def t_ipow(u, n):
    out = t_const(1, len(u) - 1)
    for _ in range(n):
        out = t_mul(out, u)
    return out


def integrand_series(centre, K, k):
    """Taylor series of f(a) = (1 + s a^2)^5 sqrt(1 + s a^2) exp(-(a^2/4 + a^4/(576 k^2) + a^6/(248832 k^4))) at a = centre + t."""
    k = Fraction(k)
    s = Fraction(1, 144 * k * k)
    a = t_var(centre, K)
    a2 = t_mul(a, a)
    a4 = t_mul(a2, a2)
    a6 = t_mul(a4, a2)
    g = t_add(t_const(1, K), t_scale(a2, s))                              # 1 + s a^2
    P = t_add(t_add(t_scale(a2, Fraction(-1, 4)), t_scale(a4, -Fraction(1, 576 * k * k))),
              t_scale(a6, -Fraction(1, 248832 * k ** 4)))
    return t_mul(t_mul(t_ipow(g, 5), t_sqrt(g)), t_exp(P))


def enclose_integral(k, T, cells, K):
    """int_0^T f da as an interval: per cell, sum_{j<K} p_j int (a-c)^j + q_K int (a-c)^K with p the point series at the
    centre and q the series over the whole cell (its K-th coefficient contains f^(K)(xi)/K! for every xi in the cell)."""
    assert K % 2 == 0
    h = Fraction(T, cells)
    total = Iv(ZERO)
    max_rem = Decimal(0)
    for i in range(cells):
        lo, hi = h * i, h * (i + 1)
        c = (lo + hi) / 2
        r = Iv.frac(h / 2)
        p = integrand_series(Iv.frac(c), K, k)
        q = integrand_series(Iv(Iv.frac(lo).lo, Iv.frac(hi).hi), K, k)
        cell = Iv(ZERO)
        rpow = r                                            # r^(j+1)
        for j in range(K):
            if j % 2 == 0:
                cell = cell + p[j] * (rpow * Iv.frac(Fraction(2, j + 1)))
            rpow = rpow * r
        rem = q[K] * (rpow * Iv.frac(Fraction(2, K + 1)))    # rpow = r^(K+1) here
        max_rem = max(max_rem, rem.width())
        total = total + cell + rem
    return total, max_rem


def tail_bound(k, T):
    """Upper bound of int_T^inf f: f <= (1 + s a^2)^6 e^{-a^2/4}, and int_T^inf a^{2j} e^{-a^2/4} <= T^{-1} int_T^inf a^{2j+1} e^{-a^2/4}
    = T^{-1} 2 4^j j! e^{-T^2/4} sum_{i<=j} (T^2/4)^i / i!  (T >= 1)."""
    k = Fraction(k)
    s = Fraction(1, 144 * k * k)
    T = Fraction(T)
    x = T * T / 4
    e = Iv.frac(-x).exp()
    from math import comb, factorial
    tot = Iv(ZERO)
    for j in range(7):
        poly = sum(x ** i / factorial(i) for i in range(j + 1))
        coef = Fraction(comb(6, j)) * s ** j * 2 * 4 ** j * factorial(j) * poly / T
        tot = tot + e * Iv.frac(coef)
    return Iv(ZERO, tot.hi)


def pi_interval():
    def atan_inv(n, terms):
        x = Fraction(1, n)
        lo_sum, hi_sum = Iv(ZERO), Iv(ZERO)
        partial = Iv(ZERO)
        sums = []
        for m in range(terms + 2):
            t = Iv.frac(x ** (2 * m + 1) / (2 * m + 1))
            partial = partial + t if m % 2 == 0 else partial - t
            sums.append(partial)
        # alternating series with decreasing terms: value between consecutive partial sums
        a, b = sums[-1], sums[-2]
        return Iv(min(a.lo, b.lo), max(a.hi, b.hi))
    A, B = atan_inv(5, 40), atan_inv(239, 20)
    return A * Iv.frac(16) - B * Iv.frac(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--cells", type=int, default=160)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--ks", default="1/2,1,2")
    args = ap.parse_args()
    t0 = time.time()
    PI = pi_interval()
    sqrt3 = Iv.frac(3).sqrt()
    out = {"object": "C6-TAIL-CONSTANT-CERTIFIED-20260930-v1", "precision_digits": PREC, "taylor_order": args.K,
           "truncation": args.T, "cells": args.cells, "pi": [str(PI.lo), str(PI.hi)], "results": {}}
    for ks in args.ks.split(","):
        k = Fraction(ks)
        half, rem = enclose_integral(k, args.T, args.cells, args.K)
        tail = tail_bound(k, args.T)
        Jhat = (half + tail) * Iv.frac(2)
        Jcusp = Jhat / (Iv.frac(192) * PI * PI * PI).sqrt()
        Cstar = Jhat * Iv.frac(Fraction(92448, 385) * k ** 5) / (sqrt3 * PI * PI)
        out["results"][str(k)] = {
            "J_hat": [str(Jhat.lo), str(Jhat.hi)], "J_hat_width": str(Jhat.width()),
            "J_cusp": [str(Jcusp.lo), str(Jcusp.hi)], "C_star_b0": [str(Cstar.lo), str(Cstar.hi)],
            "C_star_b0_width": str(Cstar.width()), "tail_upper_bound": str(tail.hi), "max_cell_remainder_width": str(rem)}
    out["seconds"] = round(time.time() - t0, 2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
