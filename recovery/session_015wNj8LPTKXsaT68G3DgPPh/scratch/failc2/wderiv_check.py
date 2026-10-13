"""Finite-difference check of the (B, w) derivative formulas, w = |u|, T = 3 w^2, for the big root (sigma=+1) and the
middle root (sigma=-1)."""
import math, random
def roots(B, T):
    # all real roots of (x+B)^2(2x-B) - T via numpy-free bisection on monotone pieces
    P = lambda x: (x + B) ** 2 * (2 * x - B) - T
    def bis(lo, hi, inc):
        for _ in range(200):
            m = 0.5 * (lo + hi)
            if (P(m) > 0) == inc: hi = m
            else: lo = m
        return 0.5 * (lo + hi)
    xm = -B if B < 0 else B / 2
    big = bis(xm, xm + abs(B) + (T / 2) ** (1 / 3) + 1, True)
    mid = None
    if B < 0 and T < abs(B) ** 3 / 2:
        mid = bis(abs(B) / 2, abs(B), False)
    return big, mid
def V_of(B, w, which):
    big, mid = roots(B, 3 * w * w)
    x = big if which == 'big' else mid
    return -3 * B * (2 * x - B) * (x + B)
def formulas(B, w, which):
    big, mid = roots(B, 3 * w * w)
    x = big if which == 'big' else mid
    sg = 1 if which == 'big' else -1
    r = B / x
    V = -3 * x ** 3 * r * (2 - r) * (1 + r)
    VB = x * x * (-6 + 4.5 * r * r - 1.5 * r ** 3)
    Vw = -sg * math.sqrt(3) * x ** 1.5 * r * (4 + r) * math.sqrt(2 - r)
    VBB = x * (6 + 3 * r - 4.5 * r * r - 0.75 * r ** 3 + 0.75 * r ** 4)
    VBw = -sg * (math.sqrt(3) / 2) * math.sqrt(x) * (2 - r) ** 1.5 * (4 + 2 * r + r * r)
    Vww = -r * (4 - r + r * r)
    return V, VB, Vw, VBB, VBw, Vww
random.seed(1)
worst = 0
for which in ('big', 'mid'):
    for _ in range(300):
        B = random.uniform(-3, 3)
        if which == 'mid':
            B = -abs(B) - 0.2
            wmax = math.sqrt(abs(B) ** 3 / 6)
            w = random.uniform(0.05, 0.9) * wmax
        else:
            w = random.uniform(0.02, 3)
        h = 1e-4
        F = lambda b, ww: V_of(b, ww, which)
        V, VB, Vw, VBB, VBw, Vww = formulas(B, w, which)
        fd = [F(B, w),
              (F(B + h, w) - F(B - h, w)) / (2 * h),
              (F(B, w + h) - F(B, w - h)) / (2 * h),
              (F(B + h, w) - 2 * F(B, w) + F(B - h, w)) / h ** 2,
              (F(B + h, w + h) - F(B + h, w - h) - F(B - h, w + h) + F(B - h, w - h)) / (4 * h * h),
              (F(B, w + h) - 2 * F(B, w) + F(B, w - h)) / h ** 2]
        an = [V, VB, Vw, VBB, VBw, Vww]
        for i in range(6):
            e = abs(fd[i] - an[i]) / (1 + abs(an[i]))
            worst = max(worst, e)
            if e > 1e-4:
                print('MISMATCH', which, i, B, w, fd[i], an[i])
print('worst rel discrepancy', worst)
