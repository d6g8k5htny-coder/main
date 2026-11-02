#!/usr/bin/env python3
"""C104 cross-provider review: log10 of C104's explicit constants (21), (27), (35), (39), (40), (47), (48) at L = 24
(standard library only; all work in the log domain, so nothing overflows).  It shows which bounds in (10) and (48) are
numerically nonvacuous at the manuscript's side length.  Ranges: COEFF's band B x K = [0, 1] x [1/2, 2] (b_* = 1),
and the single points (b, k) = (0, 1/2), (0, 1) (b_* = 0, k_- = k_+ = k)."""
import json
import math

LN10 = math.log(10)


def l10(x):
    return math.log10(x)


def lsum(a, b):                 # log10(10^a + 10^b)
    m = max(a, b)
    return m + math.log10(1 + 10 ** (min(a, b) - m))


def rising(x, q):
    r = 1.0
    for i in range(q):
        r *= x + i
    return r


def lM(d, q, t):                # log10 of M_{d,q}(t) = pi^(d/2) (d/2)_q t^(-d/2-q)
    return (d / 2) * l10(math.pi) + l10(rising(d / 2, q)) - (d / 2 + q) * l10(t)


def consts(bstar, km, kp, L, Bmeas, Kmeas):
    a0, M0 = 1 / 112, 81.0
    V = 1 + bstar + 12 * kp
    ms = 3 * V
    Ds = 20 * (112 + 81 * 112 ** 2) * V + 20 * (1 + 81 * 112) ** 2
    gam = 1 / (4 * M0)
    lG = lambda n: -(n / 2) * l10(2 * math.pi * a0) + (ms ** 2 / (2 * M0)) / LN10
    Q = lambda n: n / (2 * a0) + (1 + ms) / a0 + (1 + ms ** 2) / a0 ** 2
    CK = 1536 * kp ** 2 + 1728 * kp ** 3 + 108 * kp + 11 / (9 * km)
    c0 = gam / 6
    lS0 = lsum(lM(9, 0, c0), lM(9, 3, c0)) - l10(2 * c0 ** 2)
    lS1 = lsum(l10(1 / (2 * c0 ** 2) + 1 / c0 ** 3) + lsum(lM(9, 0, c0), lM(9, 3, c0)),
               lsum(lM(9, 1, c0), lM(9, 4, c0)) - l10(2 * c0 ** 2))
    lNs = l10(math.pi) + lG(12) + l10(CK) + lS0
    lCN = l10(6 * math.pi * Ds) + lG(12) + l10(Q(12)) + l10(CK) + lS1
    lCz = l10(36 * kp ** 2 * Ds) + lG(3) + l10(Q(3)) + lsum(lM(3, 2, gam), lM(3, 3, gam))
    lzs = l10(18 * km ** 2 * 225 / 256) - 1.5 * l10(2 * math.pi * M0) - ((3 + ms) ** 2 / (2 * a0)) / LN10
    lCa = lsum(lCN - lzs, lNs + lCz - 2 * lzs)
    lPs = -4 * l10(2 * math.pi * a0)
    lCp = l10(20) + lPs + l10(4 / a0 + V ** 2 / (2 * a0 ** 2))
    lD = lambda L_: l10(5000) + 6 * l10(1 + L_) - (L_ * L_ / 2) / LN10
    out = {'log10_Delta_L': round(lD(L), 3), 'log10_CN_Delta': round(lCN + lD(L), 2),
           'log10_Cz_Delta': round(lCz + lD(L), 2), 'log10_zstar': round(lzs, 1), 'log10_Ca_Delta': round(lCa + lD(L), 1)}
    if Bmeas > 0 and Kmeas > 0:
        lCband = l10(16 * math.pi * Bmeas * Kmeas * km ** (-5 / 3)) + lsum(lPs + lCN, lNs + lCp)
        out['log10_Cband_Delta'] = round(lCband + lD(L), 2)
    Lq = L
    while lCa + lD(Lq) >= 0:
        Lq += 1
    out['first_L_with_Ca_Delta_below_1'] = Lq
    return out


res = {'COEFF_band_[0,1]x[1/2,2]': consts(1, 0.5, 2, 24, 1, 1.5),
       'point_b0_k0.5': consts(0, 0.5, 0.5, 24, 0, 0),
       'point_b0_k1': consts(0, 1, 1, 24, 0, 0)}
print(json.dumps(res, sort_keys=True))
