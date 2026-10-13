"""Floating-point controls for CL-C8-ELDER-FAILURE-COEFFICIENT-20261001-v1 (not part of the certificate).

1. E[2(B_-)^3] by composite Simpson (float) against the certified enclosure.
2. Consistency of the split J_fail = 3k^2 (E[2(B_-)^3] + E[2T] + E[V]) with a direct box enclosure of E[H] at k = 1/2
   (same engine, different integrand: the two enclosures must intersect).
3. The certified J_1, J_2 against the Gauss-Hermite and Monte Carlo values of [NUM] (RESULTS.json on this tree).
4. C_fail^{B,K} against the cap-route constants of Math-#215 (C_{B,K}(1/4096) = 402.5297, Gamma_2 = 307.51), and the
   d = 3 constant against C_{B,K}^(3)(1/4096) = 1205.7089, Gamma_3 = 540.97.
5. The d = 3 near coefficients at b = 0 against Math-#184's floating values.
6. Lemma T: floating lattice sums of the torus jet-covariance deviation at L = 10 against the certified delta."""
import os, sys, json, math
from decimal import Decimal
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certificate as C
from ia import dn, up

NUM = os.path.join(HERE, '..', 'c6_cluster_coefficients_numerics_20260930', 'RESULTS.json')


def e2b_float(k, N=200000):
    sa = math.sqrt(2 / k)
    def g(m):
        return (m ** 3 + 6 * m) * 0.5 * (1 + math.erf(m / 2)) + math.sqrt(2) * (m * m + 4) * math.exp(-m * m / 4) / math.sqrt(2 * math.pi)
    L = 12 * sa
    h = L / N
    f = lambda a: math.exp(-a * a / (2 * sa * sa)) / (sa * math.sqrt(2 * math.pi)) * g(a * a / 12)
    S = f(0) + f(L) + sum((4 if i % 2 else 2) * f(i * h) for i in range(1, N))
    return 4 * S * h / 3


def main():
    doc = json.load(open(os.path.join(HERE, 'RESULTS.json')))
    R = doc['results']
    print('1. E[2(B_-)^3]: float Simpson vs certified')
    for k in C.KS:
        e = e2b_float(k)
        iv = R['k=%g' % k]['parts']['E[2(B_-)^3]']
        print('   k=%g  float %.12f  certified [%s, %s]  inside: %s' % (k, e, iv[0], iv[1], Decimal(iv[0]) <= Decimal(e) <= Decimal(iv[1])))
    print('2. direct E[H] boxes at k = 1/2 (tol 1e-7) vs the split')
    (lo, hi), n, _ = C.run3(0.5, 'H', 1e-7, os.cpu_count() or 2)
    tl = C.tail3(0.5)
    s = 0.75
    J = R['k=0.5']['J_fail']
    direct = (s * lo, s * (hi + tl))
    print('   direct 3k^2 E[H] in [%.9f, %.9f] (%d leaves); split J_fail [%s, %s]; intersect: %s' %
          (direct[0], direct[1], n, J[0], J[1], Decimal(direct[0]) <= Decimal(J[1]) and Decimal(J[0]) <= Decimal(direct[1])))
    print('3. [NUM] floating values against the certified enclosures')
    num = json.load(open(NUM))['per_k']
    for k, key in ((0.5, '1/2'), (1.0, '1'), (2.0, '2')):
        r = R['k=%g' % k]
        for j in ('1', '2'):
            iv = [float(x) for x in r['J_%s' % j]]
            mid = 0.5 * (iv[0] + iv[1])
            v = num[key]
            line = '   k=%g J_%s certified [%.6f, %.6f]' % (k, j, iv[0], iv[1])
            for lab in ('gh40', 'gh60', 'mc'):
                x = v['J%s_%s' % (j, lab)]
                inside = iv[0] <= x <= iv[1]
                line += '  %s %.5f (%+.3f%%%s)' % (lab, x, 100 * (x - mid) / mid, ', inside' if inside else '')
            line += '  mc s.e. %.4f -> (mc - mid)/se = %+.2f' % (v['J%s_mc_se' % j], (v['J%s_mc' % j] - mid) / v['J%s_mc_se' % j])
            print(line)
    print('4. C_fail^{B,K} against the cap-route constants')
    Cf = [float(x) for x in R['C_fail']['C_fail^{B,K}']]
    for name, val in (('C_{B,K}(1/4096), Math-#215', 402.5297), ('Gamma_2 (floating), Math-#215', 307.51)):
        print('   %s = %g : ratio to C_fail in [%.4g, %.4g]' % (name, val, val / Cf[1], val / Cf[0]))
    D3 = R['d=3 (conditional on Math-#175 (L1) and Math-#184 Theorem 1)']
    Cf3 = [float(x) for x in D3['C_fail^(3),{B,K}']]
    for name, val in (('C_{B,K}^(3)(1/4096), Math-#215', 1205.7089), ('Gamma_3 (floating), Math-#215', 540.97)):
        print('   d = 3: %s = %g : ratio to C_fail^(3) in [%.4g, %.4g]' % (name, val, val / Cf3[1], val / Cf3[0]))
    print('6. Lemma T: direct floating lattice sums of the torus deviation at L = 10 (three frames) against delta')
    import torus as TOR
    dlt = TOR.delta()
    def He(j, t):
        h0, h1 = 1.0, t
        if j == 0:
            return h0
        for i in range(1, j):
            h0, h1 = h1, t * h1 - i * h0
        return h1
    worst = 0.0
    for th in (0.0, 0.3, math.pi / 4):
        c, s_ = math.cos(th), math.sin(th)
        for g in ((6, 0), (3, 3), (4, 2), (2, 2), (5, 1), (0, 0)):
            tot = 0.0
            theta = 0.0
            for n1 in range(-3, 4):
                for n2 in range(-3, 4):
                    if n1 == 0 and n2 == 0:
                        continue
                    y1, y2 = 10.0 * (c * n1 + s_ * n2), 10.0 * (-s_ * n1 + c * n2)
                    e = math.exp(-(y1 * y1 + y2 * y2) / 2)
                    tot += (-1) ** (g[0] + g[1]) * He(g[0], y1) * He(g[1], y2) * e
                    theta += e
            d0 = TOR._D(g)
            dev = abs((tot - d0 * theta) / (1.0 + theta))      # (d0 + tot)/(1 + theta) - d0 without cancellation
            worst = max(worst, dev)
    print('   max floating deviation %.3e  <=  delta %.3e: %s' % (worst, dlt, worst <= dlt))
    print('5. d = 3 near coefficients at b = 0 against Math-#184 (floating, from [NUM] GH60 times R_3(0))')
    hd = {0.5: 4.8558, 1.0: 5.705}
    for k, v in hd.items():
        iv = [float(x) for x in D3['k=%g, b=0' % k]['a_fail^(3)']]
        print('   k=%g a_fail^(3) certified [%.6f, %.6f]  Math-#184 %.4f  (%+.3f%%)' % (k, iv[0], iv[1], v, 100 * (v / (0.5 * (iv[0] + iv[1])) - 1)))


if __name__ == '__main__':
    main()
