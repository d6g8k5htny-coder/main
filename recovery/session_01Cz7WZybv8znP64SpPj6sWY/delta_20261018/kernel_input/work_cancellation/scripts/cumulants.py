"""Condition numbers (relative-perturbation amplification) of subtraction forms vs cancellation-free forms,
plus actual interval subtraction digits for kappa4, kappa6 from kernel enclosures. TEST_ONLY L values."""
import sys, json, os
from fractions import Fraction as F
import mpmath as mp
mp.mp.dps = 400
KDIR, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, KDIR)
from moment_kernel import certify, KernelHold
from verify_certificate import verify

def ref(Lq, dw=None, dh=0):
    L = mp.mpf(Lq.numerator) / Lq.denominator
    h = 2 * mp.pi / L * (1 + dh); al = (2 * mp.pi / L) ** 2 / 2
    K = int(mp.sqrt(mp.mpf(950) / al)) + 3
    xs = [(mp.mpf(0), mp.mpf(1))]
    for m in range(1, K + 1):
        w = 2 * mp.e ** (-al * m * m) * (1 + (dw[m] if dw and m in dw else 0))
        xs.append(((h * m) ** 2, w))
    S = [sum(w * x ** k for x, w in xs) for k in range(4)]
    z = S[0]; a, b, c = S[1] / z, S[2] / z, S[3] / z
    D = mp.mpf(0); B = mp.mpf(0)
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            (xi, wi), (xj, wj) = xs[i], xs[j]
            B += wi * wj * (xi - xj) ** 2; D += wi * wj * xi * xj * (xi - xj) ** 2
    return dict(a=a, b=b, c=c, z=z, beta=B / z ** 2, D=D / z ** 2, nu3=D / z ** 2 / a, K=K)

def kappa_cf(Lq, key):
    """sum over m>=1 and h of |d log f / d log param| by central differences (cancellation-free form)."""
    r0 = ref(Lq); eps = mp.mpf(10) ** -60; tot = mp.mpf(0)
    for m in range(1, r0["K"] + 1):
        up = ref(Lq, {m: eps})[key]; dn = ref(Lq, {m: -eps})[key]
        tot += abs((up - dn) / (2 * eps) / r0[key])
    up = ref(Lq, dh=eps)[key]; dn = ref(Lq, dh=-eps)[key]
    tot += abs((up - dn) / (2 * eps) / r0[key])
    return tot

def dig(lo, hi, t):
    w = mp.mpf(hi.numerator) / hi.denominator - mp.mpf(lo.numerator) / lo.denominator
    return float(-mp.log10(w / abs(t)))

cases = [("1/2", 2), ("1/1", 3), ("2/1", 5), ("3/1", 8), ("4/1", 10), ("6/1", 16), ("8/1", 20), ("12/1", 32)]
print("L    | log10 kappa_sub: beta  nu3   k4    k6   | log10 kappa_cf: beta nu3 | k4/b      k6/c      | max-setting digits a b c | k4_sub k6_sub (digits, lo sign)")
for Ls, M in cases:
    Lq = F(Ls); r = ref(Lq)
    a, b, c = r["a"], r["b"], r["c"]
    k4 = b - 3 * a * a; k6 = c - 15 * a * b + 30 * a ** 3
    ks = dict(beta=(b + 2 * a * a) / abs(r["beta"]), nu3=(c + 3 * b * b / a) / abs(r["nu3"]),
              k4=(b + 6 * a * a) / abs(k4), k6=(c + 30 * a * b + 90 * a ** 3) / abs(k6))
    kcb, kcn = kappa_cf(Lq, "beta"), kappa_cf(Lq, "nu3")
    cert = certify(dict(L_lo=Ls, L_hi=Ls, cutoff=M, bits=128, pi_terms=32, exp_pairs=24, input_role="TEST_ONLY"))
    verify(json.loads(json.dumps(cert)))
    A, Bq, C = [tuple(F(x) for x in cert["bounds"][k]) for k in ("mu2", "mu4", "mu6")]
    k4i = (Bq[0] - 3 * A[1] ** 2, Bq[1] - 3 * A[0] ** 2)
    k6i = (C[0] - 15 * A[1] * Bq[1] + 30 * A[0] ** 3, C[1] - 15 * A[0] * Bq[0] + 30 * A[1] ** 3)
    print("%-4s | %5.2f %6.1f %5.2f %5.2f | %4.2f %4.2f | %s %s | %5.1f %5.1f %5.1f | %6.1f %s  %6.1f %s" % (
        Ls, *[float(mp.log10(ks[k])) for k in ("beta", "nu3", "k4", "k6")], float(mp.log10(kcb)), float(mp.log10(kcn)),
        mp.nstr(k4 / b, 4), mp.nstr(k6 / c, 4), dig(*A, a), dig(*Bq, b), dig(*C, c),
        dig(*k4i, k4), "+" if k4i[0] > 0 else ("-" if k4i[1] < 0 else "0"), dig(*k6i, k6), "+" if k6i[0] > 0 else ("-" if k6i[1] < 0 else "0")))
