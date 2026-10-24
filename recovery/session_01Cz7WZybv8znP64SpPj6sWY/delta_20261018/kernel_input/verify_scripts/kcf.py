import mpmath as mp
from fractions import Fraction as F
mp.mp.dps = 400
def ref(Lq, dw=None, dh=0):
    L = mp.mpf(Lq.numerator) / Lq.denominator
    h = 2 * mp.pi / L * (1 + dh); al = (2 * mp.pi / L) ** 2 / 2
    K = int(mp.sqrt(mp.mpf(950) / al)) + 3
    xs = [(mp.mpf(0), mp.mpf(1))]
    for m in range(1, K + 1):
        w = 2 * mp.e ** (-al * m * m) * (1 + (dw[m] if dw and m in dw else 0))
        xs.append(((h * m) ** 2, w))
    S = [sum(w * x ** k for x, w in xs) for k in range(4)]
    z = S[0]; a = S[1] / z
    D = mp.mpf(0); B = mp.mpf(0)
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            (xi, wi), (xj, wj) = xs[i], xs[j]
            B += wi * wj * (xi - xj) ** 2; D += wi * wj * xi * xj * (xi - xj) ** 2
    return dict(beta=B / z ** 2, D=D / z ** 2, nu3=D / z ** 2 / a, K=K)
def kappa_cf(Lq, key):
    r0 = ref(Lq); eps = mp.mpf(10) ** -60; tot = mp.mpf(0)
    for m in range(1, r0["K"] + 1):
        up = ref(Lq, {m: eps})[key]; dn = ref(Lq, {m: -eps})[key]
        tot += abs((up - dn) / (2 * eps) / r0[key])
    up = ref(Lq, dh=eps)[key]; dn = ref(Lq, dh=-eps)[key]
    tot += abs((up - dn) / (2 * eps) / r0[key])
    return tot
for Ls in ("1/2","1","2","3","4","6","8","12"):
    q=F(Ls)
    print(Ls, " ".join("%s=%.4f" % (k, float(mp.log10(kappa_cf(q,k)))) for k in ("beta","nu3","D")))
