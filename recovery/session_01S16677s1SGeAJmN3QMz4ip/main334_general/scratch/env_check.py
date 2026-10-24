import math, sys
sys.path.insert(0, '.')
from check import H_SG, log_grid, I_SG, amp_lognormal

def I_F(p, blim, nb, nk, alo, ahi, na):
    bs = [-blim + 2*blim * i / nb for i in range(nb + 1)]
    hb = 2*blim / nb
    ks, hk = log_grid(1e-8, 40.0, nk)
    As, ha = log_grid(alo, ahi, na)
    s = 0.0
    for i, b in enumerate(bs):
        wb = 0.5 if i in (0, nb) else 1.0
        for j, k in enumerate(ks):
            wk = 0.5 if j in (0, nk) else 1.0
            inner = 0.0
            for l, a in enumerate(As):
                wa = 0.5 if l in (0, na) else 1.0
                pa = p(a)
                if pa > 0:
                    inner += wa * a ** (-2.0) * H_SG(b / a, k / a) * pa * a
            s += wb * wk * k ** (-2.0 / 3.0) * inner * ha * k
    return s * hb * hk

def loguniform(lo, hi):
    Z = math.log(hi / lo)
    def p(a):
        return 1.0 / (a * Z) if lo <= a <= hi else 0.0
    m = (lo ** (-2/3) - hi ** (-2/3)) / ((2/3) * Z)
    return p, m, "log-uniform on [%g,%g]" % (lo, hi)

isg = I_SG(nb=600, nk=400)
print("I_SG = %.6f" % isg)
for p, m, name in [loguniform(0.3, 3.0), loguniform(0.1, 10.0), amp_lognormal(0.5)]:
    iF = I_F(p, blim=40.0, nb=1600, nk=160, alo=0.02, ahi=50.0, na=160)
    print("%s: m=%.6f  I_F(direct 3d)=%.6f  m*I_SG=%.6f  ratio=%.5f" % (name, m, iF, m*isg, iF/(m*isg)))
