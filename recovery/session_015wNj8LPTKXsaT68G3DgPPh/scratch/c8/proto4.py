import math
from math import erf, sqrt, exp
from proto import conditional, mc, chol, z0
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))

def block_prob(mu_p, mu_q, Lpp, Lqp, Lqq, lp, hp, lq, hq, npieces):
    """Lower bound (exact in the limit) of P(H_p in [lp,hp], H_q in [lq,hq]) for H_p = mu_p + Lpp z_a, H_q = mu_q + Lqp z_a + Lqq z_b."""
    A = (lp - mu_p) / Lpp; B = (hp - mu_p) / Lpp
    if B <= A: return 0.0
    tot = 0.0
    for i in range(npieces):
        s0 = A + (B - A) * i / npieces; s1 = A + (B - A) * (i + 1) / npieces
        # bracket over z_a in [s0, s1]: u(z) = (hq - mu_q - Lqp z)/Lqq is monotone in z; take the worst case (min of Phi(u)-Phi(l))
        vals = []
        for s in (s0, s1):
            u = (hq - mu_q - Lqp * s) / Lqq; l = (lq - mu_q - Lqp * s) / Lqq
            vals.append(Phi(u) - Phi(l))
        # Phi(u)-Phi(l) with both u,l affine in z: not monotone in general, but for a lower bound use min over endpoints minus a crude modulus (float prototype only)
        g = min(vals)
        tot += g * (Phi(s1) - Phi(s0))
    return tot

def cell_scheme(mu, C, nM, nS, cxx, cxy, span, npieces):
    sd = [sqrt(C[i][i]) for i in range(6)]
    # blocks: xx (0,3), xy (1,4), yy (2,5)
    def blk(p, q):
        Lpp = sqrt(C[p][p]); Lqp = C[q][p] / Lpp; Lqq = sqrt(C[q][q] - Lqp ** 2)
        return Lpp, Lqp, Lqq
    Lxx = blk(0, 3); Lxy = blk(1, 4); Lyy = blk(2, 5)
    lo0, hi0 = mu[0] - cxx * sd[0], mu[0] + cxx * sd[0]; lo3, hi3 = mu[3] - cxx * sd[3], mu[3] + cxx * sd[3]
    lo1, hi1 = -cxy * sd[1], cxy * sd[1]; lo4, hi4 = -cxy * sd[4], cxy * sd[4]
    if hi0 >= 0 or lo3 <= 0: return 0.0
    Pxx = block_prob(mu[0], mu[3], *Lxx, lo0, hi0, lo3, hi3, npieces)
    Pxy = block_prob(mu[1], mu[4], *Lxy, lo1, hi1, lo4, hi4, npieces)
    yminM = mu[2] - span * sd[2]; yminS = mu[5] - span * sd[5]
    eM = [yminM + (0 - yminM) * i / nM for i in range(nM + 1)]
    eS = [yminS + (0 - yminS) * i / nS for i in range(nS + 1)]
    tot = 0.0
    xymax = max(abs(lo1), abs(hi1))
    for i in range(nM):
        detM = hi0 * eM[i + 1] - xymax ** 2
        if detM <= 0: continue
        for j in range(nS):
            detS = lo3 * (-eS[j + 1])
            if detS <= 0: continue
            Pyy = block_prob(mu[2], mu[5], *Lyy, eM[i], eM[i + 1], eS[j], eS[j + 1], npieces)
            tot += detM * detS * Pxx * Pxy * Pyy
    return tot

if __name__ == '__main__':
    for r in (0.125, 0.3, 0.5):
        for b in (0.0, 1.0):
            for k in (0.5, 2.0):
                mu, C = conditional(r, b, k)
                m, se = mc(mu, C, 100000, 1)
                for (nM, nS, cxx, cxy, span, npc) in ((16, 16, 3.5, 3.0, 5.0, 4), (24, 24, 3.5, 3.0, 5.0, 8), (40, 40, 4.0, 3.5, 5.5, 8)):
                    tot = cell_scheme(mu, C, nM, nS, cxx, cxy, span, npc)
                    print('r=%.3f b=%.0f k=%.1f grid=%dx%d c=(%.1f,%.1f,%.1f) pieces=%d  MC %.4f (%.4f)  bound %.4f  ratio %.3f' % (r, b, k, nM, nS, cxx, cxy, span, npc, m / r ** 2, se / r ** 2, tot / r ** 2, tot / m))
