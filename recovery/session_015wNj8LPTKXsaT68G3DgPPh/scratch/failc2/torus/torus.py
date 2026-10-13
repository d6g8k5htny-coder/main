"""Lemma T (torus transfer) for CL-C8-ELDER-FAILURE-COEFFICIENT-20261001: the planar coefficients of the torus model of
[LP] / [S] (side L >= 10, any frame) against the reference-kernel ones.

One-site jet covariances.  Cov(d^a f, d^b f) = (-1)^|b| d^(a+b) K(0).  For the torus kernel
K_L(z) = sum_n G(z + Ln) / Theta_L, G = exp(-|z|^2/2), Theta_L = sum_n exp(-L^2 |n|^2/2) ([LP] section 2), in a rotated
frame d^g K_L(0) = sum_n (d^g G)(R^T L n) / Theta_L with |R^T L n| = L|n|, and |d^g G(y)| = |He_g1(y1) He_g2(y2)| G(y)
<= Hh_g1(|y|) Hh_g2(|y|) G(y), Hh_j(t) = sum_k j!/(k!(j-2k)! 2^k) t^(j-2k).  Grouping n != 0 by |n|_inf = j (8j points,
|n| >= j) and using that t -> Hh_a(t) Hh_c(t) e^(-t^2/2) decreases for t^2 > a + c, every entry of order |a| + |b| <= 6
differs from the reference by at most delta(10) for every L >= 10 and every frame.

Propagation.  Exact interval regressions with every entry widened by delta give: the odd conditional law
N(12k v, S_c) of (f_xxz, f_xzz, f_zzz) given (f_x, f_z, f_xxx) = (0, 0, 12k) (reference: N(0, diag(2, 2, 6))); the even
conditional law N(c1 b, s^2) of f_zz given (f, f_xx, f_xz) = (b, 0, 0) (reference N(-b, 2)); and the factors of
pi_0 = p_(f,V_u)(b, 0) p_G(0) phi_tau(12k) ([LP] section 15).  Parity (K_L even) keeps odd and even jets uncorrelated.

Expectations.  With S_c = S0^(1/2) (I + E) S0^(1/2), ||E|| <= eps, mu~ = S0^(-1/2) mu:
    1 + chi^2(N(mu, S_c) || N(0, S0)) = prod (1 - e_i^2)^(-1/2) exp(sum mu~_i^2/(1 - e_i)) <= (1 - eps^2)^(-3/2) exp(|mu~|^2/(1 - eps)),
so chi^2 <= 1.01 m e^m + 2 eps^2 (m = |mu~|^2/(1 - eps), eps^2 <= 0.1), and |E_L[I] - E_ref[I]| <= (E_ref[I^2] chi^2)^(1/2)
(Cauchy-Schwarz), with I = 3k^2 H <= 12 Pi_4 for k <= 2 (Pi_4 the tail polynomial of certificate.tail4)."""
import math
from fractions import Fraction as Fr
from functools import reduce
from ia import (dn, up, add, sub, neg, mul, div, scal, divc, sqr, pw, isqrt, exp_neg, from_frac, absmax, Phi, SQRT2PI)

L_MIN = 10.0
ORDER = 6


def _Hh(j):
    return {j - 2 * k: Fr(math.factorial(j), math.factorial(k) * math.factorial(j - 2 * k) * 2 ** k) for k in range(j // 2 + 1)}


def _poly(coef, t):
    acc = (0.0, 0.0)
    for p, c in coef.items():
        acc = add(acc, mul(from_frac(c), pw(t, p)))
    return acc


def _M(t):
    """max over a + c <= ORDER of Hh_a(t) Hh_c(t) (upper end), t >= 1."""
    best = 0.0
    for m in range(ORDER + 1):
        for a in range(m + 1):
            best = max(best, mul(_poly(_Hh(a), t), _poly(_Hh(m - a), t))[1])
    return best


def delta(L=L_MIN, J=6):
    """Bound on |c_L - c_ref| for every one-site covariance entry of order <= ORDER, every side >= L and every frame."""
    S = 0.0
    term = 0.0
    for j in range(1, J + 1):
        t = (L * j, L * j)
        term = up(8.0 * j * up(_M(t) * exp_neg(divc(sqr(t), 2.0))[1]))
        S = up(S + term)
    S = up(S + 2.0 * term)                  # shells j > J: consecutive bounds decrease by far more than 1/2
    theta_minus_1 = S                       # sum_{n != 0} e^(-L^2 |n|^2/2) <= S since Hh products are >= 1
    return up(S + 15.0 * theta_minus_1)     # |d^g G(0)| <= 15 for |g| <= 6; |1/Theta - 1| <= Theta - 1


def _dfact(n):
    r = 1
    while n > 1:
        r, n = r * n, n - 2
    return r


def _D(g):
    if g[0] % 2 or g[1] % 2:
        return 0
    return (-1) ** ((g[0] + g[1]) // 2) * _dfact(g[0] - 1) * _dfact(g[1] - 1)


def cov_ref(a, b):
    """Reference covariance of the jets d^a f and d^b f at one site (multi-indices (x-order, z-order))."""
    return (-1) ** (b[0] + b[1]) * _D((a[0] + b[0], a[1] + b[1]))


def _icov(rows, cols, d):
    return [[(dn(cov_ref(a, b) - d), up(cov_ref(a, b) + d)) for b in cols] for a in rows]


def _det2(M):
    return sub(mul(M[0][0], M[1][1]), mul(M[0][1], M[1][0]))


def _det3(M):
    return sub(add(mul(M[0][0], sub(mul(M[1][1], M[2][2]), mul(M[1][2], M[2][1]))),
                   mul(M[0][2], sub(mul(M[1][0], M[2][1]), mul(M[1][1], M[2][0])))),
               mul(M[0][1], sub(mul(M[1][0], M[2][2]), mul(M[1][2], M[2][0]))))


def _inv3(M):
    d = _det3(M)
    adj = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            r = [x for x in range(3) if x != i]
            c = [x for x in range(3) if x != j]
            m = sub(mul(M[r[0]][c[0]], M[r[1]][c[1]]), mul(M[r[0]][c[1]], M[r[1]][c[0]]))
            adj[j][i] = m if (i + j) % 2 == 0 else neg(m)
    return [[div(adj[i][j], d) for j in range(3)] for i in range(3)]


def _inv2(M):
    d = _det2(M)
    return [[div(M[1][1], d), div(neg(M[0][1]), d)], [div(neg(M[1][0]), d), div(M[0][0], d)]]


def _mm(A, B):
    return [[reduce(add, [mul(A[i][t], B[t][j]) for t in range(len(B))]) for j in range(len(B[0]))] for i in range(len(A))]


def _T(A):
    return [list(r) for r in zip(*A)]


def _expiv(x):
    """exp over a small interval x (|x| < 1): exp(-y) enclosures from ia."""
    lo = exp_neg((-x[0], -x[0])) if x[0] <= 0 else None
    # exp(x) = 1/exp(-x); use exp_neg on -x for x <= 0 and on x for the reciprocal otherwise
    def e(t):
        return exp_neg((-t, -t)) if t <= 0 else div((1.0, 1.0), exp_neg((t, t)))
    return (e(x[0])[0], e(x[1])[1])


def transfer(E_I2_upper):
    """Lemma T bounds.  E_I2_upper: an upper bound of E_ref[I^2], I = 3k^2 H, uniform over k in [1/2, 2].
    Returns a dict of bounds (all floats, upper bounds unless stated)."""
    d = delta()
    # ---- odd block: Y = (f_x, f_z, f_xxx) pinned at (0, 0, 12k); X = (f_xxz, f_xzz, f_zzz)
    Y = [(1, 0), (0, 1), (3, 0)]
    X = [(2, 1), (1, 2), (0, 3)]
    SYY, SXY, SXX = _icov(Y, Y, d), _icov(X, Y, d), _icov(X, X, d)
    Bc = _mm(SXY, _inv3(SYY))
    BS = _mm(Bc, _T(SXY))
    Sc = [[sub(SXX[i][j], BS[i][j]) for j in range(3)] for i in range(3)]
    S0 = (2.0, 2.0, 6.0)
    eps2 = 0.0
    for i in range(3):
        for j in range(3):
            dev = sub(Sc[i][j], ((S0[i] if i == j else 0.0),) * 2)
            e = up(absmax(dev) / math.sqrt(S0[i] * S0[j]) * (1.0 + 1e-15))
            eps2 = up(eps2 + up(e * e))
    eps = up(math.sqrt(eps2) * (1.0 + 1e-15))
    mu2 = 0.0
    for i in range(3):
        m_i = up(24.0 * absmax(Bc[i][2]))        # mean = 12k Bc[:, 2], k <= 2
        mu2 = up(mu2 + up(m_i * m_i / S0[i]))
    if not (eps2 <= 0.1 and mu2 < 1.0):
        raise RuntimeError('torus transfer: perturbation too large')
    m = up(mu2 / (1.0 - eps) * (1.0 + 1e-15))
    chi2 = up(up(1.01 * up(m * math.exp(m) * (1.0 + 1e-12))) + up(2.0 * eps2))
    dJ = up(math.sqrt(up(E_I2_upper * chi2)) * (1.0 + 1e-15))
    # ---- even block: W = (f, f_xx, f_xz) pinned at (b, 0, 0); A = f_zz
    W = [(0, 0), (2, 0), (1, 1)]
    A = [(0, 2)]
    SWW, SAW, SAA = _icov(W, W, d), _icov(A, W, d), _icov(A, A, d)
    Wi = _inv3(SWW)
    cA = _mm(SAW, Wi)[0]                           # regression row: mean of A = cA[0] b
    sA2 = sub(SAA[0][0], _mm(_mm(SAW, Wi), _T(SAW))[0][0])
    # p_b(0)^L / p_b(0)^ref = (sqrt2/s_L) exp(-b^2 (cA0^2/(2 s_L^2) - 1/4)), b in [0, 1]
    x = sub(div(sqr(cA[0]), scal(2.0, sA2)), (0.25, 0.25))
    ax = absmax(x)
    rp = mul(div(isqrt((2.0, 2.0)), isqrt(sA2)), _expiv((-ax, ax)))
    # pi_0 factors: p_W(b,0,0) ratio, p_G(0) ratio, phi_tau(12k) ratio
    det_ref_W = (2.0, 2.0)                         # det [[1,-1,0],[-1,3,0],[0,0,1]]
    y = sub(Wi[0][0], (1.5, 1.5))
    ay = absmax(y)
    rW = mul(isqrt(div(det_ref_W, _det3(SWW))), _expiv((-0.5 * ay, 0.5 * ay)))
    G = [(1, 0), (0, 1)]
    SGG = _icov(G, G, d)
    rG = isqrt(div((1.0, 1.0), _det2(SGG)))
    T3 = [(3, 0)]
    S3G, S33 = _icov(T3, G, d), _icov(T3, T3, d)
    tau2 = sub(S33[0][0], _mm(_mm(S3G, _inv2(SGG)), _T(S3G))[0][0])
    z = sub(div((1.0, 1.0), tau2), div((1.0, 1.0), (6.0, 6.0)))
    az = up(288.0 * absmax(z))                     # 72 k^2 |1/tau_L^2 - 1/6|, k <= 2
    rT = mul(isqrt(div((6.0, 6.0), tau2)), _expiv((-az, az)))
    rho = mul(mul(mul(rW, rG), rT), rp)            # (pi_0 p_b(0))^L / (pi_0 p_b(0))^ref, uniform in (b, k)
    return {'delta': d, 'eps': eps, 'mu2': mu2, 'chi2': chi2, 'dJ': dJ, 'mean_coeff_A': cA[0], 'var_A': sA2,
            'p_ratio': rp, 'pi0p_ratio': rho, 'tau2': tau2, 'S_c': Sc}


if __name__ == '__main__':
    import json
    t = transfer(1.0e8)
    for kk in ('delta', 'eps', 'mu2', 'chi2', 'dJ', 'mean_coeff_A', 'var_A', 'p_ratio', 'pi0p_ratio', 'tau2'):
        print(kk, t[kk])


def abs_moment(s2, n):
    """E|X|^n for X ~ N(0, s2) with s2 an exact float: n even (n-1)!! s2^(n/2); n odd 2^(n/2) Gamma((n+1)/2)/sqrt(pi) s^n,
    i.e. (n-1)!! s^n sqrt(2/pi) (enclosed)."""
    if n % 2 == 0:
        return pw(scal(float(_dfact(n - 1)), (1.0, 1.0)), 1) if n == 0 else mul((float(_dfact(n - 1)),) * 2, pw((s2, s2), n // 2))
    from ia import PI
    s = isqrt((s2, s2))
    return mul(mul((float(_dfact(n - 1)),) * 2, pw(s, n)), isqrt(div((2.0, 2.0), PI)))


def E_I2_upper():
    """E_ref[I^2] <= 144 E[Pi_4^2], I = 3k^2 H <= 12 Pi_4 for k in [1/2, 2], (a, beta, c) ~ N(0, diag(2, 2, 6)),
    Pi_4 = 68|beta|^3 + c6 a^6 + 144 c^2 + 9 a^2 beta^2, c6 = 68/216 + 576/5184."""
    c6 = add(divc((68.0, 68.0), 216.0), divc((576.0, 576.0), 5184.0))
    terms = [((68.0, 68.0), (0, 3, 0)), (c6, (6, 0, 0)), ((144.0, 144.0), (0, 0, 2)), ((9.0, 9.0), (2, 2, 0))]
    var = (2.0, 2.0, 6.0)
    tot = (0.0, 0.0)
    for c1, e1 in terms:
        for c2, e2 in terms:
            v = mul(c1, c2)
            for j in range(3):
                v = mul(v, abs_moment(var[j], e1[j] + e2[j]))
            tot = add(tot, v)
    return up(144.0 * tot[1])
