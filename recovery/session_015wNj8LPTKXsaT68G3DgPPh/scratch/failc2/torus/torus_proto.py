"""Prototype of the torus transfer bound (Lemma T)."""
import math
from fractions import Fraction as Fr
from ia import *

def Hhat(j):
    """Coefficients of Hhat_j(t) = sum_k j!/(k!(j-2k)! 2^k) t^(j-2k) >= |He_j(t)| (|t| = t >= 0)."""
    return {j - 2 * k: Fr(math.factorial(j), math.factorial(k) * math.factorial(j - 2 * k) * 2 ** k) for k in range(j // 2 + 1)}

def poly_iv(coef, t):
    acc = (0.0, 0.0)
    for p, c in coef.items():
        acc = add(acc, mul(from_frac(c), pw(t, p)))
    return acc

def M_m(m, t):
    """max over a + c = m of Hhat_a(t) Hhat_c(t) (upper bound), t >= 0 interval."""
    best = (0.0, 0.0)
    for a in range(m + 1):
        v = mul(poly_iv(Hhat(a), t), poly_iv(Hhat(m - a), t))
        best = (max(best[0], v[0]), max(best[1], v[1]))
    return best

def delta_bound(L=10.0, mmax=6, J=6):
    """Upper bound, valid for every side L' >= L and every frame, of |d^gamma K_L'(0) - d^gamma G(0)| for |gamma| <= mmax:
    |d^g G(0)| (Theta - 1) + sum_{n != 0} sup_{|y| = L'|n|} |He_g1(y1) He_g2(y2)| e^(-|y|^2/2); the shell |n|_inf = j has 8j points
    with |n| >= j, and t -> M_m(t) e^(-t^2/2) decreases for t >= sqrt(m) + 1 (t M'/M <= m), so each term is <= its value at L j;
    the shells j > J are bounded by the geometric tail of 8j M(Lj) e^(-L^2 j^2/2)."""
    S = (0.0, 0.0)
    for j in range(1, J + 1):
        t = (L * j, L * j)
        term = mul(scal(8.0 * j, M_m(mmax, t)), exp_neg(divc(sqr(t), 2.0)))
        S = add(S, term)
    # tail j > J: ratio of consecutive bounds is tiny; bound by twice the J-th term
    S = add(S, scal(2.0, term))
    theta_minus_1 = up(2.0 * S[1])          # sum_{n != 0} e^(-L^2|n|^2/2) <= S (M >= 1)
    maxD = 15.0                              # |d^gamma G(0)| <= 15 for |gamma| <= 6
    return up(S[1] + maxD * theta_minus_1)

for L in (10.0, 12.0, 24.0):
    print('L', L, 'delta <=', delta_bound(L))
