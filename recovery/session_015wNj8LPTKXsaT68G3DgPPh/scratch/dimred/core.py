"""Exact one-site jet law in d dimensions for K(x) = exp(-|x|^2/2) and the hard-direction factorization."""
import math, random, itertools
from fractions import Fraction as Fr
from math import erf, exp, sqrt, pi

def c_even(n):
    if n % 2: return Fr(0)
    v = Fr(1)
    for j in range(1, n, 2): v *= j
    return v * (-1) ** (n // 2)

def cov_point(al, be):
    """Cov(d^al f(0), d^be f(0)) for the unit Gaussian kernel, exact."""
    v = Fr((-1) ** sum(al))
    for a, b in zip(al, be): v *= c_even(a + b)
    return v

def jets(d, maxorder=3):
    out = []
    for order in range(maxorder + 1):
        for al in itertools.product(range(order + 1), repeat=d):
            if sum(al) == order: out.append(al)
    return out

def mat_inv(M):
    n = len(M); A = [row[:] + [Fr(int(i == j)) for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0); A[c], A[p] = A[p], A[c]
        pv = A[c][c]; A[c] = [x / pv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]; A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return [row[n:] for row in A]

def conditional(targets, obs, d):
    """Exact conditional covariance of targets given obs (all multi-indices at one point), and the regression matrix."""
    C = lambda a, b: cov_point(a, b)
    Soo = [[C(a, b) for b in obs] for a in obs]; Sto = [[C(a, b) for b in obs] for a in targets]
    Stt = [[C(a, b) for b in targets] for a in targets]
    inv = mat_inv(Soo)
    reg = [[sum(Sto[i][k] * inv[k][j] for k in range(len(obs))) for j in range(len(obs))] for i in range(len(targets))]
    cc = [[Stt[i][j] - sum(reg[i][k] * Sto[j][k] for k in range(len(obs))) for j in range(len(targets))] for i in range(len(targets))]
    return reg, cc

def lemma1(d):
    """Even jets (transverse Hessian) and odd jets (third derivatives except f_xxx) given the contact frame U_0."""
    e = lambda *idx: tuple(idx)
    zero = (0,) * d
    def unit(i, n=1):
        v = [0] * d; v[i] = n; return tuple(v)
    def add(*vs):
        return tuple(sum(v[i] for v in vs) for i in range(d))
    x = unit(0)
    # observations: f, grad f, f_xx, f_x y_i, f_xxx
    obs = [zero] + [unit(i) for i in range(d)] + [unit(0, 2)] + [add(x, unit(i)) for i in range(1, d)] + [unit(0, 3)]
    # even targets: transverse Hessian entries y_i y_j, i <= j
    even = [add(unit(i), unit(j)) for i in range(1, d) for j in range(i, d)]
    # odd targets: all third derivatives except f_xxx
    odd = [al for al in jets(d, 3) if sum(al) == 3 and al != unit(0, 3)]
    reg_e, cc_e = conditional(even, obs, d)
    reg_o, cc_o = conditional(odd, obs, d)
    return obs, even, odd, reg_e, cc_e, reg_o, cc_o

if __name__ == '__main__':
    for d in (2, 3, 4):
        obs, even, odd, reg_e, cc_e, reg_o, cc_o = lemma1(d)
        print('d =', d, 'obs', len(obs), 'even', len(even), 'odd', len(odd))
        # conditional mean of even jets when obs = (b, 0..., 0, 0.., 12k): only f = b contributes; print coefficient on b
        print(' even mean coeff on f (should be -1 for diagonal, 0 off):', [reg_e[i][0] for i in range(len(even))], ' on f_xxx:', [reg_e[i][-1] for i in range(len(even))])
        print(' even cond cov:', cc_e)
        print(' odd mean coeff on f_xxx:', [reg_o[i][-1] for i in range(len(odd))], ' on f:', [reg_o[i][0] for i in range(len(odd))])
        # odd conditional covariance: report diag and nonzero offdiag
        nz = [(odd[i], odd[j], cc_o[i][j]) for i in range(len(odd)) for j in range(i, len(odd)) if cc_o[i][j] != 0]
        print(' odd cond cov nonzero entries:', nz)
        # cross-covariance even/odd is identically zero by parity
        cross = [cov_point(a, b) for a in even for b in odd]
        print(' even/odd cross cov all zero:', all(c == 0 for c in cross))
