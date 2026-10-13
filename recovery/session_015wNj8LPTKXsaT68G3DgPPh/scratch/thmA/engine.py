"""Exact covariance engine: preconditioned functionals of a planar Gaussian field at axis points {-r/2, 0, r/2};
covariances as power series in r from the Taylor coefficients kappa_gamma = d^gamma K(0) of the kernel."""
from fractions import Fraction as Fr
import math

H = Fr(1, 2)
# a term: (coef, power of r, x-order a, y-order b, position multiplier x0 in {-1/2, 0, 1/2}) -> coef * r^power * d_x^a d_y^b f(x0 r, 0)
def term(c, p, a, b, x0):
    return (Fr(c), p, a, b, Fr(x0))

M_, S_, O_ = -H, H, Fr(0)

def functionals():
    pins = {
        'p1': [term(H, 0, 0, 0, M_), term(H, 0, 0, 0, S_)],
        'p2': [term(-1, -1, 0, 0, M_), term(1, -1, 0, 0, S_)],
        'p3p': [term(6, -2, 1, 0, M_), term(6, -2, 1, 0, S_), term(12, -3, 0, 0, M_), term(-12, -3, 0, 0, S_)],
        'p4': [term(-1, -1, 1, 0, M_), term(1, -1, 1, 0, S_)],
        'q1': [term(H, 0, 0, 1, M_), term(H, 0, 0, 1, S_)],
        'q2': [term(-1, -1, 0, 1, M_), term(1, -1, 0, 1, S_)],
    }
    targets = {
        'w': [term(1, 0, 0, 2, M_)],
        'v': [term(1, -1, 1, 1, M_), term(1, -2, 0, 1, M_), term(-1, -2, 0, 1, S_)],
        'om': [term(1, -1, 0, 2, S_), term(-1, -1, 0, 2, M_)],
        'Y': [term(1, 0, 0, 3, O_)],
        'T': [term(1, -2, 2, 0, M_), term(4, -3, 1, 0, M_), term(2, -3, 1, 0, S_), term(6, -4, 0, 0, M_), term(-6, -4, 0, 0, S_)],
        'tau': [term(1, -3, 2, 0, S_), term(-1, -3, 2, 0, M_), term(-6, -4, 1, 0, M_), term(-6, -4, 1, 0, S_), term(-12, -5, 0, 0, M_), term(12, -5, 0, 0, S_)],
        'nu': [term(1, -2, 1, 1, M_), term(1, -2, 1, 1, S_), term(2, -3, 0, 1, M_), term(-2, -3, 0, 1, S_)],
        'X40': [term(1, 0, 4, 0, O_)], 'X31': [term(1, 0, 3, 1, O_)], 'X22': [term(1, 0, 2, 2, O_)],
        'X13': [term(1, 0, 1, 3, O_)], 'X04': [term(1, 0, 0, 4, O_)],
    }
    return pins, targets

PIN_NAMES = ('p1', 'p2', 'p3p', 'p4', 'q1', 'q2')
TARGET_NAMES = ('w', 'v', 'om', 'Y', 'T', 'tau', 'nu', 'X40', 'X31', 'X22', 'X13', 'X04')

def weight(F):
    ws = {t[2] - t[1] for t in F}
    assert len(ws) == 1, F
    return ws.pop()

def pair_coeffs(F, G, NX):
    """pi[(gx, gy)] rational with Cov_K(F, G) = sum_gamma kappa_gamma * pi_gamma * r^(gx - wF - wG)."""
    wF, wG = weight(F), weight(G)
    out = {}
    for (cf, pf, af, bf, xf) in F:
        for (cg, pg, ag, bg, xg) in G:
            gy = bf + bg
            sgn = (-1) ** (ag + bg)
            d = xf - xg
            for gx in range(af + ag, NX + 1):
                m = gx - af - ag
                val = cf * cg * sgn * d ** m / math.factorial(m)
                if val != 0:
                    out[(gx, gy)] = out.get((gx, gy), Fr(0)) + val
    # regularity: every power must be >= 0
    for (gx, gy), v in out.items():
        if v != 0 and gx - wF - wG < 0:
            raise AssertionError('negative power %s %s' % ((gx, gy), v))
    return {k: v for k, v in out.items() if v != 0}, wF + wG

def hcoef(n):
    if n % 2:
        return 0
    return (-1) ** (n // 2) * math.prod(range(1, n, 2))

def kappa_ref(gx, gy):
    return hcoef(gx) * hcoef(gy)

def cov_series(F, G, NX):
    """Reference-kernel covariance as {power: Fraction} truncated at x-order NX."""
    pi, w = pair_coeffs(F, G, NX)
    ser = {}
    for (gx, gy), v in pi.items():
        k = kappa_ref(gx, gy)
        if k:
            ser[gx - w] = ser.get(gx - w, Fr(0)) + k * v
    return {p: c for p, c in ser.items() if c != 0}

if __name__ == '__main__':
    pins, targets = functionals()
    allf = dict(pins); allf.update(targets)
    for n in list(PIN_NAMES) + list(TARGET_NAMES):
        print(n, 'weight', weight(allf[n]))
    s = cov_series(targets['w'], targets['w'], 30)
    print('Var w', {p: s[p] for p in sorted(s)[:4]})
    s = cov_series(pins['p3p'], pins['p3p'], 40)
    print('Var p3p', {p: s[p] for p in sorted(s)[:4]})
    s = cov_series(targets['T'], targets['T'], 40)
    print('Var T', {p: s[p] for p in sorted(s)[:4]}, '(105/144 =', 105/144, ')')
    s = cov_series(targets['v'], targets['v'], 40)
    print('Var v', {p: s[p] for p in sorted(s)[:4]})
