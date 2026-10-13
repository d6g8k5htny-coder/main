"""Independent check of kappa_r, kappa_ell, kappa_t and midpoint drift on random backward-heat
polynomials H = sum_{n=2}^{7} c_n v_n(x,tau) (exact solutions of H_t = -H_xx), using Richardson
extrapolation in tau to isolate the O(tau) coefficient."""
from mpmath import mp, mpf, sqrt, findroot, quad, factorial, binomial
import random
mp.dps = 60
def v(n, x, tau):
    return sum(factorial(n) / (factorial(k) * factorial(n - 2 * k)) * (-tau) ** k * x ** (n - 2 * k) for k in range(n // 2 + 1))
random.seed(5)
for trial in range(4):
    c = {n: mpf(random.uniform(-1, 1)) for n in range(2, 8)}
    if abs(c[2]) < 0.2: c[2] = mpf('0.7')
    A, B, C = c[2], c[3], c[4]
    H = lambda x, tau: sum(c[n] * v(n, x, tau) for n in c)
    def meas(tau):
        zp = findroot(lambda x: H(x, tau), 2 * B / A * tau + sqrt(2 * tau))
        zm = findroot(lambda x: H(x, tau), 2 * B / A * tau - sqrt(2 * tau))
        r = zp - zm
        ell = abs(quad(lambda x: H(x, tau), [zm, zp]))
        return (r * r / (8 * tau) - 1) / tau, (ell / (abs(A) * r ** 3 / 6) - 1) / tau, \
               (ell / (8 * sqrt(2) / 3 * abs(A) * tau ** 1.5) - 1) / tau, ((zp + zm) / 2) / tau
    t1, t2 = mpf('1e-8'), mpf('5e-9')
    m1, m2 = meas(t1), meas(t2)
    ext = [2 * b - a for a, b in zip(m1, m2)]   # remove O(tau) term of the ratio
    kr = 4 * C / A - 2 * (B / A) ** 2; kl = 6 * (B / A) ** 2 - mpf(48) / 5 * C / A; kt = 3 * (B / A) ** 2 - mpf(18) / 5 * C / A
    print('trial %d: kappa_r %s vs %s | kappa_l %s vs %s | kappa_t %s vs %s | mid %s vs %s' % (
        trial, mp.nstr(ext[0], 10), mp.nstr(kr, 10), mp.nstr(ext[1], 10), mp.nstr(kl, 10),
        mp.nstr(ext[2], 10), mp.nstr(kt, 10), mp.nstr(ext[3], 10), mp.nstr(2 * B / A, 10)))
