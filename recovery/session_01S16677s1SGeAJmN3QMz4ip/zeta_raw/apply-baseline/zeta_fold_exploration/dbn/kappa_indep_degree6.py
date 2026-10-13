"""Independent check (math review) of kappa_r, kappa_l, kappa_t and the midpoint drift on an exact
backward-heat polynomial of degree 6 with A < 0 and nonzero v5, v6 terms; also the complex side
Im z / sqrt(2|tau|) -> 1. Stdout saved as results/dbn/kappa_indep_degree6.out."""
import mpmath as mp
from math import comb, factorial
mp.mp.dps = 60
def v(n, x, tau):
    return sum(mp.mpf(factorial(n))/(factorial(k)*factorial(n-2*k))*(-tau)**k*x**(n-2*k) for k in range(n//2+1))
coef = {2: mp.mpf('-0.7'), 3: mp.mpf('0.45'), 4: mp.mpf('-0.31'), 5: mp.mpf('0.9'), 6: mp.mpf('-1.3')}
A, B, C = coef[2], coef[3], coef[4]
H = lambda x, tau: sum(c*v(n, x, tau) for n, c in coef.items())
# check heat eq
x0, t0 = mp.mpf('0.3'), mp.mpf('0.02')
print('heat residual', mp.nstr(mp.diff(lambda t: H(x0, t), t0) + mp.diff(lambda x: H(x, t0), x0, 2), 5))
kr = 4*C/A - 2*B**2/A**2; kl = 6*B**2/A**2 - 48*C/(5*A); kt = 3*B**2/A**2 - 18*C/(5*A)
for tau in [mp.mpf('1e-3'), mp.mpf('1e-5'), mp.mpf('1e-7')]:
    s = mp.sqrt(2*tau)
    zp = mp.findroot(lambda x: H(x, tau), s); zm = mp.findroot(lambda x: H(x, tau), -s)
    r = zp - zm
    ell = abs(mp.quad(lambda x: H(x, tau), [zm, zp]))
    print('tau', mp.nstr(tau,2),
          'kr_meas', mp.nstr((r**2/(8*tau)-1)/tau, 8), 'pred', mp.nstr(kr, 8),
          '| kl_meas', mp.nstr((ell/(abs(A)*r**3/6)-1)/tau, 8), 'pred', mp.nstr(kl, 8),
          '| kt_meas', mp.nstr((ell/(8*mp.sqrt(2)/3*abs(A)*tau**1.5)-1)/tau, 8), 'pred', mp.nstr(kt, 8),
          '| mid/tau', mp.nstr((zp+zm)/2/tau, 8), 'pred', mp.nstr(2*B/A, 8))
# complex side
tau = mp.mpf('-1e-6')
zc = mp.findroot(lambda x: H(x, tau), 1j*mp.sqrt(-2*tau))
print('Im z / sqrt(2|tau|)', mp.nstr(zc.imag/mp.sqrt(-2*tau), 12), 'Re z/tau', mp.nstr(zc.real/tau, 8))
