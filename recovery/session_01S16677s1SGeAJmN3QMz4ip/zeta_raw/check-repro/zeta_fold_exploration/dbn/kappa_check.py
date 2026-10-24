"""Check second-order fold coefficients on a truncated heat-polynomial solution of dH/dt = -d2H/dx2.
Stdout saved as results/dbn/kappa_check.out."""
from mpmath import mp, mpf, findroot, quad, sqrt
mp.dps = 50
A, B, C, D = mpf(1), mpf('0.3'), mpf('0.2'), mpf('-0.15')
def v(n, x, t):
    # backward heat polynomials: sum_k n!/(k!(n-2k)!) (-t)^k x^(n-2k)
    from math import factorial
    return sum(mpf(factorial(n))/(factorial(k)*factorial(n-2*k))*(-t)**k*x**(n-2*k) for k in range(n//2+1))
H = lambda x, t: A*v(2,x,t)+B*v(3,x,t)+C*v(4,x,t)+D*v(5,x,t)
# verify PDE numerically
x0,t0=mpf('0.37'),mpf('0.11')
from mpmath import diff
print('PDE residual', diff(lambda t: H(x0,t), t0) + diff(lambda x: H(x,t0), x0, 2))
kr = 4*C/A - 2*B**2/A**2; kl = 6*B**2/A**2 - mpf(48)/5*C/A; kt = 3*B**2/A**2 - mpf(18)/5*C/A
for tau in [mpf('1e-2'), mpf('1e-3'), mpf('1e-4')]:
    s = sqrt(2*tau)
    xp = findroot(lambda x: H(x,tau), s + 2*B/A*tau); xm = findroot(lambda x: H(x,tau), -s + 2*B/A*tau)
    r = xp - xm; ell = abs(quad(lambda x: H(x,tau), [xm, xp]))
    print(float(tau), 'kr est', float((r**2/(8*tau)-1)/tau), 'pred', float(kr),
          '| kl est', float((ell/(abs(A)*r**3/6)-1)/tau), 'pred', float(kl),
          '| ktau est', float((ell/(8*sqrt(2)/3*abs(A)*tau**mpf(1.5))-1)/tau), 'pred', float(kt),
          '| mid', float(((xp+xm)/2)/tau), 'pred', float(2*B/A))
