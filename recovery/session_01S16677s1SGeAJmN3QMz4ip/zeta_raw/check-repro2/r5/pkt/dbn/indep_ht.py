"""Independent (third) implementation of H_t via mpmath tanh-sinh quadrature on subintervals.

Written for the math review as a cross-check of dbn_common.py (Gauss-Legendre) and hflow.py
(fixed-point trapezoid rule). Kept as an independent cross-check.

Usage:
    python3 dbn/indep_ht.py <dps> xi      # H_0 against Xi(z/2)/8 at four points
    python3 dbn/indep_ht.py 60 c910       # Newton solve of the (9/10) collision  (~2 min)
    python3 dbn/indep_ht.py 60 c3435      # Newton solve of the (34/35) collision (~7 min)
Stdout of the collision runs is saved as results/dbn/indep_ht_c910.out and indep_ht_c3435.out,
and that of `python3 dbn/indep_ht.py 60 xi` (seconds) as results/dbn/indep_ht_xi.out.
"""
import mpmath as mp, sys, time

mp.mp.dps = 60


def Phi(u):
    s = mp.mpf(0); x = mp.e**(4*u)
    n = 1
    while True:
        term = (2*mp.pi**2*n**4*mp.e**(9*u) - 3*mp.pi*n**2*mp.e**(5*u))*mp.e**(-mp.pi*n**2*x)
        s += term
        if n > 3 and abs(term) < mp.mpf(10)**(-mp.mp.dps-20): break
        n += 1
    return s


U = mp.mpf('1.3')


def pts(z):
    m = max(20, int(U*abs(z)/mp.pi)+1)
    return [U*k/m for k in range(m+1)]


def Hder(t, z, k):
    # k-th z-derivative of H_t(z)
    f = {0: lambda u: mp.cos(z*u), 1: lambda u: -u*mp.sin(z*u), 2: lambda u: -u**2*mp.cos(z*u),
         3: lambda u: u**3*mp.sin(z*u), 4: lambda u: u**4*mp.cos(z*u)}[k]
    return mp.quad(lambda u: mp.e**(t*u*u)*Phi(u)*f(u), pts(z))


def Xi(x):
    s = mp.mpf(0.5) + 1j*x
    return (s*(s-1)/2*mp.pi**(-s/2)*mp.gamma(s/2)*mp.zeta(s)).real


def collide(t, z, it=12):
    for i in range(it):
        H, Hz, Hzz, Hzzz = [Hder(t, z, k) for k in range(4)]
        # unknowns (t,z): F1 = H, F2 = Hz ; dH/dt = -Hzz, dHz/dt = -Hzzz
        J = mp.matrix([[-Hzz, Hz], [-Hzzz, Hzz]])
        d = mp.lu_solve(J, mp.matrix([-H, -Hz]))
        t += d[0]; z += d[1]
        print(i, mp.nstr(t, 30), mp.nstr(z, 30), mp.nstr(abs(d[0])+abs(d[1]), 5), flush=True)
        if abs(d[0]) + abs(d[1]) < mp.mpf(10)**(-mp.mp.dps//2 - 5): break
    H, Hz, Hzz, Hzzz, Hzzzz = [Hder(t, z, k) for k in range(5)]
    A = Hzz/2; B = Hzzz/6; C = Hzzzz/24
    kr = 4*C/A - 2*B**2/A**2; kl = 6*B**2/A**2 - 48*C/(5*A)
    print('t*', mp.nstr(t, 30), 'z0', mp.nstr(z, 30))
    print('A', mp.nstr(A, 12), 'B/A', mp.nstr(B/A, 12), 'C/A', mp.nstr(C/A, 12), 'kappa_r', mp.nstr(kr, 10), 'kappa_l', mp.nstr(kl, 10))
    return t, z


if __name__ == '__main__':
    mp.mp.dps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    what = sys.argv[2] if len(sys.argv) > 2 else 'xi'
    t0 = time.time()
    if what == 'xi':
        for z in [0, 10, mp.mpf('28.27'), 75]:
            h = Hder(0, mp.mpf(z), 0); x = Xi(mp.mpf(z)/2)/8
            print(z, mp.nstr(h, 25), mp.nstr(x, 25), mp.nstr(abs(h/x-1), 3))
    elif what == 'c910':
        collide(mp.mpf('-1.878'), mp.mpf('99.33'))
    elif what == 'c3435':
        collide(mp.mpf('-0.3846'), mp.mpf('223.276'))
    print('time', time.time()-t0)
