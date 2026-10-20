"""Conditional covariance of the third-point 1-jet (f, f_t, f_s)(y) given the 12 pair numbers
(value, gradient, Hessian at x_S=(0,0) and x_M=(-r,0)), unit planar squared exponential.
Checks the carrier's scales sd_f = |z_s|^3 r^6/sqrt6, sd_t = (z_s^2/2) r^5, sd_s = sqrt(3/2) z_s^2 r^4
(l.121, l.135, l.150) and where they break near the axis."""
import mpmath as mp
mp.mp.dps = 160

def He(n, x):
    if n == 0:
        return mp.mpf(1)
    h0, h1 = mp.mpf(1), x
    for k in range(1, n):
        h0, h1 = h1, x*h1 - k*h0
    return h1

def dK(a, b, ht, hs):
    # d^a/dt^a d^b/ds^b exp(-(t^2+s^2)/2)
    return (-1)**(a+b)*He(a, ht)*He(b, hs)*mp.e**(-(ht**2+hs**2)/2)

def cov(f1, f2):
    (x1, a1), (x2, a2) = f1, f2
    ht, hs = x1[0]-x2[0], x1[1]-x2[1]
    return (-1)**(a2[0]+a2[1])*dK(a1[0]+a2[0], a1[1]+a2[1], ht, hs)

def run(r, zt, zs):
    S = (mp.mpf(0), mp.mpf(0)); M = (-r, mp.mpf(0)); y = (r**2*zt, r**2*zs)
    jets = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    data = [(S, j) for j in jets] + [(M, j) for j in jets]
    tgt = [(y, (0, 0)), (y, (1, 0)), (y, (0, 1))]
    D = mp.matrix(len(data), len(data))
    for i, u in enumerate(data):
        for j, v in enumerate(data):
            D[i, j] = cov(u, v)
    C = mp.matrix(3, len(data))
    for i, u in enumerate(tgt):
        for j, v in enumerate(data):
            C[i, j] = cov(u, v)
    T = mp.matrix(3, 3)
    for i, u in enumerate(tgt):
        for j, v in enumerate(tgt):
            T[i, j] = cov(u, v)
    Sig = T - C*mp.inverse(D)*C.T
    return [mp.sqrt(Sig[i, i]) for i in range(3)]

for (r, zt, zs) in [(mp.mpf('1e-3'), 0, mp.mpf('0.7')), (mp.mpf('1e-4'), 0, mp.mpf('0.7')),
                    (mp.mpf('1e-4'), mp.mpf('0.3'), mp.mpf('0.7')),
                    (mp.mpf('1e-4'), 1, mp.mpf('1e-2')), (mp.mpf('1e-4'), 1, mp.mpf('1e-3'))]:
    sf, st, ss = run(r, mp.mpf(zt), zs)
    print(f'r={mp.nstr(r,3)} z=({mp.nstr(mp.mpf(zt),3)},{mp.nstr(zs,3)}):',
          'sd_f/(|zs|^3 r^6/sqrt6)=', mp.nstr(sf/(abs(zs)**3*r**6/mp.sqrt(6)), 8),
          ' sd_t/((zs^2/2) r^5)=', mp.nstr(st/(zs**2/2*r**5), 8),
          ' sd_s/(sqrt(3/2) zs^2 r^4)=', mp.nstr(ss/(mp.sqrt(mp.mpf(3)/2)*zs**2*r**4), 8))
