"""Exact two-point conditioning for unit squared-exponential field (planar):
conditional sds of f_t(y), f_s(y), f(y) at y = x_S + r^2 z given the 12 pair numbers
(values, gradients, Hessians at S=(0,0) and M=(-r,0)). Compare with carrier l.120, l.135, l.150."""
import mpmath as mp
mp.mp.dps = 120
def He(n, u):
    # probabilists' Hermite
    if n == 0: return mp.mpf(1)
    if n == 1: return u
    a, b = mp.mpf(1), u
    for k in range(1, n):
        a, b = b, u*b - k*a
    return b
def dK(a, b, ht, hs):
    # d^a/dt^a d^b/ds^b exp(-(t^2+s^2)/2) at (ht,hs)
    return ((-1)**a)*He(a, ht)*mp.e**(-ht**2/2) * ((-1)**b)*He(b, hs)*mp.e**(-hs**2/2)
def covpt(a, pa, b, pb):
    return (-1)**(b[0]+b[1]) * dK(a[0]+b[0], a[1]+b[1], pa[0]-pb[0], pa[1]-pb[1])
J2 = [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]
def condsd(r, zt, zs):
    S = (mp.mpf(0), mp.mpf(0)); M = (-r, mp.mpf(0)); y = (r**2*zt, r**2*zs)
    obs = [(d, p) for p in (S, M) for d in J2]
    A = mp.matrix(12, 12)
    for i, (a, pa) in enumerate(obs):
        for j, (b, pb) in enumerate(obs):
            A[i, j] = covpt(a, pa, b, pb)
    Ai = mp.inverse(A)
    res = {}
    for name, tgt in [('f', (0,0)), ('ft', (1,0)), ('fs', (0,1))]:
        c = mp.matrix([covpt(tgt, y, b, pb) for (b, pb) in obs])
        v = covpt(tgt, y, tgt, y) - (c.T*Ai*c)[0]
        res[name] = mp.sqrt(v)
    return res
for (zt, zs) in [(0, 1), (1, 1), (mp.mpf('0.5'), mp.mpf('-0.7'))]:
    for r in [mp.mpf('1e-3'), mp.mpf('1e-4')]:
        R = condsd(r, mp.mpf(zt), mp.mpf(zs))
        zs_ = mp.mpf(zs)
        print('z=(%s,%s) r=%s' % (zt, zs, mp.nstr(r, 2)),
              'sd_t/(r^5 zs^2)=', mp.nstr(R['ft']/(r**5*zs_**2), 10),
              'sd_s/(r^4 zs^2)=', mp.nstr(R['fs']/(r**4*zs_**2), 10),
              'sd_f/(r^6 |zs|^3)=', mp.nstr(R['f']/(r**6*abs(zs_)**3), 10))
print('targets: 1/2, sqrt(3/2)=', mp.nstr(mp.sqrt(1.5), 10), ' 1/sqrt6=', mp.nstr(1/mp.sqrt(6), 10))
