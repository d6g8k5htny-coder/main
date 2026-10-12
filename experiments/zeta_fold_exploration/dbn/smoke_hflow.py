"""Smoke test of hflow.HFlow: H_0 against xi/8, against a finer quadrature, the heat equation and
F' = H. Prints to stdout only, saved as results/dbn/smoke_hflow.out (formerly fold/test_basic.py;
renamed so test discovery skips it)."""
import time, mpmath as mp
from hflow import HFlow, xi
t0=time.time(); hf = HFlow(dps=80, h=mp.mpf(1)/160, U=mp.mpf('1.25')); print('init', time.time()-t0, 'N', hf.N)
hf2 = HFlow(dps=100, h=mp.mpf(1)/240, U=mp.mpf('1.4'))
mp.mp.dps=80
for z in [0, 10, 28.27, 60, 120, 200, 250]:
    t0=time.time(); d = hf.derivs(z, 0, 3); el=time.time()-t0
    d2 = hf2.derivs(z, 0, 3)
    ref = xi(mp.mpf(1)/2 + 1j*mp.mpf(z)/2)/8
    print(z, mp.nstr(d[0],15), mp.nstr(ref.real,15), 'relerr_vs_xi', mp.nstr(abs(d[0]-ref.real)/abs(ref.real),3), 'rel_vs_finer', [mp.nstr(abs(a-b)/abs(d2[0]),3) for a,b in zip(d,d2)], 'time %.3f'%el)
# heat eq check: dt vs -H''
for z,t in [(50,-0.3),(100,-1.0)]:
    d = hf.derivs(z,t,2); print('heat', mp.nstr(hf.dt(z,t),15), mp.nstr(-d[2],15))
# antiderivative check F' = H by finite difference
z=mp.mpf(60); e=mp.mpf(10)**-20
print('F check', mp.nstr((hf.F(z+e,-0.5)-hf.F(z-e,-0.5))/(2*e),15), mp.nstr(hf.H(z,-0.5),15))
