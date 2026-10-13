"""Independent re-computation of the double zeros with a finer quadrature (h=1/256, U=1.5, dps=110)
to confirm the digits of t* and x0, plus a direct real-zero count just above/below t*."""
import mpmath as mp
from hflow import HFlow
from zeros import scan
hf = HFlow(dps=110, h=mp.mpf(1) / 256, U=mp.mpf('1.5'))
for tag, tg, xg in [('pair34_35', '-0.3845611166279814148648072883452168328159', '223.2757801856155199135427198048469711889'),
                    ('pair9_10', '-1.877909072972535264766839718127927110556', '99.32846930557575094802276888599949165261')]:
    x, t = mp.mpf(xg), mp.mpf(tg)
    for _ in range(40):
        H, H1, H2, H3 = hf.derivs(x, t, 3)
        dx = mp.lu_solve(mp.matrix([[H1, -H2], [H2, -H3]]), mp.matrix([-H, -H1]))
        x += dx[0]; t += dx[1]
        if abs(dx[0]) + abs(dx[1]) < mp.mpf(10) ** -80: break
    print(tag, 't* =', mp.nstr(t, 45), ' x0 =', mp.nstr(x, 45))
    print('   diff vs tracker t*:', mp.nstr(t - mp.mpf(tg), 3), ' x0:', mp.nstr(x - mp.mpf(xg), 3))
    for dt in ['1e-3', '-1e-3']:
        zs, cs = scan(hf, t + mp.mpf(dt), x - 3, x + 3, mp.mpf('0.002'))
        print('   real zeros of H_t in [x0-3, x0+3] at t = t* %+s:' % dt, [mp.nstr(z, 12) for z in zs])
