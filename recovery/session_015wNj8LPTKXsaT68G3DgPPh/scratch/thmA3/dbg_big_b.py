exec(open('t3.py').read())
import math
F=Fr
r0, r1 = F(31,2**17), F(1,4096)
law=band_law(r0,r1)
for bb in ((F(0),F(1,512)), (F(15,32),F(15,16)), (F(1,4),F(15,32))):
    res=box_bound(law, bb, (F(63,32),F(2)), r0, r1, x0=F(1,64), theta=F(1,32))
    num = float(res['KB']*res['g0']*res['I3'])
    print([float(x) for x in bb], 'total %.6g KB*g0*I3 %.6f zt %.6f ratio %.6f A %.6g B %.4g C %.4g' % (float(res['total']), num, float(res['zt']), num/float(res['zt']), float(res['termA']), float(res['termB']), float(res['termC'])))
# model
from math import erf, sqrt, exp, pi
def m3f(b): return float(m3(F(b)).mid())
def I3f(b): return float(half_gauss_moments(F(b), F(2)**0 * sqrt_up(F(2)), 3)[3].mid())
for bl, bh in ((0, 1/512), (15/32, 15/16), (1/4, 15/32)):
    print('model', bl, bh, exp(-bl*bl/4)*I3f(bh)/(4*sqrt(2*pi))/m3f(bl))
