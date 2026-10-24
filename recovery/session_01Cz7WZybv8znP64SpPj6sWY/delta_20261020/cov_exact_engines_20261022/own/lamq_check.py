# Independent re-implementation of one lattice sum (not the archive's code):
# signed vs absolute sum of w(k) k^q over k = j*pi/12, |j| <= J, J = ceil(30/(pi/12)) + 1.
from mpmath import mp, iv, mpf, pi
mp.dps = iv.dps = 40
h = iv.pi / 12
J = int(mp.ceil(mpf(30) / (mp.pi / 12))) + 1
ks = [j * h for j in range(-J, J + 1)]
ws = [iv.exp(-k * k / 2) for k in ks]
Z1 = sum(ws, iv.mpf(0))
print('J =', J)
for q in (6, 7, 8, 9, 11, 12):
    signed = sum((w * k**q for k, w in zip(ks, ws)), iv.mpf(0)) / Z1.a
    absval = sum((w * abs(k)**q for k, w in zip(ks, ws)), iv.mpf(0)) / Z1.a
    print('q=%2d  signed/Z1.a in [%s, %s]   sum|k|^q/Z1.a ~ %s' % (q, mp.nstr(signed.a, 3), mp.nstr(signed.b, 3), mp.nstr(absval.b, 6)))
