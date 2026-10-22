# Same sum, lattice built at 100 digits (as at kernel import), summed at 40 (as in h5_run).
from mpmath import mp, iv, mpf
mp.dps = iv.dps = 100
h = iv.pi / 12; J = int(mp.ceil(mpf(30) / (mp.pi / 12))) + 1
ks = [j * h for j in range(-J, J + 1)]; ws = [iv.exp(-k * k / 2) for k in ks]; Z1 = sum(ws, iv.mpf(0))
mp.dps = iv.dps = 40
for q in (7, 9, 11):
    s = sum((w * k**q for k, w in zip(ks, ws)), iv.mpf(0)) / Z1.a
    print(q, mp.nstr(s.a, 3), mp.nstr(s.b, 3))
