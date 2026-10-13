"""Zeros of H_0 on [1, 250] against 2*Im(zetazero(k)). Stdout saved as results/dbn/validate_H0.out."""
import mpmath as mp, time
from hflow import HFlow
from zeros import scan
hf = HFlow(dps=80)
t0=time.time()
zs, cs = scan(hf, 0, 1, 250, mp.mpf('0.05'))
print('scan time', time.time()-t0, 'nzeros', len(zs))
mp.mp.dps = 30
for k, z in enumerate(zs[:50], 1):
    g = mp.zetazero(k).imag
    print(k, mp.nstr(z, 25), mp.nstr(2*g, 25), 'diff', mp.nstr(z - 2*g, 3))
